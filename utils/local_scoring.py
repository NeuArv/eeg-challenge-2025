# %%
# Imports
import sys
import math
from pathlib import Path
import pickle
import numpy as np
import random
import zipfile

import torch
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score
from braindecode.datasets.base import EEGWindowsDataset, BaseDataset
from braindecode.preprocessing import (
    create_fixed_length_windows,
    Preprocessor,
    preprocess,
    create_windows_from_events,
)
from braindecode.datasets.base import BaseConcatDataset
from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)


# %%
# Constants

SFREQ = 100
BATCH_SIZE = 1
EPOCH_LEN_S = 2.0

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# dataset wrapper


class DatasetWrapper(BaseDataset):
    def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int, target_name="externalizing", seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):  # pyright: ignore[reportIncompatibleMethodOverride]
        X, _, crop_inds = self.dataset[index]

        target = self.dataset.description[self.target_name]
        target = float(target)

        # Randomly crop the signal to the desired length:
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]  # type: ignore

        return X, target, (i_window_in_trial, i_start, i_stop), {}


# %%
# Scoring functions
def nrmse(y_trues, y_preds):
    """Normalized RMSE using difference between max and min values"""
    return rmse(y_trues, y_preds) / y_trues.std()


def score_challenge1(y_trues, y_preds):
    """Returns the score for challenge 1: response time prediction"""
    sc_rmse = rmse(y_trues, y_preds)
    sc_nrmse = nrmse(y_trues, y_preds)
    sc_r2 = -r2_score(y_trues, y_preds)

    print("Challenge 1 Scores:")
    print(f"RMSE: {sc_rmse:.4f}")
    print(f"NRMSE: {sc_nrmse:.4f}  (overall score use normalized RMSE)")
    print(f"for information only, R^2: {sc_r2:.4f} (not used in challenge 1 score)")
    return sc_nrmse


def score_challenge2(y_trues, y_preds):
    """Returns the score for challenge 2: externalizing score prediction"""
    sc_rmse = rmse(y_trues, y_preds)
    sc_nrmse = nrmse(y_trues, y_preds)
    sc_r2 = -r2_score(y_trues, y_preds)

    print("Challenge 2 Scores:")
    print(f"RMSE: {sc_rmse:.4f}")
    print(f"NRMSE: {sc_nrmse:.4f} (overall score use normalized RMSE)")
    print(f"for information R^2: {sc_r2:.4f} (not used in challenge 2 score)")
    return sc_nrmse


def score_overall(score1, score2):
    """Returns the overall score

    The score is 30% of normalized RMSE for challenge 1 and
    70% of the normalized RMSE of challenge 2."""
    sc_overall = 0.3 * score1 + 0.7 * score2
    print("Overall Score:")
    print(f"NRMSE challenge 1 (30%) + NRMSE challenge 2 (70%): {sc_overall:.4f}")
    return sc_overall


# %%
# Program functions
def ingestion(Submission, data_dir, fast_dev_run=False):
    subjects_kwargs = {"subject": "NDARFG568PXZ"} if fast_dev_run else {}

    #################################################################
    # Challenge 1
    # -----------

    print("Loading model on warmup dataset for challenge 1")

    sub = Submission(SFREQ, DEVICE)
    model_1 = sub.get_model_challenge_1()
    model_1.eval()

    print("Loading warmup dataset for challenge 1")
    dataset_1 = EEGChallengeDataset(
        release="R5",
        mini=False,
        query=dict(
            task="contrastChangeDetection",
        ),
        cache_dir=data_dir,  # type: ignore
        **subjects_kwargs,
    )

    # print("Preprocess dataset")
    preprocessors = [
        Preprocessor(
            annotate_trials_with_target,
            apply_on_array=False,
            target_field="rt_from_stimulus",
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset_1, preprocessors, n_jobs=-1)

    # Create 2-second epochs from valid contrast trial starts only
    # print("Create windows")
    SHIFT_AFTER_STIM = 0.5
    WINDOW_LEN = 2.0

    # Keep only recordings that actually contain stimulus anchors
    dataset_2 = keep_only_recordings_with("stimulus_anchor", dataset_1)

    # Create single-interval windows (stim-locked, long enough to include the response)
    dataset_3 = create_windows_from_events(
        dataset_2,
        mapping={"stimulus_anchor": 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),  # +0.5 s
        trial_stop_offset_samples=int(
            (SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ
        ),  # +2.5 s
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )

    # Bring extras (incl. target/RT/correct) into the window metadata
    dataset_3 = add_extras_columns(
        dataset_3,
        dataset_2,
        desc="stimulus_anchor",
        keys=(
            "target",
            "rt_from_stimulus",
            "rt_from_trialstart",
            "stimulus_onset",
            "response_onset",
            "correct",
            "response_type",
        ),
    )

    # print("Wrap the data into a PyTorch-compatible dataset")
    dataloader_1 = DataLoader(
        dataset_3,
        batch_size=1,
        sampler=SequentialSampler(dataset_3),
        shuffle=False,
        drop_last=False,
    )

    print("Evaluate model")
    y_preds = []
    y_trues = []
    with torch.inference_mode():
        for batch in dataloader_1:
            X, y, infos = batch
            X = X.to(dtype=torch.float32, device=DEVICE)
            y = y.to(dtype=torch.float32, device=DEVICE).unsqueeze(1)

            # Forward pass
            y_pred = model_1.forward(X)
            y_preds.append(y_pred.detach().cpu().numpy()[0][0])
            y_trues.append(y.detach().cpu().numpy()[0][0])
    challenge_1_y_preds = np.array(y_preds)
    challenge_1_y_trues = np.squeeze(np.array(y_trues))
    assert len(challenge_1_y_preds) == len(challenge_1_y_trues)

    #################################################################
    # Challenge 2
    # -----------
    print("Loading model on warmup dataset for challenge 2")

    sub = Submission(SFREQ, DEVICE)
    model_2 = sub.get_model_challenge_2()
    model_2.eval()

    print("Loading warmup dataset for challenge 2")
    dataset_4 = EEGChallengeDataset(
        release="R5",
        mini=False,
        query=dict(
            task="contrastChangeDetection",
        ),
        description_fields=["externalizing"],
        cache_dir=data_dir,  # type: ignore
        **subjects_kwargs,
    )

    # print("Preprocess dataset")
    dataset_5 = BaseConcatDataset(
        [
            ds
            for ds in dataset_4.datasets
            if ds.raw.n_times >= 4 * SFREQ and not math.isnan(ds.description["externalizing"])  # type: ignore
        ]
    )
    # print("Create windows")
    dataset_6 = create_fixed_length_windows(
        dataset_5,
        window_size_samples=4 * SFREQ,
        window_stride_samples=2 * SFREQ,
        drop_last_window=True,
    )
    dataset_6 = BaseConcatDataset(
        [DatasetWrapper(ds, crop_size_samples=2 * SFREQ, seed=42) for ds in dataset_6.datasets]  # type: ignore
    )

    # print("Wrap the data into a PyTorch-compatible dataset")
    dataloader_2 = DataLoader(
        dataset_6,
        batch_size=1,
        sampler=SequentialSampler(dataset_6),
        shuffle=False,
        drop_last=False,
    )

    print("Evaluate model")
    y_preds = []
    y_trues = []
    with torch.inference_mode():
        for batch in dataloader_2:
            X, y, crop_inds, infos = batch
            X = X.to(dtype=torch.float32, device=DEVICE)
            y = y.to(dtype=torch.float32, device=DEVICE).unsqueeze(1)

            # Forward pass
            y_pred = model_2.forward(X)
            y_preds.append(y_pred.detach().cpu().numpy()[0][0])
            y_trues.append(y.detach().cpu().numpy()[0][0])

    challenge_2_y_preds = np.array(y_preds)
    challenge_2_y_trues = np.array(y_trues)
    assert len(challenge_2_y_preds) == len(challenge_2_y_trues)

    return {
        "challenge_1": {
            "y_preds": challenge_1_y_preds,
            "y_trues": challenge_1_y_trues,
        },
        "challenge_2": {
            "y_preds": challenge_2_y_preds,
            "y_trues": challenge_2_y_trues,
        },
    }


def scoring(ingestion_output):
    score1 = score_challenge1(
        ingestion_output["challenge_1"]["y_trues"], ingestion_output["challenge_1"]["y_preds"]
    )
    score2 = score_challenge2(
        ingestion_output["challenge_2"]["y_trues"], ingestion_output["challenge_2"]["y_preds"]
    )
    overall = score_overall(score1, score2)

    return {
        "overall": np.round(overall, 5).item(),
        "challenge1": np.round(score1, 5).item(),
        "challenge2": np.round(score2, 5).item(),
    }


# %%
# Main
def main(submission_zip, data_dir, output_dir, fast_dev_run=False):
    if fast_dev_run:
        print("WARNING: fast_dev_run is enabled, results will not be valid!\n")

    # Unzip and Import
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(submission_zip, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted submission files to '{output_dir}'")
    sys.path.insert(0, str(output_dir))
    from submission import Submission

    # Run ingestion
    ingestion_output = ingestion(Submission, data_dir, fast_dev_run)

    # Save predictions
    output_path = Path(output_dir) / "predictions.pickle"
    with open(output_path, "wb") as preds_file:
        pickle.dump(ingestion_output, preds_file)
    print(f"Predictions saved to '{output_path}'")

    # Run scoring
    scoring(ingestion_output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission-zip",
        type=str,
        required=True,
        help="Path to the zipped submission file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Path to the directory where the output predictions file will be saved.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="If true, only evaluate the model on one subject. "
        "Useful for checking if a submission zip is valid and runs without errors.",
    )
    args = parser.parse_args()

    main(args.submission_zip, args.data_dir, args.output_dir, args.fast_dev_run)
