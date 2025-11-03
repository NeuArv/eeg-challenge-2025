# ##########################################################################
# # Example of submission files
# # ---------------------------
# The zip file needs to be single level depth!
# NO FOLDER
# my_submission.zip
# ├─ submission.py
# ├─ weights_challenge_1.pt
# └─ weights_challenge_2.pt

from pathlib import Path

import torch
import torch.nn as nn
from braindecode.models import EEGNeX


def resolve_path(name: str) -> str:
    """Resolve a file path on Codabench or local environment."""
    search_paths = [
        Path("/app/input/res") / name,
        Path("/app/input") / name,
        Path(name),
        Path(__file__).parent / name,
    ]
    for candidate in search_paths:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find {name} in /app/input/res/, /app/input/, current directory, or submission folder"
    )


class NormalizedModel(nn.Module):
    """Wrap a model to apply per-sample normalization before inference."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        return self.base_model(x)


# Challenge 2 ensemble architecture components
class EEGNetv4(nn.Module):
    """EEGNetv4 for Challenge 2"""
    def __init__(self, n_chans=24, n_outputs=1, n_times=200, F1=12, D=2, F2=32, kernel_length=32, drop_prob=0.3):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.n_chans = n_chans
        self.n_times = n_times
        self.kernel_length = kernel_length
        self.drop_prob = drop_prob
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(drop_prob)
        
        # Separable convolution
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(drop_prob)
        
        # Calculate output size
        out_size = F2 * (n_times // 32)
        self.fc = nn.Linear(out_size, n_outputs)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class TransformerEEGNet(nn.Module):
    """Transformer-based EEG model for Challenge 2"""
    def __init__(self, n_chans=24, n_times=200, n_outputs=1, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, n_times, d_model))
        
        # Input projection
        self.input_proj = nn.Linear(n_chans, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * n_times, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_outputs)
        )
        
    def forward(self, x):
        # x: (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, time, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Flatten and classify
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


class EnhancedEnsembleModelCh2(nn.Module):
    """Enhanced ensemble model for P-factor prediction (Challenge 2)"""
    def __init__(self, selected_indices, n_chans=24, n_times=200, n_outputs=1):
        super().__init__()
        self.selected_indices = selected_indices
        self.eegnex = EEGNeX(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
        self.transformer = TransformerEEGNet(n_chans, n_times, n_outputs)
        self.resnet = self._build_resnet(n_chans, n_times)
        
        # Ensemble fusion (3 models)
        self.fusion = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
    def _build_resnet(self, n_chans, n_times):
        """Build a simple ResNet for EEG"""
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm1d(out_channels)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(out_channels)
                    )
                    
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = torch.relu(out)
                return out
        
        class ResNetEEG(nn.Module):
            def __init__(self, n_chans, n_times):
                super().__init__()
                self.conv1 = nn.Conv1d(n_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm1d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)
                
                self.avgpool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(512, 1)
                
            def _make_layer(self, in_channels, out_channels, blocks, stride=1):
                layers = []
                layers.append(ResidualBlock(in_channels, out_channels, stride))
                for _ in range(1, blocks):
                    layers.append(ResidualBlock(out_channels, out_channels))
                return nn.Sequential(*layers)
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        return ResNetEEG(n_chans, n_times)
        
    def forward(self, x):
        # Select channels if indices provided
        if self.selected_indices is not None and len(self.selected_indices) > 0:
            x = x[:, self.selected_indices, :]
        
        pred1 = self.eegnex(x)
        pred2 = self.transformer(x)
        pred3 = self.resnet(x)
        
        # Combine predictions
        combined = torch.cat([pred1, pred2, pred3], dim=1)
        return self.fusion(combined)


class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def _load_weights(self, model: torch.nn.Module, filename: str) -> torch.nn.Module:
        path = resolve_path(filename)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def get_model_challenge_1(self):
        base_model = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        base_model = self._load_weights(base_model, "weights_challenge_1.pt")
        model = NormalizedModel(base_model).to(self.device)
        model.eval()
        return model

    def get_model_challenge_2(self):
        # Challenge 2 uses ensemble with 129 channels (no channel selection in inference)
        base_model = EnhancedEnsembleModelCh2(
            selected_indices=None,  # Use all 129 channels
            n_chans=129,
            n_times=int(2 * self.sfreq),
            n_outputs=1
        ).to(self.device)
        base_model = self._load_weights(base_model, "weights_challenge_2.pt")
        model = NormalizedModel(base_model).to(self.device)
        model.eval()
        return model


# ##########################################################################
# # How Submission class will be used
# # ---------------------------------
# from submission import Submission
#
# SFREQ = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sub = Submission(SFREQ, DEVICE)
# model_1 = sub.get_model_challenge_1()
# model_1.eval()

# warmup_loader_challenge_1 = DataLoader(HBN_R5_dataset1, batch_size=BATCH_SIZE)
# final_loader_challenge_1 = DataLoader(secret_dataset1, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_1:  # and final_loader later
#         X, y, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X.shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_1.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score1 = compute_score_challenge_1(y_true, y_preds)
# del model_1
# gc.collect()

# model_2 = sub.get_model_challenge_2()
# model_2.eval()

# warmup_loader_challenge_2 = DataLoader(HBN_R5_dataset2, batch_size=BATCH_SIZE)
# final_loader_challenge_2 = DataLoader(secret_dataset2, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_2:  # and final_loader later
#         X, y, crop_inds, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_2.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score2 = compute_score_challenge_2(y_true, y_preds)
# overall_score = compute_leaderboard_score(score1, score2)
