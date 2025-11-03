# Quick Start Guide

Get up and running with the EEG Challenge in 15 minutes!

## 🚀 Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/eeg-challenge-2025.git
cd eeg-challenge-2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Download Mini Dataset (2 minutes)

For quick testing, use the mini dataset:

```bash
# Create data directory
mkdir -p data

# The mini dataset is included in the repository
# Located at: startkit/data/ds005509-bdf-mini
# Copy to main data directory
cp -r startkit/data/ds005509-bdf-mini data/
```

Or download the full dataset from [OpenNeuro](https://openneuro.org/datasets/ds005509).

## 🏃 Quick Training (5 minutes)

### Challenge 1: Response Time Prediction

```bash
python src/train_challenge_1.py \
    --data-dir data/ds005509-bdf-mini \
    --epochs 5 \
    --batch-size 32 \
    --mini True
```

Expected output:
```
Loading data from data/ds005509-bdf-mini...
Found 20 subjects, 150 trials
Training...
Epoch 1/5 - Loss: 0.245, Val RMSE: 0.198
Epoch 2/5 - Loss: 0.189, Val RMSE: 0.182
...
Training complete! Final RMSE: 0.175
Model saved to models/weights_challenge_1.pt
```

### Challenge 2: P-Factor Prediction

```bash
python src/train_challenge_2.py \
    --data-dir data/ds005509-bdf-mini \
    --epochs 5 \
    --batch-size 16 \
    --mini True
```

## 📤 Build Submission (3 minutes)

```bash
# Build submission.zip
python submission/build_submission.py

# Verify contents
unzip -l submission.zip
```

Expected output:
```
Archive:  submission.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
    12345  2025-10-30 14:00   submission.py
   256789  2025-10-30 14:00   weights_challenge_1.pt
 49123456  2025-10-30 14:00   weights_challenge_2.pt
---------                     -------
 49392590                     3 files
```

## 🧪 Local Testing

```bash
python utils/local_scoring.py \
    --submission submission.zip \
    --data-dir data/ds005509-bdf-mini
```

## 📊 View Results in Jupyter

```bash
jupyter notebook notebooks/challenge_1.ipynb
```

## ✅ You're Ready!

Upload `submission.zip` to [Codabench](https://www.codabench.org/) and check your scores!

---

## Next Steps

- Read [TRAINING.md](docs/TRAINING.md) for detailed training guide
- Check [ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand the models
- See [PREPROCESSING.md](docs/PREPROCESSING.md) for signal processing details

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'mne'`
- **Solution**: `pip install -r requirements.txt`

**Issue**: `CUDA out of memory`
- **Solution**: Reduce batch size: `--batch-size 16`

**Issue**: Can't find data
- **Solution**: Make sure data is in `data/ds005509-bdf-mini/`

**Issue**: Training is slow
- **Solution**: Increase workers: `--num-workers 4`

For more help, see [docs/TRAINING.md](docs/TRAINING.md#troubleshooting)

