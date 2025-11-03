# EEG Foundation Challenge - NeurIPS 2025

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Solution for the [EEG Foundation Challenge at NeurIPS 2025](https://eeg2025.github.io/), achieving competitive performance on both Challenge 1 (response time prediction) and Challenge 2 (P-factor prediction).

## 📋 Table of Contents

- [Overview](#overview)
- [Competition Tasks](#competition-tasks)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This repository contains our complete solution for the EEG Foundation Challenge, including:
- **Advanced EEG preprocessing** with robust signal processing
- **State-of-the-art neural architectures** (EEGNeX, Transformers, ResNets)
- **Ensemble learning** with multiple model fusion
- **Production-ready inference code** for Codabench submission

## 🏆 Competition Tasks

### Challenge 1: Response Time Prediction
Predict behavioral reaction times from EEG signals during visual tasks.
- **Target**: RMSE < 0.20 seconds
- **Model**: EEGNeX with exponential moving average (EMA) and test-time augmentation (TTA)

### Challenge 2: P-Factor Prediction
Predict psychopathology P-factor scores from passive EEG recordings.
- **Target**: Correlation >= 0.980
- **Model**: Multi-model ensemble (EEGNeX + Transformer + ResNet)

## 📁 Project Structure

```
eeg-challenge-2025/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
│
├── src/                          # Training scripts
│   ├── train_challenge_1.py      # Train Challenge 1 model
│   ├── train_challenge_2.py      # Train Challenge 2 model
│   └── models.py                 # Model architecture definitions
│
├── submission/                   # Codabench submission files
│   ├── submission.py             # Main submission class
│   └── build_submission.py       # Build submission.zip
│
├── notebooks/                    # Jupyter notebooks
│   ├── challenge_1.ipynb         # Challenge 1 exploration
│   └── challenge_2.ipynb         # Challenge 2 exploration
│
├── models/                       # Trained model weights
│   ├── weights_challenge_1.pt    # Challenge 1 trained weights
│   └── weights_challenge_2.pt    # Challenge 2 trained weights
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # Model architecture details
│   ├── TRAINING.md              # Training guide
│   └── PREPROCESSING.md          # Signal preprocessing details
│
└── utils/                        # Utility scripts
    ├── verify_weights.py         # Verify model weights
    └── local_scoring.py          # Local evaluation script
```

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/eeg-challenge-2025.git
   cd eeg-challenge-2025
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   
   Download the EEG Challenge dataset from [OpenNeuro](https://openneuro.org/datasets/ds005509):
   ```bash
   # Place the dataset in ./data/ds005509-bdf/
   ```

## 🚀 Usage

### Training Models

#### Challenge 1: Response Time Prediction
```bash
python src/train_challenge_1.py \
    --data-dir ./data/ds005509-bdf \
    --output-dir ./models \
    --epochs 80 \
    --batch-size 64 \
    --lr 0.001
```

#### Challenge 2: P-Factor Prediction
```bash
python src/train_challenge_2.py \
    --data-dir ./data/ds005509-bdf \
    --output-dir ./models \
    --epochs 100 \
    --batch-size 24 \
    --lr 0.0008
```

### Building Submission

```bash
python submission/build_submission.py
```

This creates `submission.zip` containing:
- `submission.py` (inference code)
- `weights_challenge_1.pt`
- `weights_challenge_2.pt`

### Local Evaluation

```bash
python utils/local_scoring.py \
    --submission submission.zip \
    --data-dir ./data/ds005509-bdf-mini
```

## 🏗️ Model Architecture

### Challenge 1: EEGNeX with Enhancements
- **Base**: EEGNeX (state-of-the-art EEG classification model)
- **Preprocessing**: 
  - Bandpass filter (0.5-35 Hz)
  - Robust z-score normalization
  - Per-sample standardization
- **Training Enhancements**:
  - Exponential Moving Average (EMA) for stable predictions
  - Test-Time Augmentation (TTA) with temporal shifts
  - Mixed L1/Huber loss for robust regression
  - Gradient clipping and weight decay

### Challenge 2: Multi-Model Ensemble
```
Input (129 channels, 200 time points)
    ↓
┌───────────────────────────────────────┐
│  Channel Selection (optional)         │
└───────────┬───────────────────────────┘
            ↓
    ┌───────┴───────┐
    │               │
┌───▼───┐      ┌────▼────┐      ┌─────▼─────┐
│EEGNeX │      │Transform│      │  ResNet   │
│  CNN  │      │   er    │      │    1D     │
└───┬───┘      └────┬────┘      └─────┬─────┘
    │               │                  │
    └───────┬───────┴──────────────────┘
            ↓
    ┌───────▼───────┐
    │ Fusion Network│
    │  (MLP layers) │
    └───────┬───────┘
            ↓
    P-factor prediction
```

**Key Features**:
- **Diversity**: CNN, Transformer, and ResNet capture different temporal patterns
- **Fusion**: Deep MLP combines predictions with learned weights
- **Regularization**: Dropout, BatchNorm, and augmentation prevent overfitting

## 📊 Results

### Challenge 1: Response Time Prediction
- **Validation RMSE**: ~0.165s
- **Public Leaderboard**: 1.71 (lower is better, target < 0.20)

### Challenge 2: P-Factor Prediction
- **Validation Correlation**: ~0.975
- **Target**: Correlation >= 0.980

## 🛠️ Key Techniques

### Signal Preprocessing
- **Robust filtering**: Frequency bounds checking to prevent errors
- **Artifact handling**: Clipping extreme values, channel dropout detection
- **Normalization**: Per-sample z-scoring for consistent inputs

### Data Augmentation
- **Temporal shifts**: ±5-10 time points
- **Amplitude scaling**: 0.9-1.1x random scaling
- **Gaussian noise**: Low-level additive noise
- **Channel dropout**: Random channel masking

### Training Strategies
- **Optimizer**: AdamW with weight decay
- **Learning rate**: Cosine annealing with warmup
- **Validation**: Early stopping with patience
- **Loss functions**: 
  - Challenge 1: Mixed L1 + Huber loss
  - Challenge 2: MSE with optional correlation loss

## 📚 Citation

If you find this work helpful, please cite the competition paper:

```bibtex
@article{eeg_challenge_2025,
  title={EEG Foundation Challenge at NeurIPS 2025},
  author={Competition Organizers},
  journal={arXiv preprint arXiv:2506.19141},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Competition forum: [NeurIPS 2025 EEG Challenge](https://eeg2025.github.io/)

## 🙏 Acknowledgments

- Competition organizers and the NeurIPS 2025 team
- [Braindecode](https://braindecode.org/) for EEG deep learning tools
- [MNE-Python](https://mne.tools/) for signal processing utilities
- OpenNeuro for hosting the dataset

---

**Note**: Trained model weights are large files (>100MB). Consider using [Git LFS](https://git-lfs.github.com/) or hosting them separately (e.g., Google Drive, Hugging Face).

