# Training Guide

Complete guide for training models for the EEG Foundation Challenge.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Training Challenge 1](#training-challenge-1)
3. [Training Challenge 2](#training-challenge-2)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Troubleshooting](#troubleshooting)

---

## Data Preparation

### Download Dataset

1. **Get the data from OpenNeuro**:
   ```bash
   # Visit: https://openneuro.org/datasets/ds005509
   # Or use AWS CLI:
   aws s3 sync --no-sign-request \
       s3://openneuro.org/ds005509 \
       ./data/ds005509-bdf/
   ```

2. **Verify data structure**:
   ```
   data/
   └── ds005509-bdf/
       ├── dataset_description.json
       ├── participants.tsv
       ├── sub-NDARXXXXX/
       │   └── eeg/
       │       ├── sub-NDARXXXXX_task-*.bdf
       │       ├── sub-NDARXXXXX_task-*_channels.tsv
       │       ├── sub-NDARXXXXX_task-*_eeg.json
       │       └── sub-NDARXXXXX_task-*_events.tsv
       └── task-*_eeg.json
   ```

### Data Statistics

**Challenge 1: Response Time Prediction**
- **Subjects**: ~500 participants
- **Sessions**: Multiple per subject
- **Tasks**: Visual change detection tasks
- **Total Trials**: ~50,000+
- **EEG Channels**: 129 (128 scalp + 1 reference)
- **Sampling Rate**: 100 Hz
- **Trial Duration**: Variable (typically 2-4 seconds)

**Challenge 2: P-Factor Prediction**
- **Subjects**: ~500 participants
- **Tasks**: Passive viewing (movies), resting state
- **Total Windows**: ~100,000+ (4s windows with 2s stride)
- **EEG Channels**: 129
- **Sampling Rate**: 100 Hz
- **Window Size**: 4 seconds → cropped to 2 seconds

---

## Training Challenge 1

### Quick Start

```bash
python src/train_challenge_1.py \
    --data-dir ./data/ds005509-bdf \
    --output-dir ./models \
    --epochs 80 \
    --batch-size 64 \
    --device cuda
```

### Full Configuration

```bash
python src/train_challenge_1.py \
    --data-dir ./data/ds005509-bdf \
    --output-dir ./models \
    --mini False \
    --epochs 80 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --weight-decay 0.01 \
    --validation-split 0.15 \
    --patience 15 \
    --device cuda \
    --num-workers 4 \
    --seed 42
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 80 | Number of training epochs |
| `--batch-size` | 64 | Batch size (reduce if OOM) |
| `--learning-rate` | 0.001 | Initial learning rate |
| `--weight-decay` | 0.01 | L2 regularization strength |
| `--validation-split` | 0.15 | Fraction of data for validation |
| `--patience` | 15 | Early stopping patience |
| `--ema-decay` | 0.999 | Exponential moving average decay |

### Training Process

1. **Data Loading**:
   - Loads active task trials with response times
   - Filters valid trials (response time > 0, valid EEG)
   - Creates fixed-length windows around response events

2. **Preprocessing**:
   - Bandpass filter: 0.5-35 Hz
   - Per-sample z-score normalization
   - Clipping extreme values

3. **Augmentation** (during training):
   - Random temporal jitter: ±10 time points
   - Amplitude scaling: 0.95-1.05x
   - Gaussian noise: σ=0.01
   - Channel dropout: 1-2 channels

4. **Training Loop**:
   ```python
   for epoch in range(epochs):
       for batch in train_loader:
           # Forward pass
           predictions = model(eeg_data)
           
           # Compute loss (mixed L1 + Huber)
           loss = mixed_loss(predictions, targets)
           
           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
           
           # Update EMA
           ema.update(model)
       
       # Validation with EMA model
       val_rmse = validate(ema_model, val_loader)
       
       # Learning rate scheduling
       scheduler.step()
       
       # Early stopping check
       if val_rmse < best_rmse:
           save_checkpoint()
           patience_counter = 0
       else:
           patience_counter += 1
   ```

5. **Evaluation**:
   - Use EMA model weights
   - Apply test-time augmentation
   - Compute RMSE on validation set

### Expected Results

- **Training Time**: ~2-3 hours on single GPU (RTX 3090)
- **Validation RMSE**: ~0.165 seconds (target: <0.20)
- **Final Model Size**: ~250KB

### Training Curves

Typical training progression:
```
Epoch   Train Loss   Val RMSE    LR
1       0.250        0.235       0.001
10      0.185        0.192       0.001
20      0.168        0.178       0.0008
40      0.155        0.169       0.0005
60      0.149        0.165       0.0002
80      0.145        0.164       0.0001
```

---

## Training Challenge 2

### Quick Start

```bash
python src/train_challenge_2.py \
    --data-dir ./data/ds005509-bdf \
    --output-dir ./models \
    --epochs 100 \
    --batch-size 24 \
    --device cuda
```

### Full Configuration

```bash
python src/train_challenge_2.py \
    --data-dir ./data/ds005509-bdf \
    --output-dir ./models \
    --mini False \
    --epochs 100 \
    --batch-size 24 \
    --learning-rate 0.0008 \
    --weight-decay 0.01 \
    --validation-split 0.12 \
    --patience 20 \
    --device cuda \
    --num-workers 2 \
    --seed 42 \
    --use-full-dataset True \
    --include-additional-tasks False
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 24 | Batch size (ensemble is memory-intensive) |
| `--learning-rate` | 0.0008 | Initial learning rate |
| `--validation-split` | 0.12 | Smaller validation set (more training data) |
| `--patience` | 20 | Longer patience for ensemble |
| `--use-full-dataset` | True | Use all available releases (R2-R10) |
| `--include-additional-tasks` | False | Add active tasks to training |

### Training Process

1. **Data Loading**:
   - Loads passive viewing and resting state tasks
   - Filters subjects with valid P-factor scores
   - Creates 4-second windows with 2-second stride
   - Randomly crops to 2 seconds during training

2. **Preprocessing**:
   - Bandpass filter: 1-40 Hz (wider for passive data)
   - Robust standardization per sample
   - Clipping extreme values

3. **Augmentation** (during training):
   - Random temporal jitter: ±8 time points
   - Amplitude scaling: 0.9-1.1x
   - Gaussian noise: σ=0.015
   - Channel dropout: 1-3 channels

4. **Ensemble Training**:
   ```python
   for epoch in range(epochs):
       for batch in train_loader:
           # Forward pass through all models
           pred_eegnex = ensemble.eegnex(eeg_data)
           pred_transformer = ensemble.transformer(eeg_data)
           pred_resnet = ensemble.resnet(eeg_data)
           
           # Fusion
           combined = torch.cat([pred_eegnex, pred_transformer, pred_resnet], dim=1)
           final_pred = ensemble.fusion(combined)
           
           # MSE loss
           loss = F.mse_loss(final_pred, p_factors)
           
           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
       
       # Validation
       val_correlation = validate(model, val_loader)
       
       # ReduceLROnPlateau
       scheduler.step(val_correlation)
       
       # Early stopping on correlation
       if val_correlation > best_correlation:
           save_checkpoint()
           patience_counter = 0
   ```

5. **Evaluation**:
   - Compute Pearson correlation on validation set
   - Target: ≥0.980 correlation

### Expected Results

- **Training Time**: ~6-8 hours on single GPU (RTX 3090)
- **Validation Correlation**: ~0.975 (target: ≥0.980)
- **Final Model Size**: ~49MB (ensemble is large)

### Training Curves

Typical training progression:
```
Epoch   Train Loss   Val Corr    LR
1       0.850        0.425       0.0008
10      0.520        0.672       0.0008
20      0.385        0.785       0.0008
40      0.285        0.865       0.0006
60      0.215        0.920       0.0004
80      0.180        0.955       0.0002
100     0.165        0.975       0.0001
```

---

## Hyperparameter Tuning

### General Guidelines

1. **Learning Rate**:
   - Start with 1e-3 for single models
   - Reduce to 8e-4 for ensemble (more stable)
   - Use cosine annealing or ReduceLROnPlateau

2. **Batch Size**:
   - Larger is generally better (64-128 for Challenge 1)
   - Reduce if GPU memory is insufficient
   - Ensemble needs smaller batches (24-32)

3. **Regularization**:
   - Weight decay: 0.01 works well
   - Dropout: 0.2-0.3 in classifier heads
   - Augmentation probability: 0.3-0.5

4. **Validation Split**:
   - Challenge 1: 15% (sufficient data)
   - Challenge 2: 12% (need more training data)

### Grid Search Template

```bash
for lr in 0.001 0.0008 0.0005; do
  for wd in 0.01 0.005 0.001; do
    python src/train_challenge_1.py \
      --learning-rate $lr \
      --weight-decay $wd \
      --output-dir ./models/grid_lr${lr}_wd${wd}
  done
done
```

### Monitoring Training

Use the built-in logging or add TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# In training loop:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('RMSE/val', val_rmse, epoch)
writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
```

Then view with:
```bash
tensorboard --logdir runs/
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--batch-size 32` (or 16)
- Reduce number of workers: `--num-workers 1`
- Use gradient accumulation:
  ```python
  accumulation_steps = 2
  loss = loss / accumulation_steps
  loss.backward()
  if (step + 1) % accumulation_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
  ```

#### 2. NaN Loss

**Symptoms**: Loss becomes `nan` during training

**Solutions**:
- Check for inf/nan in data: Add `torch.isfinite(X).all()`
- Reduce learning rate: Try 1e-4 instead of 1e-3
- Use gradient clipping (already implemented)
- Check for division by zero in normalization

#### 3. Model Not Learning

**Symptoms**: Validation metric doesn't improve

**Solutions**:
- Check data loading: Print batch shapes and values
- Verify labels are correct range
- Increase learning rate
- Remove too much regularization
- Check if model is in train mode: `model.train()`

#### 4. Overfitting

**Symptoms**: Train loss << validation loss

**Solutions**:
- Increase weight decay: `--weight-decay 0.02`
- Add more dropout
- Use more aggressive augmentation
- Reduce model capacity
- Increase validation split
- Early stopping (already implemented)

#### 5. Slow Training

**Symptoms**: Very slow epochs

**Solutions**:
- Increase num_workers: `--num-workers 4`
- Use pin_memory: `DataLoader(..., pin_memory=True)`
- Check disk I/O (SSD vs HDD)
- Reduce data loading overhead:
  ```python
  # Cache preprocessed data
  torch.save(dataset, 'cached_dataset.pt')
  ```

---

## Advanced Tips

### 1. Mixed Precision Training

Speed up training with automatic mixed precision:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        predictions = model(X)
        loss = criterion(predictions, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 2. Distributed Training

Train on multiple GPUs:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    src/train_challenge_1.py \
    --batch-size 32  # per GPU
```

### 3. Cross-Validation

More robust evaluation:

```bash
for fold in 0 1 2 3 4; do
    python src/train_challenge_1.py \
        --fold $fold \
        --n-folds 5 \
        --output-dir ./models/fold_$fold
done
```

### 4. Learning Rate Finder

Find optimal learning rate:

```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()  # Shows loss vs LR
```

---

## Reproducibility

### Set All Random Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Save Full Configuration

```python
import yaml

config = {
    'model': {'architecture': 'EEGNeX', ...},
    'training': {'epochs': 80, ...},
    'data': {'path': '...', ...}
}

with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
```

---

## Next Steps

After training:
1. **Verify weights**: `python utils/verify_weights.py models/weights_challenge_1.pt`
2. **Local evaluation**: `python utils/local_scoring.py`
3. **Build submission**: `python submission/build_submission.py`
4. **Submit to Codabench**: Upload `submission.zip`

Good luck! 🚀

