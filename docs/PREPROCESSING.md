# Signal Preprocessing

Detailed documentation of EEG signal preprocessing pipeline.

## Table of Contents
1. [Overview](#overview)
2. [Filtering](#filtering)
3. [Normalization](#normalization)
4. [Artifact Handling](#artifact-handling)
5. [Data Augmentation](#data-augmentation)
6. [Implementation Details](#implementation-details)

---

## Overview

EEG signals are noisy and require careful preprocessing to extract meaningful features. Our pipeline balances artifact removal with signal preservation.

### Pipeline Summary

```
Raw EEG (.bdf files)
    ↓
┌─────────────────────────────────────┐
│ 1. Load with MNE                    │
│    - Read BioSemi .bdf format       │
│    - Extract channel info            │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. Bandpass Filtering                │
│    - Remove DC drift & high-freq     │
│    - Challenge 1: 0.5-35 Hz          │
│    - Challenge 2: 1-40 Hz            │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 3. Artifact Detection                │
│    - Identify bad channels           │
│    - Detect extreme values           │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 4. Normalization                     │
│    - Per-sample z-scoring            │
│    - Clipping outliers               │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 5. Augmentation (training only)      │
│    - Temporal jitter                 │
│    - Amplitude scaling               │
│    - Noise injection                 │
└─────────────┬───────────────────────┘
              ↓
    Preprocessed EEG → Model
```

---

## Filtering

### Bandpass Filtering

**Purpose**: Remove low-frequency drift and high-frequency noise while preserving neural oscillations.

#### Challenge 1: Active Tasks (0.5-35 Hz)

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=0.5, highcut=35, fs=100, order=4):
    """
    Bandpass filter for active task data.
    
    Args:
        data: (n_channels, n_samples) array
        lowcut: Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
    """
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Bounds checking (critical!)
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    if low >= high:
        # Fallback to highpass
        b, a = butter(order, low, btype='high')
    else:
        b, a = butter(order, [low, high], btype='band')
    
    # Zero-phase filtering
    return filtfilt(b, a, data, axis=-1)
```

**Why 0.5-35 Hz?**
- **0.5 Hz highpass**: Removes slow drift without affecting delta band (0.5-4 Hz)
- **35 Hz lowpass**: Removes muscle artifacts and power line noise (50/60 Hz)
- Includes all relevant bands:
  - Delta (0.5-4 Hz): Slow wave activity
  - Theta (4-8 Hz): Memory, attention
  - Alpha (8-13 Hz): Resting state
  - Beta (13-30 Hz): Active thinking, motor control
  - Low Gamma (30-35 Hz): Cognitive processing

#### Challenge 2: Passive Viewing (1-40 Hz)

```python
def bandpass_filter_passive(data, lowcut=1.0, highcut=40, fs=100):
    """Wider band for passive data."""
    # Similar implementation, wider frequency range
```

**Why 1-40 Hz?**
- **1 Hz highpass**: More aggressive drift removal (less concern about delta)
- **40 Hz lowpass**: Include more gamma activity during passive viewing
- Passive tasks have different spectral characteristics

### Notch Filtering (Optional)

For power line noise removal:

```python
from scipy.signal import iirnotch

def notch_filter(data, freq=60, Q=30, fs=100):
    """
    Remove power line noise at 60 Hz (or 50 Hz in Europe).
    
    Args:
        freq: Notch frequency (Hz)
        Q: Quality factor (narrowness of notch)
    """
    if freq >= fs / 2:
        return data  # Skip if invalid
    
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, data, axis=-1)
```

**Note**: We found notch filtering unnecessary after bandpass, but it's available if needed.

---

## Normalization

### Per-Sample Z-Score Normalization

**Purpose**: Ensure consistent amplitude scaling across subjects, sessions, and time.

```python
def normalize_sample(X, eps=1e-6):
    """
    Z-score normalization per sample.
    
    Args:
        X: (n_channels, n_samples) array
        eps: Small constant to prevent division by zero
    
    Returns:
        X_norm: Normalized array
    """
    # Compute statistics across time dimension
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    
    # Normalize
    X_norm = (X - mean) / (std + eps)
    
    return X_norm
```

**Why per-sample?**
- **Subject variability**: Different subjects have different EEG amplitudes
- **Session drift**: Impedance changes over time
- **Task differences**: Different tasks elicit different signal strengths
- **Electrode placement**: Slight differences in cap placement

**Why not global normalization?**
Global statistics computed on training set don't generalize well to:
- New subjects (test set)
- Different sessions
- Codabench evaluation environment

### Robust Normalization (Alternative)

For data with extreme outliers:

```python
def robust_normalize(X, percentile_low=5, percentile_high=95):
    """
    Robust normalization using percentiles.
    
    More resistant to extreme artifacts.
    """
    low = np.percentile(X, percentile_low, axis=-1, keepdims=True)
    high = np.percentile(X, percentile_high, axis=-1, keepdims=True)
    
    X_norm = (X - low) / (high - low + 1e-6)
    X_norm = 2 * X_norm - 1  # Scale to [-1, 1]
    
    return X_norm
```

### Value Clipping

Clip extreme values after normalization:

```python
def clip_values(X, min_val=-10.0, max_val=10.0):
    """
    Clip normalized values to prevent extreme outliers.
    
    Prevents model from being influenced by rare artifacts.
    """
    return np.clip(X, min_val, max_val)
```

**Why clip at ±10?**
- Z-scores beyond ±10 are extremely rare (>99.9999% of normal distribution)
- Likely artifacts rather than true signal
- Prevents gradient explosion during training

---

## Artifact Handling

### Bad Channel Detection

```python
def detect_bad_channels(X, threshold=5.0):
    """
    Detect channels with excessive noise or flatline.
    
    Args:
        X: (n_channels, n_samples) array
        threshold: Z-score threshold for outlier detection
    
    Returns:
        bad_channels: List of channel indices
    """
    # Compute channel statistics
    channel_stds = X.std(axis=-1)
    channel_means = X.mean(axis=-1)
    
    # Detect flatline channels (std too low)
    flatline = channel_stds < 0.1 * np.median(channel_stds)
    
    # Detect noisy channels (std too high)
    z_scores = np.abs((channel_stds - np.median(channel_stds)) / 
                      (channel_stds.std() + 1e-6))
    noisy = z_scores > threshold
    
    bad_channels = np.where(flatline | noisy)[0]
    return bad_channels
```

**Handling bad channels**:
1. **Interpolation** (ideal): Estimate bad channel from neighbors
2. **Zeroing** (simple): Set bad channels to zero
3. **Dropping** (last resort): Exclude trials with too many bad channels

### Artifact Rejection

Reject entire trials with severe artifacts:

```python
def is_trial_good(X, max_amplitude=200, min_amplitude=0.1):
    """
    Check if trial contains severe artifacts.
    
    Args:
        X: (n_channels, n_samples) array in microvolts
    """
    # Check amplitude range
    peak_to_peak = X.max() - X.min()
    if peak_to_peak > max_amplitude:
        return False  # Likely movement artifact
    if peak_to_peak < min_amplitude:
        return False  # Flatline or bad recording
    
    # Check for NaN or Inf
    if not np.isfinite(X).all():
        return False
    
    return True
```

### Eye Blink Removal (Optional)

For frontal channels (Fp1, Fp2):

```python
def detect_blinks(X, threshold=100):
    """
    Simple blink detection on frontal channels.
    
    Returns:
        blink_mask: Boolean array of blink periods
    """
    # Frontal channels are typically first 2
    frontal = X[:2].mean(axis=0)
    
    # Smooth
    from scipy.ndimage import gaussian_filter1d
    frontal_smooth = gaussian_filter1d(frontal, sigma=5)
    
    # Threshold
    blink_mask = np.abs(frontal_smooth) > threshold
    
    return blink_mask
```

---

## Data Augmentation

Applied **only during training** to improve generalization.

### 1. Temporal Jitter

```python
def temporal_jitter(X, max_shift=10):
    """
    Randomly shift signal in time.
    
    Args:
        X: (n_channels, n_samples) array
        max_shift: Maximum shift in samples
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(X, shift, axis=-1)
```

**Rationale**: Neural responses vary slightly in timing across trials. Model should be robust to ±100ms shifts (±10 samples at 100 Hz).

### 2. Amplitude Scaling

```python
def amplitude_scaling(X, scale_range=(0.9, 1.1)):
    """
    Randomly scale amplitude.
    
    Simulates electrode impedance variations.
    """
    scale = np.random.uniform(*scale_range)
    return X * scale
```

**Rationale**: Electrode impedance changes cause amplitude variations. Train model to be scale-invariant.

### 3. Gaussian Noise

```python
def add_gaussian_noise(X, noise_level=0.01):
    """
    Add Gaussian noise.
    
    Args:
        X: Normalized array
        noise_level: Std of noise (relative to normalized signal)
    """
    noise = np.random.randn(*X.shape) * noise_level
    return X + noise
```

**Rationale**: Real EEG always has background noise. Prevent overfitting to clean training data.

### 4. Channel Dropout

```python
def channel_dropout(X, drop_prob=0.2, max_channels=3):
    """
    Randomly zero out channels.
    
    Simulates bad channels during testing.
    """
    n_channels = X.shape[0]
    n_drop = np.random.randint(0, max_channels + 1)
    
    if n_drop > 0 and np.random.rand() < drop_prob:
        drop_indices = np.random.choice(n_channels, n_drop, replace=False)
        X_aug = X.copy()
        X_aug[drop_indices] = 0
        return X_aug
    
    return X
```

**Rationale**: Testing data may have bad channels. Model should handle missing channels gracefully.

### 5. Mixup (Advanced)

```python
def mixup(X1, y1, X2, y2, alpha=0.2):
    """
    Mixup augmentation for EEG.
    
    Reference: Zhang et al., 2018, "mixup: Beyond Empirical Risk Minimization"
    """
    lam = np.random.beta(alpha, alpha)
    X_mix = lam * X1 + (1 - lam) * X2
    y_mix = lam * y1 + (1 - lam) * y2
    return X_mix, y_mix
```

**Rationale**: Encourages model to learn smooth interpolations between examples.

### Augmentation Pipeline

```python
def augment(X, prob=0.4):
    """
    Apply random augmentations with given probability.
    
    Args:
        X: (n_channels, n_samples) array
        prob: Probability of applying each augmentation
    """
    if np.random.rand() < prob:
        X = temporal_jitter(X, max_shift=10)
    
    if np.random.rand() < prob:
        X = amplitude_scaling(X, scale_range=(0.9, 1.1))
    
    if np.random.rand() < prob:
        X = add_gaussian_noise(X, noise_level=0.01)
    
    if np.random.rand() < prob:
        X = channel_dropout(X, drop_prob=0.2, max_channels=2)
    
    return X
```

**Usage**:
```python
# Training
for X, y in train_loader:
    X_aug = augment(X, prob=0.4)
    pred = model(X_aug)
    loss = criterion(pred, y)

# Validation/Testing (no augmentation)
for X, y in val_loader:
    pred = model(X)  # Original data only
```

---

## Implementation Details

### Complete Preprocessing Function

```python
def preprocess_eeg(raw_data, apply_augmentation=False):
    """
    Complete preprocessing pipeline.
    
    Args:
        raw_data: (n_channels, n_samples) array in raw units
        apply_augmentation: Whether to apply data augmentation
    
    Returns:
        preprocessed: (n_channels, n_samples) array ready for model
    """
    # 1. Bandpass filter
    filtered = bandpass_filter(raw_data, lowcut=0.5, highcut=35, fs=100)
    
    # 2. Check for artifacts
    if not is_trial_good(filtered):
        return None  # Reject trial
    
    # 3. Detect and handle bad channels
    bad_channels = detect_bad_channels(filtered)
    if len(bad_channels) > 10:  # Too many bad channels
        return None
    if len(bad_channels) > 0:
        filtered[bad_channels] = 0  # Zero out bad channels
    
    # 4. Normalize
    normalized = normalize_sample(filtered)
    
    # 5. Clip outliers
    clipped = clip_values(normalized, min_val=-10, max_val=10)
    
    # 6. Augmentation (training only)
    if apply_augmentation:
        augmented = augment(clipped, prob=0.4)
        return augmented
    
    return clipped
```

### PyTorch Dataset Integration

```python
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, raw_data, labels, training=True):
        self.raw_data = raw_data
        self.labels = labels
        self.training = training
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        # Load raw data
        X = self.raw_data[idx]  # (n_channels, n_samples)
        y = self.labels[idx]
        
        # Preprocess
        X_processed = preprocess_eeg(X, apply_augmentation=self.training)
        
        if X_processed is None:
            # Return dummy data if preprocessing fails
            return torch.zeros_like(X), torch.tensor(0.0)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X_processed).float()
        y_tensor = torch.tensor(y).float()
        
        return X_tensor, y_tensor
```

---

## Best Practices

### Do's ✅
- Always apply same preprocessing during training and testing
- Normalize per-sample for subject-independence
- Clip extreme values to prevent outliers
- Use zero-phase filtering (filtfilt) to avoid delays
- Check for NaN/Inf values
- Validate preprocessing visually (plot signals)

### Don'ts ❌
- Don't use global statistics from training set
- Don't apply augmentation during validation/testing
- Don't use causal filtering (causes temporal shifts)
- Don't over-filter (lose important information)
- Don't normalize before filtering (affects filter response)
- Don't forget to handle bad channels

---

## Validation

### Visual Inspection

```python
import matplotlib.pyplot as plt

def plot_preprocessing_steps(raw, filtered, normalized):
    """Visualize preprocessing stages."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Raw
    axes[0].plot(raw[0, :1000])
    axes[0].set_title('Raw EEG')
    axes[0].set_ylabel('Amplitude (µV)')
    
    # Filtered
    axes[1].plot(filtered[0, :1000])
    axes[1].set_title('After Bandpass Filter')
    axes[1].set_ylabel('Amplitude (µV)')
    
    # Normalized
    axes[2].plot(normalized[0, :1000])
    axes[2].set_title('After Normalization')
    axes[2].set_ylabel('Z-score')
    axes[2].set_xlabel('Time (samples)')
    
    plt.tight_layout()
    plt.show()
```

### Spectral Analysis

```python
from scipy.signal import welch

def plot_power_spectrum(data, fs=100):
    """Plot power spectral density."""
    freqs, psd = welch(data, fs=fs, nperseg=256)
    
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('EEG Power Spectrum')
    plt.xlim(0, 50)
    plt.grid(True)
    plt.show()
```

---

## References

1. Gramfort, A., et al. (2013). "MEG and EEG data analysis with MNE-Python." *Frontiers in Neuroscience*
2. Delorme, A., & Makeig, S. (2004). "EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics." *Journal of Neuroscience Methods*
3. Bigdely-Shamlo, N., et al. (2015). "The PREP pipeline: standardized preprocessing for large-scale EEG analysis." *Frontiers in Neuroinformatics*

