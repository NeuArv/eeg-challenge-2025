# Contributing Guidelines

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Submitting Changes](#submitting-changes)
6. [Testing](#testing)

---

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

---

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, GPU)
- Error messages and stack traces

**Example**:
```
**Bug**: Training crashes with NaN loss after epoch 10

**Steps to Reproduce**:
1. Run `python src/train_challenge_1.py --epochs 20`
2. Training proceeds normally until epoch 10
3. Loss becomes NaN

**Environment**:
- OS: Ubuntu 22.04
- Python: 3.10.12
- PyTorch: 2.0.1
- GPU: RTX 3090

**Error**:
```
RuntimeError: Loss is NaN
```
```

### Suggesting Enhancements

For feature requests or enhancements:
- Describe the proposed feature clearly
- Explain the motivation and use case
- Provide examples if possible
- Consider implementation complexity

### Improving Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve code comments
- Translate documentation

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/eeg-challenge-2025.git
cd eeg-challenge-2025
```

### 2. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

### 3. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 pytest pytest-cov
```

### 4. Make Changes

Edit code, add tests, update documentation.

### 5. Run Tests

```bash
# Format code
black src/ submission/ utils/

# Check style
flake8 src/ submission/ utils/

# Run tests
pytest tests/ -v --cov
```

---

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted with `isort`
- **Docstrings**: Google style
- **Type hints**: Encouraged but not required

### Code Formatting

Use Black for automatic formatting:

```bash
black src/ submission/ utils/
```

### Example Code

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple


class MyModel(nn.Module):
    """
    Brief description of the model.

    Args:
        n_channels: Number of input channels
        n_outputs: Number of output classes

    Example:
        >>> model = MyModel(n_channels=129, n_outputs=1)
        >>> x = torch.randn(32, 129, 200)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 1])
    """

    def __init__(self, n_channels: int, n_outputs: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.conv = nn.Conv1d(n_channels, 64, kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.conv(x)


def preprocess_data(
    data: torch.Tensor,
    normalize: bool = True,
    clip_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Preprocess EEG data.

    Args:
        data: Input tensor of shape (batch, channels, time)
        normalize: Whether to apply normalization
        clip_range: Optional tuple of (min, max) for clipping

    Returns:
        Preprocessed tensor
    """
    if normalize:
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
        data = (data - mean) / (std + 1e-6)

    if clip_range is not None:
        data = torch.clamp(data, *clip_range)

    return data
```

### Comments

- Use clear, concise comments
- Explain **why**, not **what**
- Update comments when code changes

**Good**:
```python
# Use EMA for more stable predictions during evaluation
ema_model = ModelEMA(model, decay=0.999)
```

**Bad**:
```python
# Create EMA model
ema_model = ModelEMA(model, decay=0.999)
```

---

## Submitting Changes

### 1. Commit Your Changes

Write clear commit messages:

```bash
git add .
git commit -m "Add temporal attention module to Transformer

- Implement scaled dot-product attention over time dimension
- Add positional encoding for temporal relationships
- Update tests to cover new attention mechanism

Closes #42"
```

**Commit Message Format**:
```
<type>: <short description>

<detailed description>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 2. Push to Your Fork

```bash
git push origin feature/my-new-feature
```

### 3. Open a Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Manual testing completed
- [ ] Documentation updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
```

### 4. Code Review

- Address reviewer feedback
- Make requested changes
- Push updates to your branch
- PR will automatically update

---

## Testing

### Writing Tests

Place tests in `tests/` directory:

```
tests/
├── test_models.py
├── test_preprocessing.py
└── test_utils.py
```

**Example Test**:

```python
import torch
import pytest
from src.models import EEGNeX


def test_eegnex_forward():
    """Test EEGNeX forward pass."""
    model = EEGNeX(n_chans=129, n_outputs=1, n_times=200)
    x = torch.randn(32, 129, 200)
    
    output = model(x)
    
    assert output.shape == (32, 1), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_eegnex_with_different_inputs():
    """Test model with various input sizes."""
    model = EEGNeX(n_chans=64, n_outputs=3, n_times=100)
    
    test_cases = [
        (16, 64, 100),
        (1, 64, 100),
        (128, 64, 100),
    ]
    
    for batch_size, n_chans, n_times in test_cases:
        x = torch.randn(batch_size, n_chans, n_times)
        output = model(x)
        assert output.shape == (batch_size, 3)


def test_preprocessing_normalization():
    """Test normalization function."""
    from src.preprocessing import normalize_sample
    
    # Create test data
    x = torch.randn(129, 200) * 10 + 5
    x_norm = normalize_sample(x)
    
    # Check mean and std
    assert torch.abs(x_norm.mean()) < 0.1, "Mean should be near 0"
    assert torch.abs(x_norm.std() - 1.0) < 0.1, "Std should be near 1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_on_gpu():
    """Test model runs on GPU."""
    model = EEGNeX(n_chans=129, n_outputs=1, n_times=200).cuda()
    x = torch.randn(32, 129, 200).cuda()
    
    output = model(x)
    
    assert output.is_cuda, "Output should be on CUDA"
    assert output.shape == (32, 1)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"
```

### Test Coverage

Aim for >80% coverage for new code:

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Areas for Contribution

### High Priority

- [ ] Improve data augmentation strategies
- [ ] Add cross-validation splits
- [ ] Optimize inference speed
- [ ] Add more unit tests
- [ ] Improve documentation with examples

### Medium Priority

- [ ] Add TensorBoard logging
- [ ] Implement learning rate finder
- [ ] Add configuration file support (YAML)
- [ ] Create Docker container for reproducibility
- [ ] Add pre-commit hooks

### Low Priority

- [ ] Add visualization tools for EEG data
- [ ] Create model interpretability tools
- [ ] Add support for other EEG datasets
- [ ] Translate documentation to other languages

---

## Recognition

Contributors will be recognized in:
- README.md acknowledgments section
- CONTRIBUTORS.md file
- Git commit history

---

## Questions?

- Open an issue for questions
- Check existing issues and PRs first
- Be patient and respectful

Thank you for contributing! 🎉

