# Contributing to Multi-Occupant Smart Home Recommender

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- The motivation and use case
- Any relevant examples or references

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**: `pip install -e ".[dev]"`
3. **Make your changes** following the code style guidelines
4. **Add tests** for any new functionality
5. **Run tests**: `pytest tests/`
6. **Update documentation** if needed
7. **Submit a pull request**

## Code Style Guidelines

### Python Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use type hints for function signatures
- Use docstrings for all public functions/classes

```python
def predict_activity(
    features: pd.DataFrame,
    model: MultiResidentGLM,
    resident_id: int = 1
) -> np.ndarray:
    """
    Predict activity for a resident.
    
    Args:
        features: Feature DataFrame with sensor and temporal data
        model: Trained GLM model
        resident_id: Which resident to predict (1 or 2)
        
    Returns:
        Array of predicted activity IDs
        
    Raises:
        ValueError: If resident_id not in {1, 2}
    """
    ...
```

### Code Formatting

Use `black` for code formatting:
```bash
black src/ tests/ experiments/
```

Use `isort` for import sorting:
```bash
isort src/ tests/ experiments/
```

### Linting

Run `flake8` before committing:
```bash
flake8 src/ tests/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_conflict_resolution.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use pytest fixtures for setup

```python
import pytest
from src.conflict_resolution import ConflictResolver

@pytest.fixture
def resolver():
    return ConflictResolver()

def test_detect_noise_conflict(resolver):
    """Test detection of noise conflict between Sleep and TV."""
    result = resolver.detect_and_resolve(11, 12)  # Sleep vs TV
    
    assert result is not None
    conflict, resolution = result
    assert conflict.conflict_type.value == "noise"
    assert resolution.confidence > 0.8
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Short description of function.
    
    Longer description if needed, explaining the purpose
    and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> function_name(1, "test")
        True
    """
```

### Updating Documentation

- Update README.md for user-facing changes
- Update API.md for API changes
- Add docstrings for new functions/classes

## Project Structure

```
src/
├── preprocessing/     # Data preprocessing
├── pattern_mining/    # FP-Growth extension
├── models/            # GLM models
└── conflict_resolution/  # Resolution module
```

When adding new features:
- Place code in the appropriate module
- Add `__init__.py` exports
- Add tests
- Update documentation

## Release Process

1. Update version in `src/__init__.py` and `setup.py`
2. Update CHANGELOG.md
3. Create a release tag
4. Publish to PyPI (maintainers only)

## Questions?

Feel free to open an issue for any questions about contributing!
