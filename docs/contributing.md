# Contributing Guide for ML System API

## Table of Contents
- [Contributing Guide for ML System API](#contributing-guide-for-ml-system-api)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
- [Install formatters](#install-formatters)
- [Format code](#format-code)
  - [Description](#description)
  - [Type of Change](#type-of-change)
  - [Testing Steps](#testing-steps)
  - [Checklist](#checklist)
- [Run all tests](#run-all-tests)
- [Run with coverage](#run-with-coverage)
- [Run specific test file](#run-specific-test-file)

## Getting Started

1. **Fork and Clone**
   ```
   git clone https://github.com/yourusername/ML-core.git
   cd ML-core

Set Up Development Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

Create Feature Branch
git checkout -b feature/your-feature-name


Development Workflow


While Developing

Write clear, documented code
Follow code standards
Include appropriate tests
Keep changes focused and atomic


Before Submitting

Run all tests
Update documentation
Check code formatting
Write clear commit messages



Code Standards
Python Style Guide

Follow PEP 8
Use type hints
Maximum line length: 88 characters (Black default)
Use meaningful variable names

Code Formatting
# Install formatters
pip install black isort flake8

# Format code
black .
isort .
flake8 .
Example Code Style
from typing import List, Optional

def process_image(
    image_bytes: bytes,
    min_quality: float = 0.8,
    max_size: Optional[int] = None
) -> List[dict]:
    """
    Process an image and return detected features.

    Args:
        image_bytes: Raw image data
        min_quality: Minimum quality threshold (0.0-1.0)
        max_size: Maximum image size in bytes

    Returns:
        List of detected features with confidence scores
        
    Raises:
        ValueError: If image quality is below threshold
    """
    if not image_bytes:
        raise ValueError("Empty image data")
        
    # Process image
    results = []
    return results

Pull Request Template
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing Steps
1. Step 1
2. Step 2

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] All tests passing


Testing Guidelines
Writing Tests
import pytest
from app.services.document_verification import DocumentService

def test_ghana_card_verification():
    # Arrange
    service = DocumentService()
    with open("tests/data/sample_card.jpg", "rb") as f:
        image_bytes = f.read()
        
    # Act
    result = service.verify_ghana_card(image_bytes)
    
    # Assert
    assert result.is_valid
    assert result.confidence > 0.9
    assert len(result.detected_features) >= 3
Running Tests

# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_document_service.py
Documentation
Docstring Format
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: Description of error condition
    """
    pass
API Documentation

Update api.md when adding/modifying endpoints
Include request/response examples
Document error cases

Version Control
Branch Naming

feature/ - New features
bugfix/ - Bug fixes
docs/ - Documentation changes
test/ - Test additions/modifications
refactor/ - Code refactoring

Git Workflow

Update your fork
git remote add upstream [main-repo-url]
git fetch upstream
git rebase upstream/main

Create feature branch
git checkout -b feature/your-feature

Commit changes
git add .
git commit -m "feat(component): description"

Push and create PR
git push origin feature/your-feature


Additional Notes
ML Model Contributions
When contributing ML model improvements:

Include training code
Document model architecture
Provide performance metrics
Include test dataset
Document preprocessing steps