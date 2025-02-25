# Setup Guide for ML System API

This guide provides detailed instructions for setting up and running the ML System API for both development and production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Development Setup](#development-setup)
- [Production Setup](#production-setup)
- [Verification](#verification)

## Prerequisites

### System Requirements
1. **Hardware Requirements**
   - CPU: Minimum 4 cores recommended
   - RAM: Minimum 8GB (16GB recommended)
   - Storage: 10GB free space
   - GPU: CUDA-capable GPU (optional, for faster processing)

2. **Software Requirements**
   - Python 3.8 or higher
   - Git
   - pip (Python package manager)
   - Tesseract OCR engine

### Installing Tesseract OCR

#### Windows
1. Download installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer
3. Add to PATH:
C:\Program Files\Tesseract-OCR
Copy
#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
macOS
bashCopybrew install tesseract
Installation

Clone the Repository
bashCopygit clone [repository-url]
cd ml_system_api

Create Virtual Environment
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Verify Installation
# Check Python version
python --version

# Verify Tesseract installation
tesseract --version

# Verify GPU availability (if using GPU)
python -c "import torch; print(torch.cuda.is_available())"


Configuration

Environment Variables
Create a .env file in the project root:
# Environment
ENV=development
DEBUG=True

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Security
RATE_LIMIT_PER_MINUTE=60
MAX_FAILED_ATTEMPTS=5
BLOCK_DURATION=300

# Image Settings
MAX_IMAGE_SIZE=10485760  # 10MB in bytes
MIN_IMAGE_DIMENSION=224
MAX_IMAGE_DIMENSION=4096

# ML Settings
USE_CUDA=True
DOCUMENT_VERIFY_THRESHOLD=0.6
FACE_MATCH_THRESHOLD=0.5

# Logging
LOG_LEVEL=INFO

ML Models Setup
# Create models directory
mkdir -p ml_models/document_verification

# Copy model file
cp ml_models/ghana_card_modelV2.pth ml_models/document_verification/

Directory Structure Setup
# Create required directories
mkdir -p logs
mkdir -p docs


Running the Application
Development Mode
# With auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8004

# Without auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 8004
Production Mode
# Using multiple workers
uvicorn app.main:app --host 0.0.0.0 --port 8004 --workers 4 --no-access-log
Using Docker (Optional)
# Build image
docker build -t ml-system-api .

# Run container
docker run -p 8004:8004 ml-system-api
Development Setup

Install Additional Development Tools
pip install black flake8 pytest pytest-cov

Set Up Pre-commit Hooks
pip install pre-commit
pre-commit install

Configure IDE (VS Code recommended)
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black"
}


Production Setup

System Optimization
# Increase file descriptors limit
ulimit -n 65535

# Optimize TCP settings (add to /etc/sysctl.conf)
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

Verification

Check API Status
curl http://localhost:8004/health

Test Face Comparison
python tests/test_face_comparison.py

Test Ghana Card Verification
python tests/test_ghana_card.py

Test License Verification
python tests/test_license.py

Test Spoof Detection
python tests/test_spoof.py

Run All Tests
pytest


Troubleshooting Common Issues

Tesseract Not Found
# Windows: Add to system PATH
# Linux/macOS: Check installation
which tesseract

GPU Not Detected
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"

Permission Issues
# Fix log directory permissions
chmod 755 logs/


Additional Resources

FastAPI Documentation
Tesseract Documentation
PyTorch Installation Guide
PassportEye Documentation


Bug Reports: Create an issue in the repository