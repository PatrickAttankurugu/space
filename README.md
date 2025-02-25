# ML System API for KYC and Document Verification


## Table of Contents
- [ML System API for KYC and Document Verification](#ml-system-api-for-kyc-and-document-verification)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [Documentation](#documentation)
  - [Prerequisites](#prerequisites)
    - [System Requirements](#system-requirements)
  - [Quick Start](#quick-start)
    - [Windows (CMD)](#windows-cmd)
  - [ML Models](#ml-models)
  - [GPU Support](#gpu-support)

## Overview

This system provides a REST API for:
- Face comparison and matching between two images
- Ghana national ID card validation using machine learning
- MRZ (Machine Readable Zone) data extraction from ID cards

### Key Features
- ML-based Ghana card feature detection
- Face comparison with confidence scores
- MRZ data extraction and formatting
- Comprehensive error handling
- Request rate limiting
- Environment-based configurations

## Documentation

Detailed documentation is available in the `docs` folder:
- [API Documentation](docs/api.md)
- [Setup Guide](docs/setup.md)
- [Contributing Guidelines](docs/contributing.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- Tesseract OCR engine
- Docker and Docker Compose
- 16GB RAM minimum
- 50GB free disk space

## Quick Start

### Windows (CMD)
1. Clone the repository:

```bash
git clone https://github.com/Agregar01/ML-core.git
cd ML-core
```

2. Ensure system requirements:
   - At least 16GB RAM
   - 50GB free disk space
   - NVIDIA GPU (recommended)
   - Docker and docker-compose installed

3. Run the automated setup:
```bash
sudo ./scripts/run.sh
```

That's it! The system will be automatically set up, built, and deployed.

For manual deployment or troubleshooting, see docs/setup.md

## ML Models
- Ghana Card Verification Model
- Face Comparison Model
- Spoof Detection Model

## GPU Support
NVIDIA GPU with CUDA support is recommended for optimal performance.
CPU-only mode is available but will be significantly slower.

