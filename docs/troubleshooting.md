# Troubleshooting Guide for ML System API

## Table of Contents
- [Troubleshooting Guide for ML System API](#troubleshooting-guide-for-ml-system-api)
  - [Table of Contents](#table-of-contents)
  - [Common Issues](#common-issues)
    - [Installation Issues](#installation-issues)
      - [Problem: Tesseract Not Found](#problem-tesseract-not-found)
- [Windows](#windows)
- [Linux/macOS](#linuxmacos)
- [Check CPU usage](#check-cpu-usage)
- [Check memory](#check-memory)
- [Check disk I/O](#check-disk-io)
- [Tail application logs](#tail-application-logs)
- [Search for errors](#search-for-errors)
- [Monitor real-time errors](#monitor-real-time-errors)

## Common Issues

### Installation Issues

#### Problem: Tesseract Not Found
```bash
Error: tesseract is not installed or not found in PATH
Solutions:

Check installation:
# Windows
echo %PATH%  # Check if Tesseract path is included

# Linux/macOS
which tesseract

Manually set Tesseract path in code:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


Face Comparison Issues
Problem: No Face Detected
{
    "match": false,
    "face_found": false,
    "error_message": "No faces detected in image"
}
Solutions:

Check image quality:

Ensure good lighting
Face should be clearly visible
Minimal obstruction
Proper orientation


Check image size:
from PIL import Image
img = Image.open('image.jpg')
print(f"Image size: {img.size}")

Verify image format:
print(f"Image format: {img.format}")
print(f"Image mode: {img.mode}")


Ghana Card Verification Issues
Problem: Invalid Card Features
{
    "is_valid": false,
    "error_message": "Insufficient features detected"
}
Solutions:

Check image alignment:

Card should be properly aligned
All corners visible
No cropping of important features


Verify image quality:
def check_image_quality(image_path):
    img = Image.open(image_path)
    if img.size[0] < 1000 or img.size[1] < 1000:
        print("Warning: Image resolution might be too low")
    # Add more quality checks


MRZ Reading Issues
Problem: MRZ Not Readable
{
    "mrz_data": null,
    "error_message": "Could not read MRZ data"
}
Solutions:

Image Enhancement:
import cv2
import numpy as np

def enhance_mrz_image(image_bytes):
    # Convert to grayscale
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save enhanced image
    cv2.imwrite('enhanced_mrz.jpg', binary)

Check image orientation:

MRZ should be horizontal
Text should be clear and readable
No glare or shadows



API Issues
Problem: Authentication Failed
{
    "detail": "Could not validate credentials"
}
Solutions:

Check API key format:
import re

def validate_api_key(api_key: str) -> bool:
    pattern = r'^(test|live)_[a-zA-Z0-9]{32}$'
    return bool(re.match(pattern, api_key))

Verify request headers:
headers = {
    'X-API-Key': 'your_api_key',
    'Content-Type': 'application/json'
}


Problem: Rate Limit Exceeded
{
    "detail": "Rate limit exceeded"
}
Solutions:

Implement rate tracking:
from collections import deque
from time import time

class RateTracker:
    def __init__(self, limit=60, window=60):
        self.requests = deque()
        self.limit = limit
        self.window = window
        
    def can_make_request(self):
        now = time()
        while self.requests and now - self.requests[0] >= self.window:
            self.requests.popleft()
        return len(self.requests) < self.limit


Performance Issues
Problem: Slow Response Times
Solutions:

Check system resources:
# Check CPU usage
top

# Check memory
free -h

# Check disk I/O
iostat

Profile code:
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    # Your function here
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()


Error Codes and Solutions

| Error Code            | Description                       | Solution              |
|-----------------------|-----------------------------------|------------------------|
| INVALID_IMAGE_FORMAT  | Unsupported image format         | Convert to JPEG/PNG    |
| LOW_QUALITY_IMAGE     | Image quality below threshold    | Improve image quality  |
| INSUFFICIENT_FEATURES | Not enough card features detected| Retake card photo      |
| MRZ_NOT_READABLE      | Cannot read MRZ data             | Enhance image or retake|
| INTERNAL_ERROR        | Server-side error                | Check logs and retry   |


Logging and Debugging
Enable Debug Logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='debug.log'
)
Check Application Logs
# Tail application logs
tail -f logs/app.log

# Search for errors
grep "ERROR" logs/app.log

# Monitor real-time errors
tail -f logs/app.log | grep "ERROR"
System Checks
Memory Usage
import psutil

def check_system_resources():
    # CPU Usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory Usage
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    
    # Disk Usage
    disk = psutil.disk_usage('/')
    disk_used_gb = disk.used / (1024 ** 3)
    
    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory_used_gb,
        'disk_used_gb': disk_used_gb
    }
Database Connections
async def check_db_connection():
    try:
        await db.connect()
        return True
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        return False
GPU Status
import torch

def check_gpu_status():
    if torch.cuda.is_available():
        return {
            'available': True,
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated(0),
            'memory_cached': torch.cuda.memory_cached(0)
        }
    return {'available': False}
Support and Contact
If you're still experiencing issues after trying these solutions:

Check our GitHub Issues
Join our Discord Community
Contact technical support: support@example.com
Review API documentation: API Docs

Contributing to Troubleshooting
If you've solved an issue not documented here:

Fork the repository
Add your solution to this guide
Submit a pull request

Help us make this guide more comprehensive!