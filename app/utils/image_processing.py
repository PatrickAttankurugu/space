import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from typing import Tuple, Optional

def enhance_image(image: Image.Image) -> Image.Image:
    """Enhance image quality for better face detection"""
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(1.2)

def calculate_image_quality(image: Image.Image) -> Tuple[float, dict]:
    """Calculate image quality metrics"""
    img_array = np.array(image)
    
    # Calculate brightness
    brightness = np.mean(img_array) / 255.0
    
    # Calculate sharpness
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 100
    sharpness = min(sharpness, 1.0)
    
    # Calculate contrast
    contrast = img_array.std() / 255.0
    
    # Overall quality score
    quality_score = np.mean([brightness, sharpness, contrast])
    
    metrics = {
        "brightness": float(brightness),
        "sharpness": float(sharpness),
        "contrast": float(contrast)
    }
    
    return quality_score, metrics 