from typing import Dict, Optional, Any
import logging
import re
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import easyocr
import os
import cv2
from app.utils.device_utils import get_optimal_device
from app.core.config import settings

class GhanaCardOCR:
    """OCR service for Ghana Card number extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reader = None
        self.initialized = False
        self.device = get_optimal_device()
        
    def initialize(self):
        """Initialize OCR service"""
        if not self.initialized:
            try:
                # Initialize EasyOCR
                self.reader = easyocr.Reader(['en'])
                self.initialized = True
                self.logger.info(f"OCR service initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize OCR service: {str(e)}")
                self.initialized = False
                raise RuntimeError(f"OCR initialization failed: {str(e)}")
    
    def load_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV format"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_bytes))
            # Convert PIL image to numpy array
            image_np = np.array(image)
            # Convert RGB to BGR for OpenCV
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            self.logger.info(f"Processing image of shape: {image_cv.shape}")
            return image_cv
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise
    
    def get_rotations(self, image: np.ndarray) -> list:
        """Get different rotations of the image"""
        return [
            ('original', image),
            ('90_clockwise', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
            ('180', cv2.rotate(image, cv2.ROTATE_180)),
            ('270_clockwise', cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
        ]

    def check_image_quality(self, image: np.ndarray) -> bool:
        """Basic quality check before OCR"""
        try:
            # Check minimum size
            if image.shape[0] < settings.MIN_IMAGE_DIMENSION or image.shape[1] < settings.MIN_IMAGE_DIMENSION:
                return False
            
            # Check if image is too blurry
            laplacian_var = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            return laplacian_var > 100  # Threshold for blur detection
        
        except Exception as e:
            self.logger.error(f"Error checking image quality: {str(e)}")
            return False

    def extract_card_number(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract Ghana Card number with rotation support"""
        if not self.initialized:
            self.logger.warning("OCR service not initialized")
            return {
                'success': False,
                'id_number': None,
                'confidence': 0.0,
                'message': 'OCR service not initialized'
            }
        
        try:
            # Load image
            image = self.load_image(image_bytes)
            if image is None:
                raise Exception("Failed to load image")
            
            # Try all rotations until we find a card number
            pattern = r'GHA-\S+'
            best_result = None
            best_confidence = 0.0
            
            for rotation_name, rotated_image in self.get_rotations(image):
                try:
                    self.logger.debug(f"Trying {rotation_name} orientation")
                    results = self.reader.readtext(rotated_image)
                    
                    for _, text, confidence in results:
                        id_number = re.search(pattern, text)
                        if id_number and confidence > best_confidence:
                            best_confidence = confidence
                            best_result = {
                                'success': True,
                                'id_number': id_number.group(),
                                'confidence': float(confidence),
                                'message': f'Ghana card number found in {rotation_name} orientation'
                            }
                except Exception as e:
                    self.logger.warning(f"Error processing {rotation_name} rotation: {str(e)}")
                    continue
            
            if best_result:
                self.logger.info(f"Found best card number with confidence {best_confidence}")
                return best_result
            
            return {
                'success': False,
                'id_number': None,
                'confidence': 0.0,
                'message': 'No Ghana card number found in any orientation'
            }
            
        except Exception as e:
            self.logger.error(f"Error in OCR extraction: {str(e)}")
            return {
                'success': False,
                'id_number': None,
                'confidence': 0.0,
                'message': f'Error processing image: {str(e)}'
            }