import requests
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
import logging
import io
import os
import time
import traceback
from datetime import datetime
from app.schemas.document import VerificationResponse, ErrorCode, CardInfo
from app.core.config import settings
from io import BytesIO
from functools import lru_cache
import weakref
import re
from app.services.document_verification.ocr_service import ocr_service
import tempfile

class GhanaCardError(Exception):
    """Base exception for Ghana Card verification errors"""
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class DocumentService:
    FEATURES = [
        'ECOWAS Logo',
        'Ghana Coat of Arms',
        'Ghana Flag',
        'Ghana Map',
        'Valid Ghana Card'
    ]

    # Class-level cache to store model instance
    _model_cache = None

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.transform = None
        self.threshold = settings.CARD_CONFIDENCE_THRESHOLD
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Load model using cached method
        self.model = self._get_cached_model()
        
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Register cleanup on program exit
        import atexit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Cleanup temporary files and free memory"""
        try:
            if os.path.exists(settings.OCR_MODEL_PATH):
                import shutil
                shutil.rmtree(settings.OCR_MODEL_PATH)
            self.logger.info("Cleaned up OCR models")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def fetch_and_convert_image(self, image_url: str):
        """Fetch image from URL and convert to proper format"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, timeout=10, headers=headers, verify=False)
            response.raise_for_status()
            
            # Ensure the response is in binary format
            image_bytes = response.content
            
            # Try to open the image directly
            try:
                image = Image.open(BytesIO(image_bytes))
                image = image.convert('RGB')  # Convert to RGB mode if not already
            except Exception as e:
                raise GhanaCardError(
                    f"Unable to open image from URL: {str(e)}",
                    ErrorCode.INVALID_IMAGE_FORMAT
                )

            # Convert PIL image to OpenCV format (BGR)
            try:
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise GhanaCardError(
                    f"Failed to convert image format: {str(e)}",
                    ErrorCode.INVALID_IMAGE_FORMAT
                )

            # Check image dimensions
            if image_cv.shape[0] < settings.MIN_IMAGE_DIMENSION or image_cv.shape[1] < settings.MIN_IMAGE_DIMENSION:
                # Instead of error, resize small images
                scale = max(settings.MIN_IMAGE_DIMENSION/image_cv.shape[0], 
                          settings.MIN_IMAGE_DIMENSION/image_cv.shape[1])
                new_shape = (int(image_cv.shape[1] * scale), int(image_cv.shape[0] * scale))
                image_cv = cv2.resize(image_cv, new_shape)
                self.logger.warning(f"Image was too small, resized to {new_shape}")

            if image_cv.shape[0] > settings.MAX_IMAGE_DIMENSION or image_cv.shape[1] > settings.MAX_IMAGE_DIMENSION:
                # Resize large images
                scale = min(settings.MAX_IMAGE_DIMENSION/image_cv.shape[0], 
                          settings.MAX_IMAGE_DIMENSION/image_cv.shape[1])
                new_shape = (int(image_cv.shape[1] * scale), int(image_cv.shape[0] * scale))
                image_cv = cv2.resize(image_cv, new_shape)
                self.logger.info(f"Image was too large, resized to {new_shape}")

            # Apply transformations for model input
            try:
                transformed = self.transform(image=image_cv)
                image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            except Exception as e:
                raise GhanaCardError(
                    f"Failed to transform image: {str(e)}",
                    ErrorCode.INVALID_IMAGE_FORMAT
                )

            return image_cv, image_tensor, response.content

        except requests.exceptions.RequestException as e:
            raise GhanaCardError(
                f"Failed to fetch image from URL (HTTP error): {str(e)}",
                ErrorCode.URL_ACCESS_ERROR,
                {"url": image_url}
            )
        except GhanaCardError:
            raise
        except Exception as e:
            raise GhanaCardError(
                f"Error processing image: {str(e)}",
                ErrorCode.INVALID_IMAGE_FORMAT,
                {"url": image_url}
            )

    def _get_cached_model(self) -> nn.Module:
        """Get model from cache or load if not cached"""
        if DocumentService._model_cache is None or DocumentService._model_cache() is None:
            self.logger.debug("Loading model from disk - cache miss")
            try:
                model = models.resnet50(weights=None)
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_ftrs, 5)
                )
                
                model_path = settings.GHANA_CARD_MODEL_PATH
                if not model_path.exists():
                    raise GhanaCardError(
                        f"Model file not found: {model_path}",
                        ErrorCode.INTERNAL_ERROR,
                        {"path": str(model_path)}
                    )
                    
                state_dict = torch.load(
                    model_path, 
                    map_location=self.device,
                    weights_only=True
                )
                
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
                
                model = model.to(self.device)
                model.eval()
                DocumentService._model_cache = weakref.ref(model)
                self.logger.info("Model loaded and cached successfully")
                return model
                
            except Exception as e:
                error_details = {"traceback": traceback.format_exc()}
                self.logger.error(f"Error loading model: {str(e)}", extra=error_details)
                raise GhanaCardError(
                    "Failed to load verification model",
                    ErrorCode.INTERNAL_ERROR,
                    error_details
                )
        else:
            self.logger.debug("Using cached model")
            return DocumentService._model_cache()

    def _adjust_confidence(self, is_valid: bool, base_confidence: float) -> float:
        """
        Safely adjust confidence score based on validation result
        """
        try:
            if not is_valid:
                return base_confidence
            
            # Boost confidence by 25% for valid cards, cap at 99.99%
            boosted = min(base_confidence * 1.25, 99.99)
            self.logger.info(f"Adjusted confidence from {base_confidence} to {boosted}")
            return boosted
        except Exception as e:
            self.logger.error(f"Error adjusting confidence: {str(e)}")
            return base_confidence  # Fallback to original confidence if any error occurs

    async def verify_ghana_card(self, card_front_with_selfie: str, card_front: str) -> VerificationResponse:
        start_time = time.time()
        stage_time = time.time()
        
        try:
            self.logger.info("Starting Ghana card verification...")
            
            # Fetch and convert images
            self.logger.info("Fetching and converting images...")
            selfie_cv, selfie_tensor, _ = self.fetch_and_convert_image(card_front_with_selfie)
            front_cv, front_tensor, front_bytes = self.fetch_and_convert_image(card_front)
            self.logger.info(f"Images processed in {time.time() - stage_time:.2f}s")
            
            # Update stage time
            stage_time = time.time()
            
            # Run model inference on validation image
            self.logger.info("Running model inference...")
            with torch.no_grad():
                outputs = self.model(selfie_tensor)
                probabilities = torch.sigmoid(outputs)[0]
            self.logger.info(f"Model inference completed in {time.time() - stage_time:.2f}s")
            
            predictions = (probabilities > self.threshold).cpu().numpy()
            probs = probabilities.cpu().numpy()
            
            detected_features = [
                feature for idx, feature in enumerate(self.FEATURES[:-1])
                if predictions[idx]
            ]
            
            feature_probs = {
                feature: float(prob)
                for feature, prob in zip(self.FEATURES, probs)
            }
            
            num_features_detected = len(detected_features)
            confidence_score = float(np.mean(probs[predictions])) if any(predictions) else 0.0
            
            is_valid = (num_features_detected >= settings.CARD_MIN_FEATURES) or (
                feature_probs.get('Valid Ghana Card', 0) > self.threshold
            )
            
            if not is_valid:
                return VerificationResponse(
                    is_valid=False,
                    confidence=round(confidence_score * 100, 2),
                    detected_features=detected_features,
                    feature_probabilities=feature_probs,
                    num_features_detected=num_features_detected,
                    error_message="Insufficient features detected",
                    error_code=ErrorCode.INSUFFICIENT_FEATURES,
                    processing_time=round(time.time() - start_time, 2)
                )

            # Extract card information using OCR
            stage_time = time.time()
            self.logger.info("Starting OCR processing...")
            try:
                # Create a temporary file with the image bytes
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_file.write(front_bytes)
                    temp_file.flush()  # Ensure all data is written
                    temp_file_path = temp_file.name
                    self.logger.info(f"Created temporary file at: {temp_file_path}")
                
                try:
                    # Process the temporary file
                    ocr_result = ocr_service.extract_card_info(temp_file_path)
                    if ocr_result['success']:
                        self.logger.info(f"Successfully extracted card information in {time.time() - stage_time:.2f}s")
                        card_info = ocr_result['card_info']
                    else:
                        self.logger.warning(f"OCR extraction failed: {ocr_result['message']}")
                        card_info = None
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_file_path)
                        self.logger.debug("Temporary file cleaned up")
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up temporary file: {str(e)}")

            except Exception as e:
                self.logger.error(f"OCR extraction failed: {str(e)}")
                card_info = None
            
            processing_time = round(time.time() - start_time, 2)
            
            # Calculate final confidence
            base_confidence = round(confidence_score * 100, 2)
            adjusted_confidence = self._adjust_confidence(True, base_confidence)

            return VerificationResponse(
                is_valid=True,
                confidence=adjusted_confidence,
                detected_features=detected_features,
                feature_probabilities=feature_probs,
                num_features_detected=num_features_detected,
                card_info=card_info,
                error_message=None if card_info else "Card valid but information extraction failed",
                error_code=None if card_info else ErrorCode.OCR_EXTRACTION_ERROR,
                processing_time=processing_time
            )
            
        except GhanaCardError as e:
            self.logger.error(f"Ghana Card Error: {str(e)}")
            processing_time = round(time.time() - start_time, 2)
            return VerificationResponse(
                is_valid=False,
                confidence=0.0,
                detected_features=[],
                feature_probabilities={},
                num_features_detected=0,
                card_info=None,
                error_message=e.message,
                error_code=e.error_code,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing images: {str(e)}")
            return VerificationResponse(
                is_valid=False,
                confidence=0.0,
                detected_features=[],
                feature_probabilities={},
                num_features_detected=0,
                card_info=None,
                error_message=str(e),
                error_code=ErrorCode.URL_ACCESS_ERROR,
                processing_time=round(time.time() - start_time, 2)
            )

    async def verify_model_files(self) -> Dict[str, bool]:
        """Verify all required model files are present"""
        try:
            base_path = "/ml_models/document_verification"  # Updated to absolute path
            model_files = {
                "ghana_card_modelV2.pth": os.path.join(base_path, "ghana_card_modelV2.pth")
            }
            exists = {name: os.path.exists(path) for name, path in model_files.items()}
            self.logger.info(f"Model files status: {exists}")
            return exists
        except Exception as e:
            self.logger.error(f"Error verifying model files: {str(e)}")
            return {"ghana_card_modelV2.pth": False}

    async def check_health(self) -> Dict[str, Any]:
        """Check health status of the document verification service"""
        try:
            model_status = await self.verify_model_files()
            model_loaded = hasattr(self, 'model') and self.model is not None
            
            return {
                "status": "healthy" if model_loaded and all(model_status.values()) else "unhealthy",
                "model_loaded": model_loaded,
                "models": model_status,
                "device": str(self.device),
                "gpu_available": torch.cuda.is_available()
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_code": "SERVICE_ERROR"
            }

    async def _download_file(self, url: str, dest_path: str):
        """Download a file from URL to destination path"""
        try:
            import aiohttp
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download file: {response.status}")
                    
                    with open(dest_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024 * 1024)  # 1MB chunks
                            if not chunk:
                                break
                            f.write(chunk)
                    
            self.logger.info(f"Successfully downloaded file to {dest_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            raise

    async def download_model_files(self):
        """Download required model files if not present"""
        try:
            model_dir = "/ml_models/document_verification"  # Updated to absolute path
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "ghana_card_modelV2.pth")
            if not os.path.exists(model_path):
                model_url = "https://drive.google.com/uc?id=14PycPjUl3V_csDy7Z_1cxPbGoSEW6e3M"
                await self._download_file(model_url, model_path)
                
        except Exception as e:
            self.logger.error(f"Failed to download model files: {str(e)}")
            raise

# Initialize the document service instance
document_service = DocumentService()

# Export the instance
__all__ = ['document_service']