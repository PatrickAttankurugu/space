from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import logging
import time
from functools import lru_cache
import aiohttp
from io import BytesIO
from datetime import datetime
from app.core.config import settings
from urllib.parse import urlparse
import re
from facenet_pytorch import MTCNN, InceptionResnetV1
from app.utils.device_utils import get_optimal_device
import os
from app.schemas.spoof import SpoofAnalysisResponse, SpoofType, QualityMetrics
import torch.nn.functional as F

class SpoofDetectionService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.device = get_optimal_device()
            self.logger = logging.getLogger(__name__)
            self.face_detector = MTCNN(device=self.device)
            self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
            self.threshold = 0.5  # Lower threshold for better accuracy
            self.initialized = True
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.deterministic = False
            
            try:
                # Initialize MobileNetV2 for texture analysis
                self.texture_analyzer = self._init_texture_analyzer()
                
                # Image preprocessing
                self.preprocess = transforms.Compose([
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                
                self.logger.info(f"Spoof Detection Service initialized on {self.device}")
                
                if self.device.type == "cuda":
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    self.logger.info(f"GPU Memory Available: {gpu_mem:.2f}GB")
                
            except Exception as e:
                self.logger.error(f"Initialization error: {str(e)}")
                raise RuntimeError(f"Failed to initialize Spoof Detection Service: {str(e)}")

    def _init_texture_analyzer(self):
        """Initialize pretrained MobileNetV2 for texture analysis"""
        model = models.mobilenet_v2(pretrained=True)
        # Use more layers for better texture analysis
        model = nn.Sequential(*list(model.features.children())[:-1])  # Keep more features
        model.to(self.device)
        model.eval()
        return model

    def _compute_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute enhanced image quality metrics"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Enhanced brightness calculation
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 1.5  # Center around 0.5
        brightness_score = max(min(brightness_score, 1.0), 0.0)
        
        # Enhanced sharpness calculation using multiple methods
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_lap = np.var(laplacian) / 100
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness_sobel = (np.var(sobel_x) + np.var(sobel_y)) / 200
        
        sharpness = max(min((sharpness_lap + sharpness_sobel) / 2, 1.0), 0.0)
        
        # Enhanced contrast calculation
        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)
        contrast = (p95 - p5) / 255.0
        contrast = max(min(contrast * 1.5, 1.0), 0.0)
        
        return {
            "brightness": float(brightness_score),
            "sharpness": float(sharpness),
            "contrast": float(contrast)
        }

    async def validate_url(self, url: str) -> bool:
        """Validate if the given URL is properly formatted and accessible"""
        try:
            # Check URL format
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
            
            # Validate URL scheme
            if result.scheme not in ['http', 'https']:
                return False
            
            # Test URL accessibility with GET instead of HEAD
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10, ssl=False) as response:
                    if response.status != 200:
                        return False
                    
                    # More lenient content type checking
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                        # If content-type check fails, try to validate by file extension
                        ext = result.path.lower().split('.')[-1] if '.' in result.path else ''
                        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
                            return False
            
            return True
        except Exception as e:
            self.logger.error(f"URL validation failed: {str(e)}")
            return False

    def _boost_scores(self, score: float, boost_percentage: float = 0.25, cap: float = 0.9999) -> float:
        """Boost score by percentage with a cap"""
        try:
            boosted = score * (1 + boost_percentage)
            return min(boosted, cap)
        except Exception as e:
            self.logger.error(f"Error boosting score: {str(e)}")
            return score

    async def analyze_image(self, image_url: str) -> SpoofAnalysisResponse:
        """Analyze image for spoof detection using combined metrics"""
        try:
            if not await self.validate_url(image_url):
                raise ValueError("Invalid image URL provided")
            
            start_time = time.time()
            
            # Download and process image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise ValueError("Failed to download image")
                    image_data = await response.read()
                    pil_image = Image.open(BytesIO(image_data))
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
            
            # Detect face and get aligned face
            face_tensor = self.face_detector(pil_image)
            if face_tensor is None:
                raise ValueError("No face detected in image")
            
            # Process face for spoof detection
            face_tensor = face_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Get face embedding for liveness
            with torch.no_grad():
                face_embedding = self.facenet(face_tensor)
                texture_features = self.texture_analyzer(face_tensor)
            
            # Compute texture consistency score
            texture_variance = torch.var(texture_features).item()
            texture_score = 1.0 - min(texture_variance / 0.2, 1.0)
            
            # Compute face embedding quality
            embedding_norm = torch.norm(face_embedding).item()
            embedding_score = min(embedding_norm / 8.0, 1.0)
            
            # Convert face tensor to numpy for quality metrics
            face_np = face_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            face_np = ((face_np * 0.5 + 0.5) * 255).astype(np.uint8)
            
            # Compute quality metrics
            quality_metrics = self._compute_quality_metrics(face_np)
            quality_score = np.mean(list(quality_metrics.values()))
            
            # Adjust weights to be more balanced
            spoof_score = (
                0.20 * (1 - texture_score) +      # Increased weight for texture
                0.30 * (1 - embedding_score) +     # Keep embedding weight
                0.30 * (1 - quality_score) +       # Reduced quality weight
                0.20 * (1 - self._calculate_naturalness_score(face_np))  # Reduced naturalness weight
            )
            
            # More balanced threshold and confidence calculation
            is_spoof = spoof_score > 0.75  # Adjusted threshold
            quality_boost = 1.0 + (quality_score * 0.3)  # Reduced quality boost
            confidence = min((1 - spoof_score) * quality_boost * 1.25, 0.95) * 100  # Cap at 95%
            
            # Determine spoof type if detected
            spoof_type = self._determine_spoof_type(quality_metrics, texture_score)
            
            processing_time = time.time() - start_time
            
            # Create and return SpoofAnalysisResponse
            liveness_score = 1 - spoof_score
            if not is_spoof:
                # More balanced boost for high-quality real images
                quality_factor = (quality_metrics['brightness'] + quality_metrics['sharpness'] + quality_metrics['contrast']) / 3
                liveness_boost = 1.0 + (quality_factor * 0.3)  # Reduced quality boost
                liveness_score = min(liveness_score * liveness_boost * 1.25, 0.95)  # Cap at 0.95

            # Before returning the response, apply boosting for non-spoof cases
            if not is_spoof:
                confidence = self._boost_scores(float(confidence) / 100) * 100  # Convert to 0-100 scale
                liveness_score = self._boost_scores(float(liveness_score))
            
            return SpoofAnalysisResponse(
                is_spoof=is_spoof,
                is_deepfake=is_spoof,  # Mirror the spoof detection result
                confidence=float(confidence),
                deepfake_percentage=round((1 - liveness_score) * 100, 2),  # Convert liveness to deepfake percentage
                spoof_type=spoof_type,
                liveness_score=float(liveness_score),
                quality_score=float(quality_score),
                quality_metrics=QualityMetrics(**quality_metrics),
                processing_time=float(processing_time)
            )
            
        except Exception as e:
            self.logger.error(f"Spoof analysis error: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _determine_spoof_type(self, quality_metrics: Dict[str, float], texture_score: float) -> SpoofType:
        """Determine spoof type based on quality and texture metrics"""
        if texture_score < 0.2:
            return SpoofType.MASK
        elif quality_metrics['sharpness'] < 0.2:
            return SpoofType.PRINT
        elif quality_metrics['contrast'] < 0.2:
            return SpoofType.REPLAY
        return SpoofType.NONE

    async def check_health(self) -> Dict:
        """Check the health status of the spoof detection service"""
        try:
            if not hasattr(self, 'initialized'):
                raise RuntimeError("Service not properly initialized")

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "spoof_detection": {
                        "status": "healthy",
                        "service_available": True,
                        "model_loaded": True,
                        "device": self.device,
                        "gpu_available": torch.cuda.is_available()
                    }
                },
                "system": {
                    "device": self.device,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count(),
                    "environment": settings.ENV,
                    "version": settings.VERSION
                }
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "error_code": "SERVICE_ERROR",
                "services": {
                    "spoof_detection": {
                        "status": "unhealthy",
                        "error": str(e),
                        "error_code": "SERVICE_ERROR"
                    }
                },
                "system": {
                    "device": self.device,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count(),
                    "environment": settings.ENV,
                    "version": settings.VERSION
                }
            }

    def _calculate_naturalness_score(self, face_image: np.ndarray) -> float:
        """Calculate naturalness score based on image statistics"""
        try:
            # Convert to YUV color space for better skin tone analysis
            yuv = cv2.cvtColor(face_image, cv2.COLOR_RGB2YUV)
            
            # Calculate color distribution statistics
            y_mean = np.mean(yuv[:,:,0]) / 255.0
            u_std = np.std(yuv[:,:,1]) / 255.0
            v_std = np.std(yuv[:,:,2]) / 255.0
            
            # Calculate texture entropy
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            entropy = self._calculate_entropy(gray)
            
            # Natural images typically have moderate brightness and color variation
            naturalness_score = (
                0.35 * (1 - abs(y_mean - 0.5)) +  # Luminance should be moderate
                0.25 * min(u_std * 5, 1.0) +      # Color variation should be natural
                0.25 * min(v_std * 5, 1.0) +      # Color variation should be natural
                0.15 * min(entropy / 4.0, 1.0)    # Natural images have moderate entropy
            )
            
            return float(naturalness_score)
        except Exception as e:
            self.logger.error(f"Error calculating naturalness score: {str(e)}")
            return 0.8  # Default to high score for error cases

    def _calculate_entropy(self, gray_image: np.ndarray) -> float:
        """Calculate image entropy as a measure of texture complexity"""
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        histogram = histogram.ravel() / histogram.sum()
        non_zero = histogram > 0
        return -np.sum(histogram[non_zero] * np.log2(histogram[non_zero]))

    async def analyze_multiple_images(self, image_urls: List[str], min_images: int = 1, max_images: int = 3) -> SpoofAnalysisResponse:
        """Analyze multiple images for more accurate spoof detection"""
        if not image_urls:
            raise ValueError("No images provided")
        
        if len(image_urls) > max_images:
            raise ValueError(f"Maximum {max_images} images allowed")
        
        if len(image_urls) < min_images:
            raise ValueError(f"Minimum {min_images} images required")
        
        start_time = time.time()
        results = []
        
        try:
            # Process each image
            async with aiohttp.ClientSession() as session:
                for image_url in image_urls:
                    if not await self.validate_url(image_url):
                        raise ValueError(f"Invalid image URL: {image_url}")
                    
                    # Download and process image
                    async with session.get(image_url) as response:
                        if response.status != 200:
                            raise ValueError(f"Failed to download image: {image_url}")
                        image_data = await response.read()
                        pil_image = Image.open(BytesIO(image_data))
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                    
                    # Detect face and get aligned face
                    face_tensor = self.face_detector(pil_image)
                    if face_tensor is None:
                        raise ValueError(f"No face detected in image: {image_url}")
                    
                    # Process face for spoof detection
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    
                    # Get face embedding and texture features
                    with torch.no_grad():
                        face_embedding = self.facenet(face_tensor)
                        texture_features = self.texture_analyzer(face_tensor)
                    
                    # Compute individual scores
                    texture_variance = torch.var(texture_features).item()
                    texture_score = 1.0 - min(texture_variance / 0.2, 1.0)
                    
                    embedding_norm = torch.norm(face_embedding).item()
                    embedding_score = min(embedding_norm / 8.0, 1.0)
                    
                    results.append({
                        'texture_score': texture_score,
                        'embedding_score': embedding_score,
                        'embedding': face_embedding,
                    })
            
            # Cross-validate embeddings between images
            if len(results) > 1:
                embedding_consistency = self._check_embedding_consistency([r['embedding'] for r in results])
            else:
                embedding_consistency = 1.0
            
            # Aggregate scores
            avg_texture_score = sum(r['texture_score'] for r in results) / len(results)
            avg_embedding_score = sum(r['embedding_score'] for r in results) / len(results)
            
            # Final score with weighted components
            final_score = (
                0.35 * avg_texture_score + 
                0.35 * avg_embedding_score + 
                0.30 * embedding_consistency  # Increased weight for consistency between multiple images
            )
            
            is_spoof = final_score < self.threshold
            base_confidence = final_score * 100
            
            # Boost confidence by 25% if not a spoof, cap at 99.99
            confidence = self._boost_scores(float(base_confidence) / 100) * 100 if not is_spoof else base_confidence
            
            # Convert face tensor to numpy for quality metrics
            face_np = face_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            face_np = ((face_np * 0.5 + 0.5) * 255).astype(np.uint8)
            
            # Compute quality metrics
            quality_metrics = self._compute_quality_metrics(face_np)
            quality_score = np.mean(list(quality_metrics.values()))
            
            # Determine spoof type
            spoof_type = self._determine_spoof_type(quality_metrics, avg_texture_score)
            
            # Calculate liveness score
            liveness_score = final_score
            if not is_spoof:
                quality_factor = (quality_metrics['brightness'] + quality_metrics['sharpness'] + quality_metrics['contrast']) / 3
                liveness_boost = 1.0 + (quality_factor * 0.3)
                liveness_score = min(liveness_score * liveness_boost * 1.25, 0.95)
            
            return SpoofAnalysisResponse(
                is_spoof=is_spoof,
                is_deepfake=is_spoof,
                confidence=round(confidence, 2),
                deepfake_percentage=round((1 - liveness_score) * 100, 2),  # Convert liveness to deepfake percentage
                spoof_type=spoof_type,
                liveness_score=float(liveness_score),
                quality_score=float(quality_score),
                quality_metrics=QualityMetrics(**quality_metrics),
                processing_time=round(time.time() - start_time, 2),
                num_images_processed=len(image_urls)
            )
            
        except Exception as e:
            raise ValueError(f"Error processing images: {str(e)}")

    def _check_embedding_consistency(self, embeddings: List[torch.Tensor]) -> float:
        """Check consistency between face embeddings"""
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = F.cosine_similarity(embeddings[i], embeddings[j], dim=1)
                similarities.append(sim.item())
        
        return sum(similarities) / len(similarities) if similarities else 0.0

@lru_cache()
def get_spoof_detector_service() -> SpoofDetectionService:
    """Get or create singleton instance of SpoofDetectionService"""
    return SpoofDetectionService()