import os
import warnings
# Filter ONNX Runtime warnings about CUDA
warnings.filterwarnings('ignore', category=UserWarning, 
                      message='Specified provider \'CUDAExecutionProvider\' is not in available provider names')

import numpy as np
import cv2
import time
import logging
import traceback
from PIL import Image
import asyncio
from fastapi import HTTPException
import aiohttp
from PIL import ImageEnhance
from insightface.app import FaceAnalysis
from typing import Tuple, Optional, List, Dict, Union
from app.schemas.kyc import ComparisonResponse, ImageQuality, ErrorCode
from app.core.config import settings
import torch
from app.utils.device_utils import get_optimal_device
import insightface

class FaceError(Exception):
    """Base exception for Face Comparison errors"""
    def __init__(self, message: str, error_code: ErrorCode, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class FaceComparisonService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.threshold = float(os.getenv("FACE_MATCH_THRESHOLD", 0.6))
        self.initialized = False
        self.device = get_optimal_device()  # Using the utility function from device_utils.py
        
        try:
            # Initialize CUDA if available
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.set_device(0)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Calculate GPU memory limit (70% of available memory)
                gpu_mem = int(torch.cuda.get_device_properties(0).total_memory * 0.7)
                
                # Configure CUDA provider options
                provider_options = {
                    'device_id': 0,
                    'gpu_mem_limit': gpu_mem,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                
                # Initialize FaceAnalysis with proper provider configuration
                self.app = FaceAnalysis(
                    name="buffalo_l",
                    root="ml_models/face",
                    allowed_modules=['detection', 'recognition'],
                    providers=[
                        ('CUDAExecutionProvider', provider_options),
                        'CPUExecutionProvider'
                    ]
                )
                
                # Prepare with CUDA context
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                
                # Verify CUDA initialization
                test_img = np.zeros((160, 160, 3), dtype=np.uint8)
                _ = self.app.get(test_img)
                
                self.initialized = True
                self.logger.info(f"Face comparison service initialized on CUDA device: {torch.cuda.get_device_name(0)}")
                
                # Log GPU memory information
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"GPU Memory Available: {gpu_mem:.2f}GB")
                
            else:
                # CPU initialization
                self.app = FaceAnalysis(
                    name="buffalo_l",
                    root="ml_models/face",
                    allowed_modules=['detection', 'recognition']
                )
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
                self.initialized = True
                self.logger.info("Face comparison service initialized on CPU")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize face comparison service: {str(e)}")
            raise RuntimeError(f"Face service initialization failed: {str(e)}")

    def _initialize_providers(self):
        """Initialize ONNX Runtime providers with proper configuration"""
        if self.device.type == "cuda":
            try:
                import onnxruntime as ort
                
                # Configure ONNX Runtime session options
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.enable_mem_pattern = True
                sess_options.enable_cpu_mem_arena = True
                
                # Calculate GPU memory limit (70% of available memory)
                gpu_mem = int(torch.cuda.get_device_properties(0).total_memory * 0.7)
                
                # Configure CUDA provider options
                cuda_provider_options = {
                    'device_id': 0,
                    'gpu_mem_limit': gpu_mem,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                
                # Initialize CUDA context
                torch.cuda.init()
                torch.cuda.set_device(0)
                
                # Prepare InsightFace with CUDA context
                self.app.prepare(
                    ctx_id=0,
                    det_size=(640, 640)
                )
                
                # Then separately set the providers for each model
                for model_name, model in self.app.models.items():
                    if hasattr(model, 'session'):
                        model.session.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])
                
                # Verify CUDA is being used
                for model_name, model in self.app.models.items():
                    if hasattr(model, 'session'):
                        providers = model.session.get_providers()
                        if 'CUDAExecutionProvider' in providers:
                            self.logger.info(f"Model {model_name} using CUDA successfully")
                        else:
                            self.logger.warning(f"Model {model_name} not using CUDA")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize CUDA providers: {str(e)}")
                return False
        return False



    async def _process_images(self, image1_url: str, image2_url: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process two images in parallel"""
        try:
            tasks = [
                asyncio.create_task(self.fetch_image(image1_url)),
                asyncio.create_task(self.fetch_image(image2_url))
            ]
            
            img1_cv2, img2_cv2 = await asyncio.gather(*tasks)
            
            img1_cv2 = self.preprocess_image(img1_cv2)
            img2_cv2 = self.preprocess_image(img2_cv2)
            
            return img1_cv2, img2_cv2
            
        except Exception as e:
            self.logger.error(f"Error processing images: {str(e)}")
            raise

    async def fetch_image(self, image_url: str) -> np.ndarray:
        """Fetch image from URL and convert to OpenCV format"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, headers=headers, ssl=False) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        raise FaceError(
                            "Failed to decode image from URL",
                            ErrorCode.INVALID_IMAGE_FORMAT
                        )
                    
                    return image

        except aiohttp.ClientError as e:
            raise FaceError(
                f"Failed to fetch image from URL: {str(e)}",
                ErrorCode.URL_ACCESS_ERROR,
                {"url": image_url}
            )
        except Exception as e:
            raise FaceError(
                f"Error processing image: {str(e)}",
                ErrorCode.INVALID_IMAGE_FORMAT,
                {"url": image_url}
            )

    def get_rotations(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Get different rotations of the image if no face is found"""
        rotations = [('original', image)]
        
        faces = self.app.get(image)
        if not faces:
            rotations.extend([
                ('90_clockwise', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                ('180', cv2.rotate(image, cv2.ROTATE_180)),
                ('90_counterclockwise', cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            ])
        return rotations

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Basic image preprocessing with smart resizing"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image input")
            
            height, width = image.shape[:2]
            
            # Handle small images by upscaling
            if height < settings.MIN_IMAGE_DIMENSION or width < settings.MIN_IMAGE_DIMENSION:
                self.logger.info(f"Image dimensions ({width}x{height}) below minimum, upscaling...")
                scale = max(settings.MIN_IMAGE_DIMENSION/height, 
                           settings.MIN_IMAGE_DIMENSION/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_LANCZOS4)
                self.logger.info(f"Upscaled to {new_width}x{new_height}")
            
            # Handle large images by downscaling
            elif height > settings.MAX_IMAGE_DIMENSION or width > settings.MAX_IMAGE_DIMENSION:
                scale = min(settings.MAX_IMAGE_DIMENSION/height, 
                           settings.MAX_IMAGE_DIMENSION/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
                self.logger.info(f"Downscaled to {new_width}x{new_height}")
            
            # Ensure RGB format
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {str(e)}")
            raise FaceError(
                f"Image preprocessing failed: {str(e)}",
                ErrorCode.INVALID_IMAGE_FORMAT
            )

    def assess_image_quality(self, image: np.ndarray) -> Optional[ImageQuality]:
        """Assess image quality metrics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = image.shape[:2]

            brightness = np.mean(gray)
            contrast = np.std(gray)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            resolution = height * width

            # Quality scoring
            brightness_score = min(brightness / 100, 2 - brightness / 100)
            contrast_score = min(contrast / 30, 1)
            blur_score = min(laplacian_var / 100, 1)
            resolution_score = min(resolution / (50 * 50), 1)

            quality_score = max(
                (brightness_score * 0.2 +
                 contrast_score * 0.2 +
                 blur_score * 0.3 +
                 resolution_score * 0.3),
                0.4
            )

            return ImageQuality(
                overall_quality=round(quality_score, 2),
                width=width,
                height=height
            )
        except Exception as e:
            self.logger.error(f"Error assessing image quality: {str(e)}")
            return None

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between face embeddings"""
        try:
            if self.device.type == "cuda":
                # Convert to torch tensors for GPU computation
                t1 = torch.from_numpy(embedding1).to(self.device)
                t2 = torch.from_numpy(embedding2).to(self.device)
                
                norm1 = torch.linalg.norm(t1)
                norm2 = torch.linalg.norm(t2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0
                    
                similarity = torch.dot(t1, t2) / (norm1 * norm2)
                similarity = (similarity + 1) / 2
                
                return float(similarity.cpu())
            else:
                # CPU computation
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0
                    
                similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
                similarity = (similarity + 1) / 2
                return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    async def compare_faces(self, image1_url: str, image2_url: str) -> ComparisonResponse:
        """Compare faces in two images using InsightFace"""
        start_time = time.time()
        
        if not hasattr(self, 'app') or self.app is None:
            return ComparisonResponse(
                match=False,
                confidence=0.0,
                face_found=False,
                error_message="Face comparison service not properly initialized",
                error_code=ErrorCode.INTERNAL_ERROR,
                processing_time=round(time.time() - start_time, 2)
            )
        
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                
            img1_cv2, img2_cv2 = await self._process_images(image1_url, image2_url)
            
            faces1 = self.app.get(img1_cv2)
            faces2 = self.app.get(img2_cv2)
            
            if not faces1:
                for rotation_name, rotated_img in self.get_rotations(img1_cv2)[1:]:
                    faces1 = self.app.get(rotated_img)
                    if faces1:
                        img1_cv2 = rotated_img
                        break
                        
            if not faces2:
                for rotation_name, rotated_img in self.get_rotations(img2_cv2)[1:]:
                    faces2 = self.app.get(rotated_img)
                    if faces2:
                        img2_cv2 = rotated_img
                        break
            
            if not faces1 or not faces2:
                return ComparisonResponse(
                    match=False,
                    confidence=0.0,
                    face_found=False,
                    error_message="No faces detected in one or both images",
                    error_code=ErrorCode.FACE_NOT_FOUND,
                    processing_time=round(time.time() - start_time, 2)
                )
            
            image1_quality = self.assess_image_quality(img1_cv2)
            image2_quality = self.assess_image_quality(img2_cv2)
            
            similarity = self.calculate_similarity(faces1[0].embedding, faces2[0].embedding)
            base_similarity_score = max(0, min(100, similarity * 100))
            
            is_match = similarity >= self.threshold
            similarity_score = base_similarity_score

            if is_match:
                # Boost similarity score for matches
                similarity_boost = 1.40  # 25% boost
                similarity_score = min(99.99, base_similarity_score * similarity_boost)
                
                # Boost confidence as before
                confidence_boost = 1.40
                confidence = min(99.99, similarity_score * confidence_boost)
            else:
                confidence = similarity_score

            # Apply quality factor to confidence
            if image1_quality and image2_quality:
                quality_factor = (image1_quality.overall_quality + image2_quality.overall_quality) / 2
                confidence = max(0, min(99.99, confidence * quality_factor))

            match_category = self.get_match_category(similarity)
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return ComparisonResponse(
                match=is_match,
                face_found=True,
                similarity_score=round(similarity_score, 2),
                confidence=round(confidence, 2),
                match_category=match_category,
                image1_quality=image1_quality,
                image2_quality=image2_quality,
                processing_time=round(time.time() - start_time, 2)
            )
            
        except Exception as e:
            self.logger.error(f"Face comparison error: {str(e)}\n{traceback.format_exc()}")
            return ComparisonResponse(
                match=False,
                confidence=0.0,
                face_found=False,
                error_message=f"Internal error: {str(e)}",
                error_code=ErrorCode.INTERNAL_ERROR,
                processing_time=round(time.time() - start_time, 2)
            )

    def get_match_category(self, similarity: float) -> str:
        """Get match category based on similarity score"""
        if similarity >= 0.95:
            return "Definite Match - Very High Confidence"
        elif similarity >= 0.85:
            return "Strong Match - High Confidence"
        elif similarity >= 0.75:
            return "Match - Good Confidence"
        elif similarity >= 0.65:
            return "Potential Match - Moderate Confidence"
        else:
            return "No Match - Low Confidence"

    async def check_health(self) -> dict:
        """Check health status of the face comparison service"""
        try:
            if not self.initialized or self.app is None:
                await self.load_model()
            
            test_img = np.zeros((160, 160, 3), dtype=np.uint8)
            try:
                _ = self.app.get(test_img)
                model_working = True
                model_loaded = True
                
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"Model test failed: {str(e)}")
                model_working = False
                model_loaded = False

            return {
                "status": "healthy" if model_working else "unhealthy",
                "model_loaded": model_loaded,
                "model_working": model_working,
                "version": settings.VERSION,
                "device": str(self.device),
                "gpu_available": torch.cuda.is_available() if self.device.type == "cuda" else False
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_code": "SERVICE_ERROR",
                "model_loaded": False,
                "model_working": False,
                "version": settings.VERSION,
                "device": str(self.device),
                "gpu_available": torch.cuda.is_available() if self.device.type == "cuda" else False
            }

    async def validate_models(self) -> bool:
        """Validate that all required models are loaded and working"""
        try:
            health_status = await self.check_health()
            if not health_status["model_loaded"] or not health_status["model_working"]:
                raise ValueError("Models not properly initialized")
            return True
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            raise

    async def compare_faces_in_image(self, image_url: str) -> ComparisonResponse:
        """Compare two most prominent faces in a single image using InsightFace"""
        start_time = time.time()
        
        if not hasattr(self, 'app') or self.app is None:
            return ComparisonResponse(
                match=False,
                confidence=0.0,
                face_found=False,
                error_message="Face comparison service not properly initialized",
                error_code=ErrorCode.INTERNAL_ERROR,
                processing_time=round(time.time() - start_time, 2)
            )
        
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Fetch and process single image
            image = await self.fetch_image(image_url)
            image = self.preprocess_image(image)
            
            # Detect all faces and sort by size (largest first)
            faces = self.app.get(image)
            if faces:
                faces = sorted(faces, 
                             key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), 
                             reverse=True)
            
            # Try different rotations if faces not found
            if len(faces) < 2:
                for rotation_name, rotated_img in self.get_rotations(image)[1:]:
                    rotated_faces = self.app.get(rotated_img)
                    if len(rotated_faces) >= 2:
                        faces = rotated_faces
                        image = rotated_img
                        break
            
            if len(faces) < 2:
                return ComparisonResponse(
                    match=False,
                    confidence=0.0,
                    face_found=False,
                    error_message=f"Need 2 faces, found {len(faces)} faces",
                    error_code=ErrorCode.FACE_NOT_FOUND,
                    processing_time=round(time.time() - start_time, 2)
                )
            
            # Get the two largest faces
            face1, face2 = faces[:2]
            
            # Assess quality of the face regions
            face1_region = image[int(face1.bbox[1]):int(face1.bbox[3]), 
                               int(face1.bbox[0]):int(face1.bbox[2])]
            face2_region = image[int(face2.bbox[1]):int(face2.bbox[3]), 
                               int(face2.bbox[0]):int(face2.bbox[2])]
            
            image1_quality = self.assess_image_quality(face1_region)
            image2_quality = self.assess_image_quality(face2_region)
            
            # Calculate similarity between the two faces
            similarity = self.calculate_similarity(face1.embedding, face2.embedding)
            similarity_score = max(0, min(100, similarity * 100))

            is_match = similarity >= self.threshold

            if is_match:
                # Boost similarity score by 47% for matches
                similarity_boost = 1.47
                similarity_score = min(99.99, similarity_score * similarity_boost)
                
            confidence = similarity_score
            if image1_quality and image2_quality:
                quality_factor = (image1_quality.overall_quality + image2_quality.overall_quality) / 2
                confidence = max(0, min(99.99, confidence * quality_factor))

            if is_match:
                # Use same confidence boost as before
                confidence = min(99.99, confidence * 1.40)

            # Calculate match category based on boosted similarity score
            match_category = self.get_match_category(similarity_score/100)  # Convert back to 0-1 range
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return ComparisonResponse(
                match=is_match,
                face_found=True,
                similarity_score=round(similarity_score, 2),
                confidence=round(confidence, 2),
                match_category=match_category,
                image1_quality=image1_quality,
                image2_quality=image2_quality,
                processing_time=round(time.time() - start_time, 2)
            )
            
        except Exception as e:
            self.logger.error(f"Face comparison error: {str(e)}\n{traceback.format_exc()}")
            return ComparisonResponse(
                match=False,
                confidence=0.0,
                face_found=False,
                error_message=f"Internal error: {str(e)}",
                error_code=ErrorCode.INTERNAL_ERROR,
                processing_time=round(time.time() - start_time, 2)
            )

# Initialize the face comparison service
face_service = FaceComparisonService()

# Export the instance
__all__ = ['face_service']