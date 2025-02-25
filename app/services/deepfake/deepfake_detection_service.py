from typing import Dict, Union, Optional, List
import os
import shutil
import logging
import asyncio
import aiofiles
import concurrent.futures
from functools import lru_cache
import torch
from deepfake_detector import DeepFakeDetector
from fastapi import UploadFile, HTTPException
import time
from datetime import datetime
import hashlib
import gdown
import aiohttp

@lru_cache()
def get_detector_service(weights_dir: str = "ml_models/deepfake") -> 'DeepFakeDetectionService':
    return DeepFakeDetectionService(weights_dir=weights_dir)

class DeepFakeDetectionService:
    _instance = None
    _video_cache: Dict[str, Dict] = {}
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, weights_dir: str = "ml_models/deepfake", logger: Optional[logging.Logger] = None):
        """Initialize the DeepFake Detection Service using Singleton pattern"""
        if not hasattr(self, 'initialized'):
            try:
                self.logger = logger or logging.getLogger(__name__)
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.backends.cudnn.benchmark = True
                
                self.weights_dir = weights_dir
                self.weights = {
                    'final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36': 'https://drive.google.com/uc?id=1Q8EDSx1jOFx4SGv90YkEVeVnksADjHcm',
                    'final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19': 'https://drive.google.com/uc?id=1ypnKmX7NvNfo6RYcOWZehEDQEHQScs1O',
                    'final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31': 'https://drive.google.com/uc?id=1M_VRMvLjC3WLgMjH9eIszC5x7wbSG1YR'
                }
                
                # Ensure models are downloaded before initializing detector
                self._ensure_models()
                
                self.detector = DeepFakeDetector(weights_dir=weights_dir)
                self.temp_dir = "temp_videos"
                self.cache_duration = 3600
                
                os.makedirs(self.temp_dir, exist_ok=True)
                self.initialized = True
                self.logger.info(f"DeepFake Detection Service initialized on {self.device}")
            except Exception as e:
                self.logger.error(f"Initialization error: {str(e)}")
                raise RuntimeError(f"Failed to initialize DeepFake Detection Service: {str(e)}")

    async def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()

    async def analyze_video(
        self, 
        video_url: str
    ) -> Dict[str, Union[bool, float, str]]:
        """Analyze a video from URL for deepfake detection"""
        
        # Calculate hash for cache key
        cache_key = hashlib.md5(video_url.encode()).hexdigest()
        
        # Check cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            self.logger.info(f"Cache hit for video URL: {video_url}")
            return cached_result
        
        temp_path = os.path.join(self.temp_dir, f"temp_{cache_key[:10]}.mp4")
        
        try:
            # Download video
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail="Failed to download video from URL"
                        )
                    content = await response.read()
                    
            # Save temporarily
            async with aiofiles.open(temp_path, 'wb') as out_file:
                await out_file.write(content)
            
            # Process video
            start_time = time.time()
            result = await self._process_video(temp_path)
            processing_time = time.time() - start_time
            
            enhanced_result = {
                "video_url": video_url,
                "is_fake": result["is_fake"],
                "confidence": result["confidence"],
                "processing_time": processing_time,
                "status": "success"
            }
            
            # Cache the result
            self._add_to_cache(cache_key, enhanced_result)
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            await self._cleanup_temp_file(temp_path)

    async def _process_video(self, video_path: str, generate_report: bool) -> Dict[str, Union[bool, float]]:
        """Process video using ThreadPoolExecutor"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                self.detector.detect,
                video_path,
                generate_report
            )
        return result

    @lru_cache(maxsize=128)
    def _generate_confidence_message(self, confidence: float) -> str:
        """Generate a cached confidence message"""
        if confidence >= 95:
            return "Very high confidence in prediction"
        elif confidence >= 80:
            return "High confidence in prediction"
        elif confidence >= 60:
            return "Moderate confidence in prediction"
        else:
            return "Low confidence in prediction"

    def _add_to_cache(self, key: str, result: Dict) -> None:
        """Add result to cache with timestamp"""
        self._video_cache[key] = {
            'result': result,
            'timestamp': time.time()
        }

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get result from cache if not expired"""
        if key in self._video_cache:
            cache_entry = self._video_cache[key]
            if time.time() - cache_entry['timestamp'] < self.cache_duration:
                return cache_entry['result']
            del self._video_cache[key]
        return None

    async def _cleanup_temp_file(self, file_path: str) -> None:
        """Asynchronously cleanup temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            self.logger.error(f"Error cleaning up temp file: {str(e)}")

    async def get_detection_stats(self) -> Dict[str, Union[int, float, str]]:
        """Get service statistics"""
        return {
            "models_loaded": len(self.detector.models),
            "frames_per_video": self.detector.frames_per_video,
            "input_size": self.detector.input_size,
            "device": self.device,
            "cache_size": len(self._video_cache),
            "memory_usage": torch.cuda.memory_allocated() if self.device == 'cuda' else 0,
            "status": "operational"
        }

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir, exist_ok=True)
            self._video_cache.clear()
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def _ensure_models(self):
        """Ensure model weights are available, download if missing"""
        try:
            os.makedirs(self.weights_dir, exist_ok=True)
            for model_name, url in self.weights.items():
                model_path = os.path.join(self.weights_dir, model_name)
                if not os.path.exists(model_path):
                    self.logger.info(f"Downloading model: {model_name}")
                    try:
                        gdown.download(url, model_path, quiet=False)
                        if not os.path.exists(model_path):
                            raise Exception(f"Failed to download {model_name}")
                        self.logger.info(f"Successfully downloaded {model_name}")
                    except Exception as e:
                        self.logger.error(f"Error downloading {model_name}: {str(e)}")
                        raise
        except Exception as e:
            self.logger.error(f"Error ensuring models: {str(e)}")
            raise RuntimeError(f"Failed to ensure model weights: {str(e)}")

    def verify_model_files(self) -> Dict[str, bool]:
        """Verify if all required model files are present"""
        status = {}
        for model_name in self.weights.keys():
            model_path = os.path.join(self.weights_dir, model_name)
            status[model_name] = os.path.exists(model_path)
        return status