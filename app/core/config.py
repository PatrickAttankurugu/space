from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Set, Dict, List, Any, Optional, Tuple, Literal
import os
import torch

class Settings(BaseSettings):
    # Environment Settings
    ENV: str = "production"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    
    # Worker Settings for g4dn.2xlarge (8 vCPUs)
    @property
    def WORKERS(self) -> int:
        try:
            if torch.cuda.is_available():
                return 1  # Single worker for GPU workloads
            else:
                import multiprocessing
                num_cores = multiprocessing.cpu_count()
                return max(min(num_cores, 4), 2)  # 2-4 workers for CPU mode
        except:
            return 1  # Default to single worker
    
    # Celery Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    @property
    def REDIS_URL(self) -> str:
        """Get Redis URL with password if set"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # Celery Worker Settings
    CELERY_WORKER_CONCURRENCY: int = 1  # For GPU tasks
    CELERY_TASK_TIME_LIMIT: int = 3600  # 1 hour
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3300  # 55 minutes
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = 50
    CELERY_WORKER_MAX_MEMORY: int = 8000000  # 8GB
    
    # GPU Settings
    CUDA_DEVICE: str = "cuda:0"
    GPU_MEMORY_FRACTION: float = 0.8  # Reserve 80% of GPU memory
    BATCH_SIZE: int = 32  # Optimal for T4 GPU
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Machine Learning API"
    PROJECT_DESCRIPTION: str = "API for Machine Learning services"
    VERSION: str = "1.0.0"
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # ML Model Paths
    ML_MODELS_DIR: Path = BASE_DIR / "ml_models"
    GHANA_CARD_MODEL_PATH: Path = ML_MODELS_DIR / "document_verification" / "ghana_card_modelV2.pth"
    SPOOF_MODEL_PATH: str = "ml_models/spoof/spoof_detector.pth"
    
    # InsightFace Settings
    INSIGHTFACE_MODEL_PATH: Path = ML_MODELS_DIR / "insightface"
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    _INSIGHTFACE_DET_SIZE: str = "640,640"  # Accept string from env var
    
    @property
    def INSIGHTFACE_DET_SIZE(self) -> Tuple[int, int]:
        try:
            width, height = map(int, self._INSIGHTFACE_DET_SIZE.split(','))
            return (width, height)
        except:
            return (640, 640)  # Default value
            
    INSIGHTFACE_CTX_ID: int = 0  # Use GPU
    INSIGHTFACE_DOWNLOAD_TIMEOUT: int = 600
    INSIGHTFACE_CACHE_DIR: Path = BASE_DIR / "model_cache" / "insightface"
    
    # Service configurations
    FACE_MATCH_THRESHOLD: float = 0.58
    DOCUMENT_VERIFY_THRESHOLD: float = 0.55
    IMPROVED_FACE_MATCH_THRESHOLD: float = 0.5
    FACE_MATCH_MIN_QUALITY: float = 0.6
    FACE_MATCH_MIN_CONFIDENCE: float = 0.7
    FACE_DETECTION_SCORE_THRESHOLD: float = 0.6
    FACE_RECOGNITION_SCORE_THRESHOLD: float = 0.5

    # File validation settings
    CARD_MIN_FEATURES: int = 2
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: Set[str] = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 
        'image/tiff', 'image/webp',
        'application/pdf'
    }
    MIN_IMAGE_DIMENSION: int = 224  # Minimum dimension for input images
    MAX_IMAGE_DIMENSION: int = 4096  # Maximum dimension for input images
    TARGET_FACE_DIMENSION: int = 640
    QUALITY_THRESHOLD: float = 0.3

    # PDF processing settings
    PDF_DPI: int = 300
    MAX_PDF_PAGES: int = 10
    
    # Image quality thresholds
    MIN_IMAGE_QUALITY: float = 0.5
    MIN_BRIGHTNESS: float = 0.3
    MIN_CONTRAST: float = 0.3
    MIN_SHARPNESS: float = 0.3

    # Ghana Card specifications
    CARD_ASPECT_RATIO: float = 1.58
    ASPECT_RATIO_TOLERANCE: float = 0.1
    CARD_CONFIDENCE_THRESHOLD: float = 0.5

    # MRZ Settings
    MRZ_COUNTRY_CODE: str = "GHA"
    MRZ_REQUIRED_FIELDS: Set[str] = {
        "type",
        "country",
        "number",
        "date_of_birth",
        "sex",
        "expiration_date"
    }

    # Model settings
    MODEL_INPUT_SIZE: Tuple[int, int] = (224, 224)
    MODEL_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    MODEL_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    NUM_WORKERS: int = 4

    # File conversion settings
    CONVERSION_TIMEOUT: int = 30
    MAX_CONVERSION_RETRIES: int = 3
    CONVERSION_TEMP_DIR: Path = BASE_DIR / "temp"
    CLEANUP_TEMP_FILES: bool = True

    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = LOG_DIR / "app.log"
    MAX_LOG_SIZE: int = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT: int = 5

    # API Security Settings
    API_KEYS: Dict[str, str] = {
        "test_key_123": "test-environment",
        "live_key_456": "production-environment"
    }
    RATE_LIMIT_PER_MINUTE: int = 60

    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://api.agregartech.com", "https://app.agregartech.com", "https://staging-videokyc.agregartech.com"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "api.agregartech.com", "app.agregartech.com", "165.22.70.167", "staging-videokyc.agregartech.com"]
    
    # Device Settings
    FORCE_CPU: bool = False  # Can be set via environment variable
    
    class Config:
        case_sensitive: bool = True
        env_file: str = ".env"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name in ["API_KEYS", "SECURITY_HEADERS", "CORS_ORIGINS", "ALLOWED_HOSTS", "ALLOWED_FILE_TYPES"]:
                try:
                    import json
                    return json.loads(raw_val)
                except:
                    return {}
            return raw_val

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.CONVERSION_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.INSIGHTFACE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self.INSIGHTFACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Validate model file existence
        if not self.GHANA_CARD_MODEL_PATH.exists():
            print(f"Warning: Model file not found at {self.GHANA_CARD_MODEL_PATH}")

    def verify_api_key(self, api_key: str) -> bool:
        """Verify if an API key is valid"""
        return api_key in self.API_KEYS

    def get_api_key_environment(self, api_key: str) -> str:
        """Get the environment for an API key"""
        return self.API_KEYS.get(api_key, "unknown")

    def cleanup(self):
        """Cleanup temporary files if enabled"""
        if self.CLEANUP_TEMP_FILES and self.CONVERSION_TEMP_DIR.exists():
            import shutil
            try:
                shutil.rmtree(self.CONVERSION_TEMP_DIR)
                self.CONVERSION_TEMP_DIR.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp directory: {str(e)}")

    # Health check settings
    HEALTH_CHECK_TIMEOUT: float = 30.0
    MAX_HEALTH_CHECK_RETRIES: int = 3
    HEALTH_CHECK_INTERVAL: float = 30.0

    # Service thresholds
    MIN_HEALTHY_SERVICES: float = 0.8

    # OCR Settings
    ENABLE_CARD_OCR: bool = True
    OCR_MIN_CONFIDENCE: float = 0.7
    OCR_CLEANUP_THRESHOLD: int = 10
    OCR_MODEL_PATH: str = "/tmp/ocr_models"
    OCR_CONFIDENCE_THRESHOLD: float = 0.5

    @property
    def DEVICE(self) -> str:
        if self.FORCE_CPU:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

# Initialize settings
settings = Settings()
# Setup directories
settings.setup_directories()