import os
from pathlib import Path
from insightface.app import FaceAnalysis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download required InsightFace models"""
    try:
        models_dir = Path('/ml_models/face/models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing InsightFace model download...")
        
        app = FaceAnalysis(
            name='buffalo_l',
            root=str(models_dir)
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        
        logger.info("InsightFace models downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading InsightFace models: {str(e)}")
        return False

if __name__ == "__main__":
    download_models()