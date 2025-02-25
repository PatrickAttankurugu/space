from app.celery_app.celery import celery_app
import logging
import time
from typing import Dict, Any
from app.services.document_verification.document_service import document_service
from app.celery_app.tasks.base import GPUTask, run_async
import asyncio
import traceback

logger = logging.getLogger(__name__)

@celery_app.task(
    bind=True,
    base=GPUTask,
    name='app.celery_app.tasks.document_tasks.document_verification',
    queue='high_priority',
    rate_limit='10/m',
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=300,
    time_limit=360
)
def document_verification(self, card_front_with_selfie: str, card_front: str) -> Dict[str, Any]:
    """
    Run document verification task
    
    Args:
        card_front_with_selfie: URL of image showing Ghana card front with selfie for validation
        card_front: URL of clear Ghana card front image for information extraction
        
    Returns:
        Dict containing verification results including:
        - Validation status
        - Detected security features
        - Extracted card information
        - Processing time and metrics
    """
    try:
        start_time = time.time()
        
        # Run the async verification in a sync context
        result = run_async(document_service.verify_ghana_card(
            card_front_with_selfie=card_front_with_selfie, 
            card_front=card_front
        ))
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "result": result.dict(),
            "processing_time": processing_time,
            "task_id": self.request.id
        }
    except Exception as e:
        logger.error(f"Document verification failed: {str(e)}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time": time.time() - start_time,
            "task_id": self.request.id
        }