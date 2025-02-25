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
def document_verification(self, front_image_url: str, back_image_url: str) -> Dict[str, Any]:
    """
    Run document verification task
    
    Args:
        front_image_url: URL of document front image
        back_image_url: URL of document back image
        
    Returns:
        Dict containing verification results
    """
    try:
        start_time = time.time()
        # Run the async verification in a sync context
        result = run_async(document_service.verify_ghana_card(
            card_front=front_image_url, 
            card_back=back_image_url
        ))
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "result": result.dict(),
            "processing_time": processing_time,
            "task_id": self.request.id
        }
    except Exception as e:
        logger.error(f"Document verification failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time": time.time() - start_time,
            "task_id": self.request.id
        }