from app.celery_app.celery import celery_app
import logging
import time
from typing import Dict, Any, Union, List
from app.services.spoof.spoof_detection_service import get_spoof_detector_service
from app.celery_app.tasks.base import GPUTask
import asyncio
import traceback

logger = logging.getLogger(__name__)
spoof_service = get_spoof_detector_service()

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@celery_app.task(
    bind=True,
    base=GPUTask,
    name='app.celery_app.tasks.spoof_tasks.spoof_detection',
    queue='high_priority',
    rate_limit='10/m',
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=300,
    time_limit=360
)
def spoof_detection(self, image_urls: Union[str, List[str]]) -> Dict[str, Any]:
    """Run spoof detection task"""
    try:
        start_time = time.time()
        
        # Handle both single URL and list of URLs for backward compatibility
        if isinstance(image_urls, str):
            image_urls = [image_urls]
            
        # Run the async analysis in a sync context
        result = run_async(spoof_service.analyze_multiple_images(image_urls))
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "result": result.dict(),
            "processing_time": processing_time,
            "task_id": self.request.id
        }
    except Exception as e:
        logger.error(f"Spoof detection failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time": time.time() - start_time,
            "task_id": self.request.id
        }