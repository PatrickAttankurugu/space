from app.celery_app.celery import celery_app
import logging
import time
from typing import Dict, Any, Optional
from app.services.kyc.face_service import face_service
from app.celery_app.tasks.base import GPUTask
from celery.exceptions import Retry
import traceback
import asyncio

logger = logging.getLogger(__name__)

def run_async(coro):
    """Helper function to run coroutines in a sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@celery_app.task(
    bind=True,
    base=GPUTask,
    name='app.celery_app.tasks.face_tasks.face_comparison',
    queue='high_priority',
    rate_limit='10/m',
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=300,
    time_limit=360
)
def face_comparison(self, image1_url: str, image2_url: str) -> Dict[str, Any]:
    """
    Run face comparison task with enhanced error handling and retries
    
    Args:
        image1_url: URL of first image
        image2_url: URL of second image
        
    Returns:
        Dict containing comparison results
    """
    start_time = time.time()
    logger.info(f"Starting face comparison task {self.request.id}")
    
    try:
        # Run the async comparison in a sync context
        result = run_async(face_service.compare_faces(image1_url, image2_url))
        processing_time = time.time() - start_time
        
        logger.info(f"Face comparison task {self.request.id} completed successfully")
        return {
            "status": "success",
            "result": result.dict(),
            "processing_time": processing_time,
            "task_id": self.request.id
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time": processing_time,
            "task_id": self.request.id
        }
        logger.error(f"Face comparison failed: {error_info}")
        
        # Determine if we should retry
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < self.max_retries:
            raise self.retry(
                exc=e,
                countdown=2 ** self.request.retries * 60,  # Exponential backoff
                kwargs={"image1_url": image1_url, "image2_url": image2_url}
            )
            
        return {
            "status": "error",
            **error_info
        }

@celery_app.task(
    bind=True,
    name='app.celery_app.tasks.face_tasks.check_task_status',
    queue='high_priority'
)
def check_task_status(self, task_id: str) -> Dict[str, Any]:
    """
    Check the status of a task with enhanced error handling
    
    Args:
        task_id: The ID of the task to check
        
    Returns:
        Dict containing task status and results
    """
    try:
        task = celery_app.AsyncResult(task_id)
        logger.info(f"Checking task {task_id} status: {task.status}")
        
        # Build response based on task state
        response = {
            "task_id": task_id,
            "status": task.status,
            "date_done": task.date_done.isoformat() if task.date_done else None,
        }
        
        if task.ready():
            if task.successful():
                response["result"] = task.result
            else:
                response["error"] = str(task.result) if task.result else "Task failed"
                response["traceback"] = task.traceback
        
        logger.info(f"Task status result: {response}")
        return response
        
    except Exception as e:
        error_msg = f"Error checking task status: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "task_id": task_id,
            "status": "ERROR",
            "error": error_msg
        }

@celery_app.task(
    bind=True,
    base=GPUTask,
    name='app.celery_app.tasks.face_tasks.face_comparison_single_image',
    queue='high_priority',
    rate_limit='10/m',
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=300,
    time_limit=360
)
def face_comparison_single_image(self, image_url: str) -> Dict[str, Any]:
    """
    Run face comparison task for two faces in a single image
    
    Args:
        image_url: URL of image containing two faces
        
    Returns:
        Dict containing comparison results
    """
    start_time = time.time()
    logger.info(f"Starting single-image face comparison task {self.request.id}")
    
    try:
        result = run_async(face_service.compare_faces_in_image(image_url))
        processing_time = time.time() - start_time
        
        logger.info(f"Single-image face comparison task {self.request.id} completed successfully")
        return {
            "status": "success",
            "result": result.dict(),
            "processing_time": processing_time,
            "task_id": self.request.id
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time": processing_time,
            "task_id": self.request.id
        }
        logger.error(f"Single-image face comparison failed: {error_info}")
        
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < self.max_retries:
            raise self.retry(
                exc=e,
                countdown=2 ** self.request.retries * 60,
                kwargs={"image_url": image_url}
            )
            
        return {
            "status": "error",
            **error_info
        }