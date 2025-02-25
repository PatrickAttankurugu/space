from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.spoof.spoof_detection_service import get_spoof_detector_service
from app.schemas.spoof import SpoofDetectionRequest, SpoofAnalysisResponse
from app.schemas.health import HealthCheckResponse
from app.core.config import settings
import logging
from datetime import datetime
import torch
from app.celery_app.tasks import spoof_detection
from typing import Union
from app.schemas.tasks import TaskResponse, TaskResult

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
spoof_service = get_spoof_detector_service()

@router.post(
    "/analyze",
    response_model=TaskResponse,
    summary="Analyze Image for Spoof Detection",
    description="""
    Analyze a selfie/live photo to detect potential spoofing attempts during identity verification.
    
    Technical Implementation:
    - Face detection using MTCNN
    - Deep learning-based spoof detection using MobileNetV2
    - Multi-stage verification process
    - GPU-accelerated inference
    - Asynchronous processing via Celery
    
    Detection Capabilities:
    - Print attacks (2D photos)
    - Replay attacks (digital screens)
    - Mask detection
    - Deep fake detection
    - Quality assessment
    
    Requirements:
    - Single person facing the camera
    - Clear face visibility (no occlusions)
    - Good lighting conditions
    - Minimum resolution: 480p
    - Maximum file size: 10MB
    - RGB image format
    
    Response Format:
    ```json
    {
        "task_id": "uuid-string",
        "status": "success",
        "result": {
            "spoof_detected": false,
            "confidence": 0.98,
            "quality_score": 0.85,
            "liveness_score": 0.95,
            "processing_time": 1.2
        }
    }
    ```
    
    Integration Example:
    ```python
    import requests
    
    response = requests.post(
        "http://api.example.com/v1/spoof/analyze",
        json={
            "image_url": "https://example.com/selfie.jpg"
        }
    )
    
    task_id = response.json()["task_id"]
    ```
    """,
    responses={
        200: {
            "description": "Task submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "task_id": "1234-5678-90ab-cdef",
                        "status": "processing",
                        "message": "Spoof detection task submitted successfully"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid URL format or inaccessible image"
                    }
                }
            }
        }
    }
)
async def analyze_spoof(request: SpoofDetectionRequest):
    """Submit images for spoof detection analysis"""
    try:
        logger.info("Received spoof detection request")
        
        # Validate all URLs
        for url in request.image_urls:
            if not await spoof_service.validate_url(str(url)):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": f"Invalid image URL: {url}",
                        "error_code": "INVALID_URL"
                    }
                )

        logger.info(f"Processing {len(request.image_urls)} images")
        
        # Submit to Celery
        task = spoof_detection.delay(request.image_urls)
        
        return TaskResponse(
            task_id=task.id,
            status="processing",
            message=f"Spoof detection task submitted successfully for {len(request.image_urls)} images"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting spoof detection task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting task: {str(e)}"
        )

@router.get(
    "/task/{task_id}",
    response_model=Union[TaskResponse, SpoofAnalysisResponse],
    summary="Check Spoof Detection Task Status",
    responses={
        200: {"description": "Task status retrieved successfully"},
        404: {"description": "Task not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_task_status(task_id: str):
    """Get the status of a submitted spoof detection task"""
    try:
        logger.info(f"Checking status for spoof detection task: {task_id}")
        task = spoof_detection.AsyncResult(task_id)
        
        if not task:
            logger.error(f"Spoof detection task not found: {task_id}")
            raise HTTPException(
                status_code=404,
                detail={"message": "Task not found", "error_code": "TASK_NOT_FOUND"}
            )
        
        if task.ready():
            result = task.get()
            logger.info(f"Spoof detection task {task_id} completed")
            if result.get("status") == "success":
                return SpoofAnalysisResponse(**result["result"])
            else:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Task failed",
                        "error": result.get("error"),
                        "traceback": result.get("traceback")
                    }
                )
        
        logger.info(f"Spoof detection task {task_id} still processing")
        return TaskResponse(
            task_id=task_id,
            status=task.status,
            message="Spoof detection task is still processing"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking spoof detection task status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error checking task status",
                "error_code": "TASK_CHECK_ERROR",
                "error": str(e)
            }
        )

@router.post(
    "/analyze-sync",
    response_model=SpoofAnalysisResponse,
    summary="Analyze Images for Spoof Detection Synchronously",
    description="Analyze 1-3 images for more accurate spoof detection"
)
async def analyze_spoof_sync(request: SpoofDetectionRequest):
    """Analyze multiple images for spoof detection synchronously"""
    try:
        logger.info("Received synchronous spoof detection request")
        
        # Validate all URLs
        for url in request.image_urls:
            if not await spoof_service.validate_url(str(url)):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": f"Invalid image URL: {url}",
                        "error_code": "INVALID_URL"
                    }
                )

        logger.info(f"Processing {len(request.image_urls)} images")
        
        result = await spoof_service.analyze_multiple_images(
            [str(url) for url in request.image_urls]
        )
        
        logger.info(
            f"Analysis completed. "
            f"Is Spoof: {result.is_spoof}, "
            f"Confidence: {result.confidence:.2f}, "
            f"Liveness: {result.liveness_score:.2f}, "
            f"Images Processed: {result.num_images_processed}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Spoof analysis failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to analyze images: {str(e)}"
        )

@router.get(
    "/health",
    tags=["Health"],
    response_model=HealthCheckResponse,
    summary="Spoof Detection Service Health Check",
    description="""
    Check the health status of the spoof detection service.
    
    Verifies:
    - Model initialization status
    - GPU/CPU availability
    - Service availability
    - System resources
    
    Returns:
        HealthCheckResponse containing:
        - Service status
        - Model status
        - System information
    """
)
async def health_check():
    """Check the health status of the spoof detection service"""
    try:
        # Test service initialization
        if not hasattr(spoof_service, 'initialized'):
            raise HTTPException(
                status_code=503, 
                detail="Spoof detection service not properly initialized"
            )
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "spoof_detection": {
                    "status": "healthy",
                    "service_available": True,
                    "model_loaded": True,
                    "device": spoof_service.device,
                    "gpu_available": torch.cuda.is_available()
                }
            },
            "system": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "environment": settings.ENV,
                "version": settings.VERSION
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        error_response = {
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
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "environment": settings.ENV,
                "version": settings.VERSION
            }
        }
        return JSONResponse(status_code=503, content=error_response)