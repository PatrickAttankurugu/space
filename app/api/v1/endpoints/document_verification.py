from fastapi import APIRouter, HTTPException
from app.services.document_verification.document_service import DocumentService, GhanaCardError
from app.services.file_preprocessor import FileConverterService
from app.schemas.document import VerificationResponse, GhanaCardRequest, ErrorCode
from app.services.kyc.face_service import FaceComparisonService
import logging
from urllib.parse import urlparse
import validators
from fastapi.responses import JSONResponse
import torch
from datetime import datetime
from app.core.config import settings
import time
from app.schemas.health import HealthCheckResponse
from pydantic import BaseModel, Field
import traceback
from app.celery_app.tasks.document_tasks import document_verification
from app.schemas.tasks import TaskResponse
from typing import Union

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
document_service = DocumentService()
file_converter = FileConverterService()
face_service = FaceComparisonService()

def validate_url(url: str) -> bool:
    """Validate if the provided URL is valid and accessible"""
    try:
        if not url:
            return False
        
        # Parse URL to ensure it's well-formed
        parsed = urlparse(str(url))
        if not all([parsed.scheme, parsed.netloc]):
            return False
        
        # Check if it's a valid URL format
        if not validators.url(str(url)):
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"URL validation error: {str(e)}")
        return False

@router.post(
    "/verify-ghana-card",
    response_model=TaskResponse,
    summary="Verify Ghana National ID Card",
    description="""
    Verify the authenticity of a Ghana National ID Card using machine learning.
    
    The service analyzes both front and back images of the card to:
    - Detect required security features (ECOWAS Logo, Ghana Coat of Arms, Ghana Flag, etc.)
    - Validate card layout and design
    - Extract and verify MRZ (Machine Readable Zone) data
    - Calculate confidence score based on detected features
    
    Requirements:
    - Both images must be accessible via HTTPS URLs
    - Images should be clear and well-lit
    - Minimum resolution: 300 DPI recommended
    
    The verification process returns:
    - Overall validity status
    - Confidence score (0-100%)
    - List of detected security features
    - Extracted MRZ data (when available)
    - Processing time and performance metrics
    """,
    responses={
        200: {
            "description": "Task submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "task_id": "1234-5678-90ab-cdef",
                        "status": "processing",
                        "message": "Document verification task submitted successfully"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "message": "Invalid image URL",
                            "error_code": "INVALID_URL"
                        }
                    }
                }
            }
        }
    }
)
async def verify_ghana_card(request: GhanaCardRequest):
    """Submit Ghana card images for verification"""
    try:
        logger.info("Received document verification request")
        
        # Validate URLs
        if not validate_url(request.front_image):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid URL for front image",
                    "error_code": ErrorCode.INVALID_URL
                }
            )
        
        if not validate_url(request.back_image):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid URL for back image",
                    "error_code": ErrorCode.INVALID_URL
                }
            )

        logger.info(f"Processing front image: {request.front_image}")
        logger.info(f"Processing back image: {request.back_image}")
        
        # Submit to Celery
        task = document_verification.delay(
            front_image_url=str(request.front_image),
            back_image_url=str(request.back_image)
        )
        
        return TaskResponse(
            task_id=task.id,
            status="processing",
            message="Document verification task submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting verification task: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error submitting task",
                "error_code": "TASK_SUBMISSION_ERROR",
                "error": str(e)
            }
        )

@router.get(
    "/task/{task_id}",
    response_model=Union[TaskResponse, VerificationResponse],
    summary="Check Document Verification Task Status",
    responses={
        200: {"description": "Task status retrieved successfully"},
        404: {"description": "Task not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_task_status(task_id: str):
    """Get the status of a submitted document verification task"""
    try:
        logger.info(f"Checking status for document verification task: {task_id}")
        task = document_verification.AsyncResult(task_id)
        
        if not task:
            logger.error(f"Document verification task not found: {task_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Task not found",
                    "error_code": "TASK_NOT_FOUND"
                }
            )
        
        if task.ready():
            result = task.get()
            logger.info(f"Document verification task {task_id} completed")
            if result.get("status") == "success":
                return VerificationResponse(**result["result"])
            else:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Task failed",
                        "error": result.get("error"),
                        "traceback": result.get("traceback")
                    }
                )
        
        logger.info(f"Document verification task {task_id} still processing")
        return TaskResponse(
            task_id=task_id,
            status=task.status,
            message="Document verification task is still processing"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking document verification task status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error checking task status",
                "error_code": "TASK_CHECK_ERROR",
                "error": str(e)
            }
        )

@router.post(
    "/verify-ghana-card-sync",
    response_model=VerificationResponse,
    summary="Verify Ghana National ID Card Synchronously",
    description="Synchronous version of Ghana Card verification for backward compatibility"
)
async def verify_ghana_card_sync(request: GhanaCardRequest):
    try:
        # Log the incoming request
        logger.info(f"Starting synchronous verification for images: {str(request.front_image)[:50]}... and {str(request.back_image)[:50]}...")
        
        result = await document_service.verify_ghana_card(
            card_front=str(request.front_image),
            card_back=str(request.back_image)
        )
        return result
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}\nTraceback: {traceback.format_exc()}")
        
        error_message = str(e)
        if "Connection aborted" in error_message or "RemoteDisconnected" in error_message:
            error_message = "Failed to download image. Please check the image URLs and try again."
        
        return JSONResponse(
            status_code=400,
            content={
                "is_valid": False,
                "error_message": error_message,
                "error_code": ErrorCode.URL_ACCESS_ERROR,
                "processing_time": 0.0
            }
        )

@router.get(
    "/health",
    tags=["Health"],
    response_model=HealthCheckResponse,
    summary="Check Document Verification Health Status"
)
async def health_check():
    try:
        doc_service = await document_service.check_health()
        response = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "document_verification": {
                    **doc_service,
                    "status": "healthy"
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
        return response
    except Exception as e:
        error_response = {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "error_code": "SERVICE_ERROR",
            "services": {
                "document_verification": {
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

class GhanaCardRequest(BaseModel):
    front_image: str = Field(..., description="URL or base64 of the Ghana Card front image")
    back_image: str = Field(..., description="URL or base64 of the Ghana Card back image")
    
    class Config:
        json_schema_extra = {
            "example": {
                "front_image": "https://example.com/card_front.jpg",
                "back_image": "https://example.com/card_back.jpg"
            }
        }