from fastapi import APIRouter, HTTPException
from app.services.kyc.face_service import FaceComparisonService
from app.services.file_preprocessor import FileConverterService
from app.schemas.kyc import ComparisonResponse, FaceComparisonRequest, ErrorCode, TaskResponse, SingleImageComparisonRequest
from app.schemas.health import HealthCheckResponse
import logging
import validators
from urllib.parse import urlparse
from datetime import datetime
from fastapi.responses import JSONResponse
from app.core.config import settings
import psutil
import torch
from app.celery_app.tasks.face_tasks import face_comparison, face_comparison_single_image
from typing import Union, Dict, Any
from celery.result import AsyncResult
import traceback
from fastapi import BackgroundTasks

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
face_service = FaceComparisonService()
file_converter = FileConverterService()

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
    "/compare-faces",
    response_model=Union[TaskResponse, ComparisonResponse],
    summary="Compare Faces in Two Images",
    description="""
    Compare faces between two images to determine if they belong to the same person.
    
    Features:
    - Face detection and recognition using InsightFace
    - Quality assessment of face images
    - Feature extraction and matching
    - Confidence score calculation
    - Pose and lighting analysis
    
    Technical Details:
    - Uses InsightFace's buffalo_l model for face detection and recognition
    - GPU acceleration when available (CUDA)
    - Asynchronous processing via Celery
    - Automatic image format conversion
    
    Requirements:
    - Both images must be accessible via HTTPS URLs
    - Clear, front-facing faces
    - Minimum resolution: 640x640 pixels
    - Maximum file size: 10MB
    - Supported formats: JPG, PNG, WEBP
    
    Response Codes:
    - 200: Successful submission or comparison
    - 400: Invalid image URLs or format
    - 404: Images not accessible
    - 500: Internal processing error
    
    Example Usage:
    ```python
    import requests
    
    response = requests.post(
        "http://api.example.com/v1/kyc/compare-faces",
        json={
            "image1": "https://example.com/id-photo.jpg",
            "image2": "https://example.com/selfie.jpg"
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
                    "examples": {
                        "task": {
                            "value": {
                                "task_id": "7a1b3b11-3568-424f-8ca9-ce1da94d3344",
                                "status": "processing",
                                "message": "Face comparison task submitted successfully"
                            }
                        },
                        "result": {
                            "value": {
                                "match": True,
                                "confidence": 95.5,
                                "face_found": True,
                                "similarity_score": 98.2,
                                "match_category": "HIGH_MATCH",
                                "processing_time": 1.2
                            }
                        }
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
async def compare_faces(request: FaceComparisonRequest):
    try:
        logger.info("Received face comparison request")

        # Validate image URLs
        if not validate_url(request.image1):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid URL for first image",
                    "error_code": ErrorCode.INVALID_URL
                }
            )
        
        if not validate_url(request.image2):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid URL for second image",
                    "error_code": ErrorCode.INVALID_URL
                }
            )

        logger.info(f"Processing first file: {request.image1}")
        logger.info(f"Processing second file: {request.image2}")
        
        # Submit task to Celery
        task = face_comparison.delay(str(request.image1), str(request.image2))
        
        return TaskResponse(
            task_id=task.id,
            status="processing",
            message="Face comparison task submitted successfully"
        )
        
    except Exception as e:
        logger.error(f"Error during face comparison: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.get(
    "/task/{task_id}",
    response_model=Union[TaskResponse, ComparisonResponse],
    summary="Check Face Comparison Task Status"
)
async def get_task_status(task_id: str):
    """Get the status of a face comparison task"""
    try:
        logger.info(f"Checking status for task: {task_id}")
        
        # Get task result using Celery's AsyncResult
        try:
            task = AsyncResult(task_id)
            logger.info(f"Task state: {task.state}, Task info: {task.info}")
        except Exception as task_error:
            logger.error(f"Error retrieving task: {str(task_error)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Error retrieving task",
                    "error_code": "TASK_RETRIEVAL_ERROR",
                    "error": str(task_error),
                    "stack_trace": traceback.format_exc()
                }
            )
        
        # Check task existence and state
        if task is None:
            logger.error(f"Task not found: {task_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Task not found",
                    "error_code": "TASK_NOT_FOUND",
                    "task_id": task_id
                }
            )
        
        if task.state == 'PENDING':
            logger.info(f"Task {task_id} is pending")
            return TaskResponse(
                task_id=task_id,
                status="PENDING",
                message="Task is pending or not found"
            )
        
        if task.state == 'FAILURE':
            error_info = str(task.result) if task.result else "Unknown error"
            logger.error(f"Task {task_id} failed: {error_info}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Task failed",
                    "error_code": "TASK_EXECUTION_FAILED",
                    "error": error_info,
                    "task_state": task.state
                }
            )
        
        if task.state == 'SUCCESS':
            try:
                result = task.get()
                logger.info(f"Task {task_id} completed with result: {result}")
                
                if not isinstance(result, dict):
                    raise ValueError(f"Invalid result type: {type(result)}. Expected dict, got {type(result)}")
                
                if result.get("status") == "success":
                    try:
                        return ComparisonResponse(**result["result"])
                    except Exception as parse_error:
                        logger.error(f"Error parsing comparison response: {str(parse_error)}", exc_info=True)
                        raise HTTPException(
                            status_code=500,
                            detail={
                                "message": "Error parsing comparison response",
                                "error_code": "RESPONSE_PARSE_ERROR",
                                "error": str(parse_error),
                                "result": result
                            }
                        )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "message": "Task failed",
                            "error_code": "TASK_FAILED",
                            "error": result.get("error", "Unknown error"),
                            "task_state": task.state,
                            "result": result
                        }
                    )
            except Exception as result_error:
                logger.error(f"Error processing task result: {str(result_error)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Error processing task result",
                        "error_code": "RESULT_PROCESSING_ERROR",
                        "error": str(result_error),
                        "stack_trace": traceback.format_exc()
                    }
                )
        
        # Task is still processing
        logger.info(f"Task {task_id} is in state: {task.state}")
        return TaskResponse(
            task_id=task_id,
            status=task.state,
            message=f"Task is {task.state.lower()}"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error checking task status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Unexpected error checking task status",
                "error_code": "UNEXPECTED_ERROR",
                "error": str(e),
                "stack_trace": traceback.format_exc()
            }
        )

@router.post(
    "/compare-faces-sync",
    response_model=ComparisonResponse,
    summary="Compare Faces Synchronously",
    description="Synchronous version of face comparison endpoint for backward compatibility"
)
async def compare_faces_sync(request: FaceComparisonRequest):
    """Synchronous face comparison for backward compatibility"""
    try:
        logger.info("Received synchronous face comparison request")

        # Validate image URLs
        if not validate_url(request.image1):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid URL for first image",
                    "error_code": ErrorCode.INVALID_URL
                }
            )
        
        if not validate_url(request.image2):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid URL for second image",
                    "error_code": ErrorCode.INVALID_URL
                }
            )

        logger.info(f"Processing first file: {request.image1}")
        logger.info(f"Processing second file: {request.image2}")
        
        result = await face_service.compare_faces(
            str(request.image1),
            str(request.image2)
        )
        
        logger.info(
            f"Comparison completed. Match: {result.match}, "
            f"Confidence: {result.confidence:.2f}, "
            f"Face Found: {result.face_found}"
        )
        
        return result
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error during face comparison: {str(e)}")
        return ComparisonResponse(
            match=False,
            confidence=0.0,
            face_found=False,
            error_message=str(e),
            error_code=ErrorCode.INTERNAL_ERROR,
            processing_time=0.0
        )

@router.get(
    "/health",
    tags=["Health"],
    response_model=HealthCheckResponse,
    summary="Face Comparison Service Health Check"
)
async def health_check():
    """Check health of face comparison services"""
    try:
        # Check if face service is initialized by checking if model is loaded
        health_status = await face_service.check_health()
        memory = psutil.virtual_memory()
        
        service_status = {
            "status": "healthy" if health_status.get("model_working", False) else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "face_comparison": {
                    "status": "healthy" if health_status.get("model_working", False) else "unhealthy",
                    "version": settings.VERSION,
                    "models": {
                        "insightface": "active" if health_status.get("model_working", False) else "inactive"
                    },
                    "model_loaded": health_status.get("model_loaded", False),
                    "model_working": health_status.get("model_working", False),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "gpu_available": torch.cuda.is_available()
                }
            },
            "system": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "environment": settings.ENV,
                "version": settings.VERSION,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "memory": {
                    "total": float(memory.total/1e9),
                    "available": float(memory.available/1e9),
                    "percent_used": memory.percent
                }
            }
        }
        
        return service_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        error_response = {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "error_code": "SERVICE_ERROR",
            "services": {
                "face_comparison": {
                    "status": "unhealthy",
                    "error": str(e),
                    "error_code": "SERVICE_ERROR",
                    "model_loaded": False,
                    "model_working": False
                }
            },
            "system": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "environment": settings.ENV,
                "version": settings.VERSION,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count()
            }
        }
        return JSONResponse(status_code=503, content=error_response)

@router.post(
    "/compare-faces-in-image",
    response_model=Union[TaskResponse, ComparisonResponse],
    summary="Compare Two Faces in Single Image",
    description="""
    Compare two faces detected in a single image (e.g., selfie with ID card).
    
    Technical Implementation:
    - Automatic detection and ranking of faces by size
    - Face detection and recognition using InsightFace
    - Quality assessment for each detected face
    - Feature extraction and matching
    - GPU-accelerated processing
    - Asynchronous task handling via Celery
    
    Detection Capabilities:
    - Multiple face detection
    - Face size ranking
    - Quality assessment
    - Pose estimation
    - Lighting analysis
    - Blur detection
    
    Requirements:
    - Image must contain exactly two faces
    - Image must be accessible via HTTPS URL
    - Clear, distinguishable faces
    - Minimum resolution: 640x640 pixels
    - Maximum file size: 10MB
    - Supported formats: JPG, PNG, WEBP
    
    Response Format:
    - Async Mode: Returns task_id for status polling
    - Match result (true/false)
    - Confidence score (0-100%)
    - Face quality metrics
    - Processing time
    
    Common Use Cases:
    - ID verification with selfie
    - Document holder verification
    - Identity proofing
    """,
    responses={
        200: {
            "description": "Task submitted successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "task": {
                            "value": {
                                "task_id": "7a1b3b11-3568-424f-8ca9-ce1da94d3344",
                                "status": "processing",
                                "message": "Face comparison task submitted successfully"
                            }
                        },
                        "result": {
                            "value": {
                                "match": True,
                                "confidence": 95.5,
                                "face_found": True,
                                "similarity_score": 98.2,
                                "match_category": "HIGH_MATCH",
                                "image1_quality": {
                                    "overall_quality": 0.85,
                                    "width": 800,
                                    "height": 600
                                },
                                "image2_quality": {
                                    "overall_quality": 0.88,
                                    "width": 800,
                                    "height": 600
                                },
                                "processing_time": 1.2
                            }
                        }
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
                            "message": "Invalid image URL or no faces detected",
                            "error_code": "INVALID_INPUT"
                        }
                    }
                }
            }
        },
        500: {
            "description": "Processing error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "message": "Error processing image",
                            "error_code": "PROCESSING_ERROR"
                        }
                    }
                }
            }
        }
    }
)
async def compare_faces_in_single_image(request: SingleImageComparisonRequest):
    try:
        logger.info("Received single-image face comparison request")

        # Validate image URL
        if not validate_url(str(request.image_url)):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid image URL",
                    "error_code": ErrorCode.INVALID_URL
                }
            )

        logger.info(f"Processing image: {request.image_url}")
        
        # Submit task to Celery
        task = face_comparison_single_image.delay(str(request.image_url))
        
        return TaskResponse(
            task_id=task.id,
            status="processing",
            message="Face comparison task submitted successfully"
        )
        
    except Exception as e:
        logger.error(f"Error during face comparison: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.post(
    "/compare-faces-in-image-sync",
    response_model=ComparisonResponse,
    summary="Compare Two Faces in Single Image Synchronously",
    description="Synchronous version of single-image face comparison endpoint for backward compatibility"
)
async def compare_faces_in_single_image_sync(request: SingleImageComparisonRequest):
    """Synchronous single-image face comparison for backward compatibility"""
    try:
        logger.info("Received synchronous single-image face comparison request")

        # Validate image URL
        if not validate_url(str(request.image_url)):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid image URL",
                    "error_code": ErrorCode.INVALID_URL
                }
            )

        logger.info(f"Processing image: {request.image_url}")
        
        result = await face_service.compare_faces_in_image(str(request.image_url))
        
        logger.info(
            f"Comparison completed. Match: {result.match}, "
            f"Confidence: {result.confidence:.2f}, "
            f"Face Found: {result.face_found}"
        )
        
        return result
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error during face comparison: {str(e)}")
        return ComparisonResponse(
            match=False,
            confidence=0.0,
            face_found=False,
            error_message=str(e),
            error_code=ErrorCode.INTERNAL_ERROR,
            processing_time=0.0
        )