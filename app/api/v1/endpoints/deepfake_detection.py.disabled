from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Dict, Union
from app.services.deepfake.deepfake_detection_service import get_detector_service
from app.schemas.deepfake import DeepfakeDetectionRequest, DeepfakeAnalysisResponse, DetectionStats
from app.schemas.errors import ErrorResponse
from app.schemas.health import HealthCheckResponse
from app.core.config import settings
import logging
from datetime import datetime
import torch

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
detector_service = get_detector_service()

@router.post(
    "/analyze",
    response_model=DeepfakeAnalysisResponse,
    summary="Analyze Video for Deepfake Detection",
    description="""
    Analyze a video to detect potential deepfake manipulation.
    
    The service performs:
    - Frame extraction and analysis
    - Face detection and tracking
    - Deepfake artifact detection
    - Confidence score calculation
    - Temporal consistency check
    
    Features:
    - Supports multiple video formats (MP4, AVI, MOV)
    - Maximum video length: 5 minutes
    - Minimum resolution: 480p
    - Maximum file size: 100MB
    
    Notes:
    - Video must be accessible via HTTPS URL
    - Processing time varies based on video length and complexity
    - Higher resolution videos may take longer to process
    """,
    responses={
        200: {
            "description": "Successful analysis",
            "content": {
                "application/json": {
                    "example": {
                        "is_deepfake": False,
                        "confidence": 98.5,
                        "processing_time": 15.2,
                        "frames_analyzed": 450,
                        "video_duration": 30.5,
                        "detection_method": "ensemble",
                        "frame_level_results": {
                            "real_frames": 445,
                            "fake_frames": 5,
                            "average_confidence": 98.5
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid input or analysis failed",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Failed to analyze video",
                        "detail": "Video format not supported",
                        "error_code": "INVALID_FORMAT"
                    }
                }
            }
        },
        413: {
            "description": "Video file too large",
            "content": {
                "application/json": {
                    "example": {
                        "error": "File size exceeds limit",
                        "detail": "Maximum file size is 100MB",
                        "error_code": "FILE_TOO_LARGE"
                    }
                }
            }
        }
    }
)
async def analyze_video(video_url: str):
    """Analyze video from URL for deepfake detection"""
    try:
        return await detector_service.analyze_video(video_url)
    except Exception as e:
        logger.error(f"Video analysis failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to analyze video: {str(e)}"
        )

@router.get(
    "/stats/",
    response_model=DetectionStats,
    summary="Get Service Statistics",
    description="""
    Retrieve operational statistics for the deepfake detection service.
    
    Returns:
    - Total number of videos processed
    - Average processing time
    - Model performance metrics
    - Resource utilization
    - Service uptime
    - Success/failure rates
    
    Used for monitoring and performance analysis.
    """,
    responses={
        200: {
            "description": "Statistics retrieved successfully",
            "content": {"application/json": {"example": DetectionStats.Config.json_schema_extra["example"]}}
        },
        500: {
            "description": "Error retrieving statistics",
            "content": {"application/json": {"example": ErrorResponse.Config.json_schema_extra["example"]}}
        }
    }
)
async def get_detection_stats():
    """
    Get statistics about the detection service
    
    Returns:
        Dict containing service statistics including:
        - Number of models loaded
        - Processing device (CPU/GPU)
        - Batch size
        - Service status
    """
    try:
        return await detector_service.get_detection_stats()
    except Exception as e:
        logger.error(f"Error getting detection stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving service statistics: {str(e)}"
        )

@router.get(
    "/health",
    tags=["Health"],
    response_model=HealthCheckResponse,
    summary="Deepfake Detection Service Health Check",
    description="""
    Check the health status of the deepfake detection service.
    
    Verifies:
    - Model availability and initialization
    - GPU/CPU availability
    - Memory usage
    - Processing pipeline status
    - External dependencies
    
    Returns:
        HealthCheckResponse containing:
        - Service status
        - Model status
        - System information
        - Resource utilization
    """,
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-03-20T10:30:00Z",
                        "services": {
                            "deepfake_detection": {
                                "status": "healthy",
                                "model_loaded": True,
                                "models": {},
                                "device": "cuda:0",
                                "gpu_available": True
                            }
                        },
                        "system": {
                            "device": "cuda:0",
                            "cuda_available": True,
                            "cuda_device_count": 1,
                            "environment": "production",
                            "version": "1.0.0"
                        }
                    }
                }
            }
        },
        503: {
            "description": "Service is unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2024-03-20T10:30:00Z",
                        "error": "Model failed to load",
                        "error_code": "MODEL_LOADING_ERROR"
                    }
                }
            }
        }
    }
)
async def health_check():
    """Check the health status of the deepfake detection service"""
    try:
        stats = await detector_service.get_detection_stats()
        model_status = detector_service.verify_model_files()
        return {
            "status": "healthy" if all(model_status.values()) else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "deepfake_detection": {
                    "status": "healthy" if all(model_status.values()) else "unhealthy",
                    "models": model_status,
                    "device": detector_service.device,
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
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "error_code": "SERVICE_ERROR",
            "services": {
                "deepfake_detection": {
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