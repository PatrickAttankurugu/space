from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, List
from enum import Enum

class ErrorCode(str, Enum):
    """Error codes for KYC and face comparison operations"""
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    LOW_QUALITY_IMAGE = "LOW_QUALITY_IMAGE"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_URL = "INVALID_URL"
    URL_ACCESS_ERROR = "URL_ACCESS_ERROR"
    FILE_CONVERSION_ERROR = "FILE_CONVERSION_ERROR"
    NO_IMAGES_FOUND = "NO_IMAGES_FOUND"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    FACE_DETECTION_FAILED = "FACE_DETECTION_FAILED"
    MODEL_LOADING_ERROR = "MODEL_LOADING_ERROR"
    IMAGE_RESIZE_ERROR = "IMAGE_RESIZE_ERROR"

class ImageQuality(BaseModel):
    """Quality metrics for processed images"""
    overall_quality: float = Field(..., description="Overall image quality score (0-1)", ge=0, le=1)
    width: int = Field(..., description="Image width in pixels", gt=0)
    height: int = Field(..., description="Image height in pixels", gt=0)

class FaceComparisonRequest(BaseModel):
    """Request model for face comparison endpoint"""
    image1: HttpUrl = Field(
        ..., 
        description="URL of first image/document containing face",
        example="https://example.com/face1.jpg"
    )
    image2: HttpUrl = Field(
        ..., 
        description="URL of second image/document containing face",
        example="https://example.com/face2.jpg"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image1": "https://example.com/face1.jpg",
                "image2": "https://example.com/face2.jpg"
            }
        }

class ComparisonResponse(BaseModel):
    """Response model for face comparison results"""
    match: bool = Field(..., description="Whether the faces match")
    face_found: bool = Field(..., description="Whether faces were detected in both images")
    similarity_score: Optional[float] = Field(None, description="Face similarity score (0-100)", ge=0, le=100)
    confidence: float = Field(..., description="Confidence score (0-100)", ge=0, le=100)
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    error_code: Optional[ErrorCode] = Field(None, description="Error code if operation failed")
    match_category: Optional[str] = Field(None, description="Match confidence category")
    image1_quality: Optional[ImageQuality] = Field(None, description="Quality metrics for first image")
    image2_quality: Optional[ImageQuality] = Field(None, description="Quality metrics for second image")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "match": True,
                "face_found": True,
                "similarity_score": 98.2,
                "confidence": 95.5,
                "match_category": "Definite Match - Very High Confidence",
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
                "processing_time": 1.23
            }
        }

class TaskResponse(BaseModel):
    """Response model for asynchronous task submission"""
    task_id: str = Field(..., description="Unique identifier for the submitted task")
    status: str = Field(..., description="Current status of the task")
    message: str = Field(..., description="Status message or description")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "7a1b3b11-3568-424f-8ca9-ce1da94d3344",
                "status": "processing",
                "message": "Face comparison task submitted successfully"
            }
        }

class SingleImageComparisonRequest(BaseModel):
    """Request model for single-image face comparison endpoint"""
    image_url: HttpUrl = Field(
        ..., 
        description="URL of image containing two faces (selfie with ID)",
        example="https://example.com/selfie_with_id.jpg"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/selfie_with_id.jpg"
            }
        }