from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List
from enum import Enum
from pydantic import validator, root_validator, model_validator

class SpoofType(str, Enum):
    PRINT = "print"
    REPLAY = "replay"
    MASK = "mask"
    NONE = "none"

class QualityMetrics(BaseModel):
    """Quality metrics for the analyzed face"""
    brightness: float = Field(..., description="Image brightness score", ge=0, le=1)
    sharpness: float = Field(..., description="Image sharpness score", ge=0, le=1)
    contrast: float = Field(..., description="Image contrast score", ge=0, le=1)

class SpoofDetectionRequest(BaseModel):
    """Request model for spoof detection"""
    image_urls: List[str] = Field(
        min_items=1,
        max_items=3,
        description="List of image URLs (1-3 images) for spoof detection"
    )
    image_url: Optional[str] = Field(None, description="Deprecated: Use image_urls instead")

    @model_validator(mode='before')
    @classmethod
    def check_urls(cls, values):
        if not values.get('image_urls') and not values.get('image_url'):
            raise ValueError("Either image_urls or image_url must be provided")
        if not values.get('image_urls'):
            values['image_urls'] = [values.get('image_url')]
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "image_urls": ["https://example.com/face1.jpg", "https://example.com/face2.jpg"]
            }
        }

class SpoofAnalysisResponse(BaseModel):
    """Response model for spoof detection analysis"""
    is_spoof: bool = Field(..., description="Whether the image is detected as a spoof")
    is_deepfake: bool = Field(..., description="Whether the image is detected as a deepfake (mirrors spoof detection)")
    confidence: float = Field(..., description="Confidence score (0-100)", ge=0, le=100)
    deepfake_percentage: float = Field(..., description="Percentage likelihood of being a deepfake (0-100)", ge=0, le=100)
    spoof_type: Optional[SpoofType] = Field(None, description="Type of spoofing detected if any")
    liveness_score: float = Field(..., description="Liveness detection score", ge=0, le=1)
    quality_score: float = Field(..., description="Overall quality score", ge=0, le=1)
    quality_metrics: QualityMetrics = Field(..., description="Detailed quality metrics")
    num_images_processed: int = Field(..., description="Number of images processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")
    error_code: Optional[str] = Field(None, description="Error code if analysis failed")

    class Config:
        json_schema_extra = {
            "example": {
                "is_spoof": False,
                "is_deepfake": False,
                "confidence": 95.5,
                "deepfake_percentage": 0.0,
                "spoof_type": None,
                "liveness_score": 0.98,
                "quality_score": 0.95,
                "quality_metrics": {
                    "brightness": 0.92,
                    "sharpness": 0.95,
                    "contrast": 0.90
                },
                "num_images_processed": 3,
                "processing_time": 1.2
            }
        } 