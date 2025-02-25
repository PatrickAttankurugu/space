from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any
from enum import Enum

class ErrorCode(str, Enum):
    """Error codes for document verification operations"""
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    LOW_QUALITY_IMAGE = "LOW_QUALITY_IMAGE"
    INSUFFICIENT_FEATURES = "INSUFFICIENT_FEATURES"
    MRZ_NOT_READABLE = "MRZ_NOT_READABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_URL = "INVALID_URL"
    URL_ACCESS_ERROR = "URL_ACCESS_ERROR"
    FILE_CONVERSION_ERROR = "FILE_CONVERSION_ERROR"
    NO_IMAGES_FOUND = "NO_IMAGES_FOUND"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"

class GhanaCardRequest(BaseModel):
    """Request model for Ghana card verification"""
    front_image: HttpUrl = Field(
        ..., 
        description="HTTPS URL of the front side of Ghana card. Must be publicly accessible and return a valid image file.",
        example="https://example.com/card_front.jpg"
    )
    back_image: HttpUrl = Field(
        ..., 
        description="HTTPS URL of the back side of Ghana card. Must be publicly accessible and return a valid image file.",
        example="https://example.com/card_back.jpg"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "front_image": "https://example.com/card_front.jpg",
                "back_image": "https://example.com/card_back.jpg"
            },
            "description": """
            Both image URLs must:
            - Use HTTPS protocol
            - Be publicly accessible
            - Return valid image files (JPG, PNG)
            - Have sufficient resolution for feature detection
            """
        }

class OCRResult(BaseModel):
    success: bool
    id_number: Optional[str]
    confidence: float
    message: str

class VerificationResponse(BaseModel):
    """Response model for document verification results"""
    is_valid: bool
    confidence: float
    detected_features: List[str]
    feature_probabilities: Dict[str, float]
    num_features_detected: int
    mrz_data: Optional[Dict[str, Any]] = None
    id_number: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    processing_time: float

    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "confidence": 95.5,
                "detected_features": ["ECOWAS Logo", "Ghana Coat of Arms", "Ghana Flag"],
                "feature_probabilities": {
                    "ECOWAS Logo": 0.98,
                    "Ghana Coat of Arms": 0.95,
                    "Ghana Flag": 0.93
                },
                "num_features_detected": 3,
                "mrz_data": {
                    "Document Type": "ID",
                    "Country Code": "GHA",
                    "Document Number": "GHA-123456789-0",
                    "Surname": "DOE",
                    "Given Names": "JOHN",
                    "Nationality": "GHA",
                    "Date of Birth": "17th July, 1995",
                    "Gender": "M",
                    "Expiry Date": "8th September, 2030"
                },
                "id_number": {
                    "success": True,
                    "id_number": "GHA-123456789-0",
                    "confidence": 95.5,
                    "message": "Card valid and MRZ readable"
                },
                "processing_time": 1.23
            }
        }