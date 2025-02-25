from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any
from enum import Enum

class ErrorCode(str, Enum):
    """Error codes for document verification operations"""
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    LOW_QUALITY_IMAGE = "LOW_QUALITY_IMAGE"
    INSUFFICIENT_FEATURES = "INSUFFICIENT_FEATURES"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_URL = "INVALID_URL"
    URL_ACCESS_ERROR = "URL_ACCESS_ERROR"
    FILE_CONVERSION_ERROR = "FILE_CONVERSION_ERROR"
    NO_IMAGES_FOUND = "NO_IMAGES_FOUND"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    OCR_EXTRACTION_ERROR = "OCR_EXTRACTION_ERROR"
    CARD_VALIDATION_ERROR = "CARD_VALIDATION_ERROR"

class GhanaCardRequest(BaseModel):
    """Request model for Ghana card verification"""
    card_with_selfie: HttpUrl = Field(
        ..., 
        description="HTTPS URL of image showing Ghana card front with selfie. Only the card portion will be used for validation.",
        example="https://example.com/card_with_selfie.jpg"
    )
    card_front: HttpUrl = Field(
        ..., 
        description="HTTPS URL of clear, straight-on photo of Ghana card front for information extraction. Should be well-lit and glare-free.",
        example="https://example.com/card_front.jpg"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "card_with_selfie": "https://example.com/card_with_selfie.jpg",
                "card_front": "https://example.com/card_front.jpg"
            },
            "description": """
            Both image URLs must:
            - Use HTTPS protocol
            - Be publicly accessible
            - Return valid image files (JPG, PNG)
            - Have sufficient resolution (minimum 300 DPI recommended)
            - Card front image must be clear and glare-free for information extraction
            """
        }

class CardInfo(BaseModel):
    """Model for extracted card information"""
    surname: Optional[str] = Field(None, description="Cardholder's surname")
    given_names: Optional[str] = Field(None, description="Cardholder's given names")
    nationality: Optional[str] = Field(None, description="Cardholder's nationality")
    id_number: Optional[str] = Field(None, description="Personal ID number (GHA format)")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (DD/MM/YYYY)")
    sex: Optional[str] = Field(None, description="Gender (M/F)")
    date_of_issue: Optional[str] = Field(None, description="Card issue date (DD/MM/YYYY)")
    date_of_expiry: Optional[str] = Field(None, description="Card expiry date (DD/MM/YYYY)")
    height: Optional[str] = Field(None, description="Height in meters")
    document_number: Optional[str] = Field(None, description="Document number")
    place_of_issuance: Optional[str] = Field(None, description="Place where card was issued")

class OCRResult(BaseModel):
    """Result of OCR processing"""
    success: bool = Field(..., description="Whether information extraction was successful")
    card_info: Optional[CardInfo] = Field(None, description="Extracted card information")
    confidence: float = Field(..., description="Overall extraction confidence score")
    message: str = Field(..., description="Processing status or error message")

class VerificationResponse(BaseModel):
    """Response model for document verification results"""
    is_valid: bool = Field(..., description="Whether the card is valid")
    confidence: float = Field(..., description="Overall confidence score (0-100)")
    detected_features: List[str] = Field(..., description="Security features detected on the card")
    feature_probabilities: Dict[str, float] = Field(..., description="Confidence scores for each detected feature")
    num_features_detected: int = Field(..., description="Number of security features detected")
    card_info: Optional[CardInfo] = Field(None, description="Extracted card information")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    error_code: Optional[ErrorCode] = Field(None, description="Error code if operation failed")
    processing_time: float = Field(..., description="Total processing time in seconds")

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
                "card_info": {
                    "surname": "DOE",
                    "given_names": "JOHN",
                    "nationality": "GHANAIAN",
                    "id_number": "GHA-123456789-0",
                    "date_of_birth": "17/07/1995",
                    "sex": "M",
                    "date_of_issue": "08/09/2020",
                    "date_of_expiry": "08/09/2030",
                    "height": "1.75",
                    "document_number": "AR5151853",
                    "place_of_issuance": "ACCRA"
                },
                "processing_time": 1.23
            }
        }