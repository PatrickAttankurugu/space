from pydantic import BaseModel, HttpUrl
from typing import Optional

class ErrorCode:
    INVALID_URL = "INVALID_URL"
    CONVERSION_ERROR = "CONVERSION_ERROR"
    VERIFICATION_ERROR = "VERIFICATION_ERROR"
    NO_IMAGES_FOUND = "NO_IMAGES_FOUND"

class LicenseRequest(BaseModel):
    image_url: HttpUrl

class LicenseResponse(BaseModel):
    is_valid: bool
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    error_code: Optional[str] = None