from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from app.schemas.kyc import ErrorCode
from enum import Enum

class ErrorResponse(BaseModel):
    """Standardized error response"""
    status: bool = Field(False, description="Operation status")
    error_message: str = Field(..., description="Human-readable error message")
    error_code: ErrorCode = Field(..., description="Machine-readable error code")
    processing_time: float = Field(..., description="Time taken before error occurred")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": False,
                "error_message": "Invalid image format provided",
                "error_code": "INVALID_IMAGE_FORMAT",
                "processing_time": 0.23,
                "request_id": "req_123456",
                "timestamp": "2024-03-20T10:30:00Z"
            }
        } 

class SpoofError(Exception):
    """Base exception for Spoof Detection errors"""
    def __init__(self, message: str, error_code: ErrorCode, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message) 