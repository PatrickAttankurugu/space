from fastapi import APIRouter
from app.api.v1.endpoints import (
    kyc,
    document_verification,
    spoof_detection
)

api_router = APIRouter()

# Register routers
api_router.include_router(kyc.router, prefix="/kyc", tags=["KYC"])
api_router.include_router(document_verification.router, prefix="/documents", tags=["Documents"])
api_router.include_router(spoof_detection.router, prefix="/spoof", tags=["Spoof Detection"])