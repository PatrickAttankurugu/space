from fastapi import APIRouter
from .kyc import router as kyc_router
from .document_verification import router as document_router
from . import health
from . import kyc
from . import document_verification
from . import spoof_detection

api_router = APIRouter()

api_router.include_router(kyc_router, prefix="/kyc", tags=["KYC"])
api_router.include_router(document_router, prefix="/document", tags=["Document"])

__all__ = [
    "health",
    "kyc",
    "document_verification",
    "spoof_detection"
]
