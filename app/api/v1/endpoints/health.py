from fastapi import APIRouter, HTTPException
from datetime import datetime
import torch
import psutil

from app.core.config import settings

router = APIRouter()

@router.get(
    "/",
    summary="Basic Health Check",
    description="Returns the basic health status of the API"
)
async def health_check():
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "environment": settings.ENV,
                "version": settings.VERSION,
                "memory": {
                    "available": float(psutil.virtual_memory().available / (1024 * 1024 * 1024)),
                    "total": float(psutil.virtual_memory().total / (1024 * 1024 * 1024))
                }
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )
