import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import torch

from app.main import app
from app.services.kyc.face_service import FaceComparisonService
from app.core.config import settings

client = TestClient(app)

def test_kyc_health_check_healthy():
    """Test KYC health check when service is healthy"""
    response = client.get("/api/v1/kyc/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "services" in data
    assert "face_comparison" in data["services"]
    
    face_service = data["services"]["face_comparison"]
    assert face_service["status"] == "healthy"
    assert face_service["version"] == settings.VERSION
    assert "model_loaded" in face_service
    assert "model_working" in face_service
    
    system = data["system"]
    assert system["device"] == "cpu"
    assert isinstance(system["cuda_available"], bool)
    assert isinstance(system["cuda_device_count"], int)
    assert system["environment"] == settings.ENV
    assert system["version"] == settings.VERSION

@pytest.mark.asyncio
async def test_kyc_health_check_unhealthy(mocker):
    """Test KYC health check when service is unhealthy"""
    mock_health = {
        "status": "unhealthy",
        "error": "Model initialization failed",
        "error_code": "SERVICE_ERROR",
        "model_loaded": False,
        "model_working": False,
        "version": settings.VERSION
    }
    mocker.patch.object(FaceComparisonService, 'check_health', return_value=mock_health)
    
    response = client.get("/api/v1/kyc/health")
    assert response.status_code == 503
    
    data = response.json()
    assert data["status"] == "unhealthy"
    assert "error" in data
    assert data["error_code"] == "SERVICE_ERROR"
    assert data["services"]["face_comparison"]["status"] == "unhealthy"

def test_kyc_health_check_response_structure():
    """Test KYC health check response matches expected structure"""
    response = client.get("/api/v1/kyc/health")
    data = response.json()
    
    required_fields = {
        "status", "timestamp", "services", "system"
    }
    assert all(field in data for field in required_fields)
    
    service_fields = {
        "status", "version", "models", "model_loaded", "model_working"
    }
    assert all(field in data["services"]["face_comparison"] for field in service_fields)
    
    system_fields = {
        "device", "environment", "version", "cuda_available", 
        "cuda_device_count", "memory"
    }
    assert all(field in data["system"] for field in system_fields)
