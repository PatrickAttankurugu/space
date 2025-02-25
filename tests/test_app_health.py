import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import torch
import psutil

from app.main import app
from app.services.kyc.face_service import FaceComparisonService
from app.services.document_verification.document_service import DocumentService
from app.services.license_verification.license_service import LicenseService
from app.core.config import settings

client = TestClient(app)

def test_app_health_check_all_healthy():
    """Test main application health check when all services are healthy"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "services" in data
    
    services = data["services"]
    assert "kyc" in services
    assert "license" in services
    assert "document" in services
    
    system = data["system"]
    assert "device" in system
    assert system["environment"] == settings.ENV
    assert system["version"] == settings.VERSION
    assert "memory" in system
    assert isinstance(system["memory"]["available"], float)
    assert isinstance(system["memory"]["total"], float)

@pytest.mark.asyncio
async def test_app_health_check_partial_unhealthy(mocker):
    """Test main health check when some services are unhealthy"""
    mock_kyc = {
        "status": "unhealthy",
        "error": "Face service unavailable",
        "error_code": "SERVICE_ERROR",
        "services": {
            "face_comparison": {
                "status": "unhealthy",
                "error": "Face service unavailable",
                "error_code": "SERVICE_ERROR"
            }
        }
    }
    mocker.patch('app.main.kyc_health_check', return_value=mock_kyc)
    
    response = client.get("/health")
    assert response.status_code == 503
    
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["services"]["kyc"]["status"] == "unhealthy"
    assert data["services"]["license"]["status"] == "healthy"
    assert data["services"]["document"]["status"] == "healthy"

@pytest.mark.asyncio
async def test_app_health_check_all_unhealthy(mocker):
    """Test main health check when all services are unhealthy"""
    # Mock all services to return unhealthy status
    mock_unhealthy = {
        "status": "unhealthy",
        "error": "Service unavailable",
        "error_code": "SERVICE_ERROR"
    }
    
    mocker.patch.object(FaceComparisonService, 'check_health', return_value=mock_unhealthy)
    mocker.patch.object(DocumentService, 'check_health', return_value=mock_unhealthy)
    mocker.patch.object(LicenseService, 'check_health', return_value=mock_unhealthy)
    
    response = client.get("/health")
    assert response.status_code == 503
    
    data = response.json()
    assert data["status"] == "unhealthy"
    assert all(service["status"] == "unhealthy" for service in data["services"].values())

def test_app_health_check_response_structure():
    """Test main health check response structure"""
    response = client.get("/health")
    data = response.json()
    
    required_fields = {
        "status", "timestamp", "services", "system"
    }
    assert all(field in data for field in required_fields)
    
    services = {"kyc", "license", "document"}
    assert all(service in data["services"] for service in services)
    
    system_fields = {
        "device", "environment", "version", "memory"
    }
    assert all(field in data["system"] for field in system_fields) 