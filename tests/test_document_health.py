import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import torch

from app.main import app
from app.services.document_verification.document_service import DocumentService
from app.core.config import settings

client = TestClient(app)

def test_document_health_check_healthy():
    """Test document verification health check when service is healthy"""
    response = client.get("/api/v1/document/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "services" in data
    assert "document_verification" in data["services"]
    
    doc_service = data["services"]["document_verification"]
    assert doc_service["status"] == "healthy"
    assert "model_loaded" in doc_service
    assert "models" in doc_service
    assert "ghana_card_modelV2.pth" in doc_service["models"]
    
    system = data["system"]
    assert isinstance(system["cuda_available"], bool)
    assert isinstance(system["cuda_device_count"], int)
    assert system["environment"] == settings.ENV
    assert system["version"] == settings.VERSION

@pytest.mark.asyncio
async def test_document_health_check_unhealthy(mocker):
    """Test document verification health check when service is unhealthy"""
    # Mock document service to return unhealthy status
    mock_health = {
        "status": "unhealthy",
        "error": "Model failed to load",
        "error_code": "SERVICE_ERROR",
        "model_loaded": False,
        "models": {"ghana_card_modelV2.pth": False}
    }
    mocker.patch.object(DocumentService, 'check_health', return_value=mock_health)
    
    response = client.get("/api/v1/document/health")
    assert response.status_code == 503
    
    data = response.json()
    assert data["status"] == "unhealthy"
    assert "error" in data
    assert data["error_code"] == "SERVICE_ERROR"
    assert data["services"]["document_verification"]["status"] == "unhealthy"

def test_document_health_check_model_verification():
    """Test document verification model file check"""
    response = client.get("/api/v1/document/health")
    data = response.json()
    
    doc_service = data["services"]["document_verification"]
    assert "models" in doc_service
    model_status = doc_service["models"]
    assert "ghana_card_modelV2.pth" in model_status
    assert isinstance(model_status["ghana_card_modelV2.pth"], bool) 