from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

class SystemInfo(BaseModel):
    device: str
    cuda_available: bool
    cuda_device_count: int
    environment: str
    version: str
    memory: Optional[Dict[str, Any]] = None

class ServiceStatus(BaseModel):
    status: str
    error: Optional[str] = None
    error_code: Optional[str] = None
    model_loaded: Optional[bool] = None
    model_working: Optional[bool] = None
    service_available: Optional[bool] = None

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, ServiceStatus]
    system: SystemInfo 