from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
import time
import logging
import os
from datetime import datetime
import psutil
import torch
import json
from app.services.kyc.face_service import face_service
from app.utils.gpu_utils import optimize_gpu_memory, get_gpu_memory_usage
from app.utils.device_utils import get_device_info, verify_cuda_operation
from app.celery_app.celery import celery_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Import routers
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.kyc import router as kyc_router
from app.api.v1.endpoints.document_verification import router as document_verification_router
from app.api.v1.endpoints.spoof_detection import router as spoof_detection_router

# Enhanced GPU detection and configuration
if torch.cuda.is_available():
    try:
        # Verify CUDA operations
        if verify_cuda_operation():
            logger.info("CUDA operations verified successfully")
            
            # Get GPU info
            device_info = get_device_info()
            gpu_name = device_info.get('gpu_name', 'Unknown GPU')
            logger.info(f"Optimized settings applied for {gpu_name}")
            
            # Optimize GPU memory
            if optimize_gpu_memory():
                logger.info("GPU memory optimization successful")
                
                # Get and log GPU memory info
                memory_info = get_gpu_memory_usage()
                if memory_info:
                    logger.info(f"Initial GPU memory state: {json.dumps(memory_info, indent=2)}")
            else:
                logger.warning("GPU memory optimization failed, using default settings")
        else:
            raise RuntimeError("CUDA verification failed")
            
    except Exception as e:
        logger.error(f"Error configuring GPU: {str(e)}")
        logger.error("Falling back to CPU mode due to GPU initialization failure")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
else:
    logger.warning("No GPU detected - running in CPU mode")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ML API Service for KYC and Document Verification"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure number of workers based on available resources
try:
    if torch.cuda.is_available():
        num_workers = 1  # Single worker for GPU workload
        logger.info("GPU detected - configuring single worker for GPU workload")
    else:
        workers_per_core = 0.5  # Half the vCPUs
        num_cores = os.cpu_count() or 1  # Fallback to 1 if detection fails
        num_workers = max(int(num_cores * workers_per_core), 1)
        logger.info(f"CPU mode - configuring {num_workers} workers across {num_cores} cores")
    
    logger.info(f"Worker configuration: {num_workers} workers")
except Exception as e:
    num_workers = 1  # Fallback to single worker
    logger.error(f"Error configuring workers: {str(e)}")
    logger.info("Falling back to single worker configuration")

# Include routers
app.include_router(health_router, prefix="/health", tags=["Health Check"])
app.include_router(kyc_router, prefix="/api/v1/kyc", tags=["KYC"])
app.include_router(document_verification_router, prefix="/api/v1/document", tags=["Document Verification"])
app.include_router(spoof_detection_router, prefix="/api/v1/spoof", tags=["Spoof Detection"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Log device information
        device_info = get_device_info()
        logger.info(f"Device configuration: {json.dumps(device_info, indent=2)}")
        
        # Initialize face service
        logger.info("Face service initialized successfully")

        # Test Celery connection
        try:
            celery_ping = celery_app.control.inspect().ping()
            if celery_ping:
                logger.info("Celery connection successful")
            else:
                logger.warning("Celery workers not responding")
        except Exception as ce:
            logger.warning(f"Could not connect to Celery: {str(ce)}")
        
        # Log startup completion
        logger.info(f"Application startup completed successfully on {device_info['device']}")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # Clean GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
        
        # Clean temporary files
        if settings.CLEANUP_TEMP_FILES:
            settings.cleanup()
            logger.info("Temporary files cleaned")
            
        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

@app.get("/gpu-info")
async def gpu_information():
    """Get detailed GPU information"""
    try:
        device_info = get_device_info()
        if torch.cuda.is_available():
            memory_info = get_gpu_memory_usage()
            return {
                "status": "active",
                "device_info": device_info,
                "memory_info": memory_info
            }
        return {
            "status": "inactive",
            "device_info": device_info
        }
    except Exception as e:
        logger.error(f"Error getting GPU information: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health/system")
async def system_health():
    try:
        # Update these lines to use the router objects
        kyc_response = await kyc_router.health_check()
        doc_response = await document_verification_router.health_check()
        spoof_response = await spoof_detection_router.health_check()
        
        def get_response_data(response):
            if isinstance(response, dict):
                return response
            if isinstance(response, JSONResponse):
                return json.loads(response.body.decode())
            return {"services": {"status": "unhealthy"}}
        
        kyc_data = get_response_data(kyc_response)
        doc_data = get_response_data(doc_response)
        spoof_data = get_response_data(spoof_response)
        
        services = {
            "kyc": kyc_data.get("services", {}).get("face_comparison", {"status": "unhealthy"}),
            "document": doc_data.get("services", {}).get("document_verification", {"status": "unhealthy"}),
            "spoof": spoof_data.get("services", {}).get("spoof_detection", {"status": "unhealthy"})
        }
        
        # Check Celery status
        try:
            celery_ping = celery_app.control.inspect().ping()
            services["celery"] = {
                "status": "healthy" if celery_ping else "unhealthy",
                "workers_responding": bool(celery_ping)
            }
        except Exception as ce:
            services["celery"] = {
                "status": "unhealthy",
                "error": str(ce)
            }
        
        all_healthy = all(service.get("status") == "healthy" for service in services.values())
        
        device_info = get_device_info()
        memory_info = get_gpu_memory_usage() if torch.cuda.is_available() else None
        
        response = {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services,
            "system": {
                **device_info,
                "environment": settings.ENV,
                "version": settings.VERSION,
                "memory": {
                    "system": {
                        "available": float(psutil.virtual_memory().available / (1024**3)),
                        "total": float(psutil.virtual_memory().total / (1024**3))
                    },
                    "gpu": memory_info['memory_stats'] if memory_info else None
                }
            }
        }
        
        return JSONResponse(
            status_code=200 if all_healthy else 503,
            content=response
        )
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "error_code": "SYSTEM_ERROR",
                "system": get_device_info()
            }
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Process Time: {process_time:.4f}s"
    )
    return response