import psutil
import torch
import logging
from typing import Dict
from .gpu_utils import get_gpu_memory_usage

logger = logging.getLogger(__name__)

def get_system_resources() -> Dict[str, any]:
    """
    Get current system resource usage including CPU, RAM, and GPU if available
    """
    resources = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "disk_usage_percent": psutil.disk_usage('/').percent
    }
    
    # Add detailed GPU information if available
    if torch.cuda.is_available():
        try:
            gpu_memory = get_gpu_memory_usage()
            resources.update({
                "gpu_info": {
                    "name": torch.cuda.get_device_name(0),
                    "memory": gpu_memory,
                    "utilization": torch.cuda.utilization(),
                    "temperature": torch.cuda.temperature()
                }
            })
        except Exception as e:
            logger.error(f"Error getting GPU information: {str(e)}")
    
    return resources 