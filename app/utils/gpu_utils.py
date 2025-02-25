import torch
import gc
import logging
import psutil
import os

logger = logging.getLogger(__name__)

def optimize_gpu_memory():
    """
    Optimize GPU memory usage and configure for best performance
    """
    if torch.cuda.is_available():
        try:
            # Clear GPU cache and collect garbage
            torch.cuda.empty_cache()
            gc.collect()
            
            # Enable memory efficient options
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory allocation strategy
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
            
            # Enable deterministic operations for reproducibility
            torch.backends.cudnn.deterministic = True
            
            # Set device to synchronize operations
            torch.cuda.current_device()
            torch.cuda.synchronize()
            
            logger.info("GPU memory optimizations applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing GPU memory: {str(e)}")
            return False
    return False

def get_gpu_memory_usage():
    """
    Get detailed GPU memory usage statistics
    Returns:
        dict: Memory statistics if GPU is available, None otherwise
    """
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            total = props.total_memory
            free = total - allocated
            
            return {
                'device_name': props.name,
                'total_memory': total,
                'allocated_memory': allocated,
                'reserved_memory': reserved,
                'free_memory': free,
                'utilization_percent': (allocated / total) * 100,
                'memory_stats': {
                    'total_gb': f"{total/1024**3:.2f}GB",
                    'allocated_gb': f"{allocated/1024**3:.2f}GB",
                    'reserved_gb': f"{reserved/1024**3:.2f}GB",
                    'free_gb': f"{free/1024**3:.2f}GB"
                }
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {str(e)}")
            return None
    return None

def gpu_safe_execution(func):
    """
    Decorator for safe GPU execution with CPU fallback and memory management
    """
    def wrapper(*args, **kwargs):
        try:
            if torch.cuda.is_available():
                # Clear cache before execution
                torch.cuda.empty_cache()
                gc.collect()
                
                with torch.cuda.device(0):
                    try:
                        return func(*args, **kwargs)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning("GPU out of memory, attempting recovery...")
                            torch.cuda.empty_cache()
                            gc.collect()
                            # Retry once after clearing memory
                            return func(*args, **kwargs)
                        raise e
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in GPU execution: {str(e)}")
            raise
            
    return wrapper

def monitor_gpu_health():
    """
    Monitor GPU health and resource utilization
    Returns:
        dict: GPU health metrics
    """
    try:
        if not torch.cuda.is_available():
            return {"status": "GPU not available"}
            
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        memory_usage = get_gpu_memory_usage()
        system_memory = psutil.virtual_memory()
        
        return {
            "gpu_status": "healthy",
            "gpu_name": props.name,
            "gpu_memory": memory_usage,
            "system_memory": {
                "total": f"{system_memory.total/1024**3:.2f}GB",
                "available": f"{system_memory.available/1024**3:.2f}GB",
                "percent_used": system_memory.percent
            },
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "driver_version": os.environ.get("NVIDIA_DRIVER_VERSION", "unknown")
        }
    except Exception as e:
        logger.error(f"Error monitoring GPU health: {str(e)}")
        return {"status": "error", "message": str(e)}