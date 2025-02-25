import torch
import logging
import gc

logger = logging.getLogger(__name__)

def get_optimal_device():
    """
    Determine the optimal device for model inference.
    Returns:
        torch.device: The optimal device (CUDA if available, CPU otherwise)
    """
    if torch.cuda.is_available():
        try:
            # Initialize CUDA
            torch.cuda.init()
            
            # Clear any existing cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Test CUDA memory allocation
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
            
            # Configure CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info(f"CUDA is available and working - using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
            
        except Exception as e:
            logger.error(f"CUDA initialization error: {str(e)}")
            logger.warning("Falling back to CPU despite CUDA being available")
            return torch.device("cpu")
    else:
        logger.warning("CUDA is not available - using CPU")
        return torch.device("cpu")

def get_device_info():
    """
    Get detailed information about the current device configuration.
    Returns:
        dict: Device information including type, memory, etc.
    """
    try:
        device = get_optimal_device()
        info = {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_initialized": torch.cuda.is_initialized() if torch.cuda.is_available() else False,
        }
        
        if device.type == "cuda":
            current_device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(current_device)
            
            info.update({
                "gpu_name": properties.name,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_total_memory": f"{properties.total_memory/1024**3:.2f}GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(current_device)/1024**3:.2f}GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved(current_device)/1024**3:.2f}GB",
                "gpu_free_memory": f"{(properties.total_memory - torch.cuda.memory_allocated(current_device))/1024**3:.2f}GB",
                "gpu_memory_utilization": f"{(torch.cuda.memory_allocated(current_device)/properties.total_memory)*100:.2f}%",
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32
            })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting device info: {str(e)}")
        return {"error": str(e), "device": "cpu"}

def verify_cuda_operation():
    """
    Verify CUDA operations are working correctly.
    Returns:
        bool: True if CUDA operations are working, False otherwise
    """
    if not torch.cuda.is_available():
        return False
        
    try:
        # Test basic operations
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.error(f"CUDA operation verification failed: {str(e)}")
        return False