from celery import Task
import torch
import logging
from contextlib import contextmanager
import asyncio
import os
import gc

logger = logging.getLogger(__name__)

class GPUTask(Task):
    """Base class for ML tasks with GPU optimization"""
    abstract = True
    _gpu_id = 0
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            try:
                # Clear GPU memory first
                torch.cuda.empty_cache()
                gc.collect()
                
                # Configure CUDA optimizations
                torch.cuda.set_device(self._gpu_id)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Set memory allocation strategy
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                
                # Initialize CUDA context
                torch.cuda.init()
                
                # Log GPU information
                logger.info(f"CUDA initialized successfully. Device: {self.device}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
                
                # Test CUDA operations
                test_tensor = torch.zeros(1, device=self.device)
                del test_tensor
                
            except Exception as e:
                logger.error(f"Error initializing CUDA: {str(e)}")
                logger.warning("Falling back to CPU")
                self.device = torch.device('cpu')
    
    def initialize_gpu(self):
        """Initialize GPU settings for task execution"""
        if self.device.type == 'cuda':
            try:
                # Set thread settings
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'
                
                # Configure CUDA providers
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self._gpu_id)
                os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
                os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility,video'
                
                # Set ONNX Runtime providers
                os.environ['ONNXRUNTIME_PROVIDERS'] = '["CUDAExecutionProvider", "CPUExecutionProvider"]'
                os.environ['ONNXRUNTIME_DEVICE'] = 'cuda:0'
                os.environ['ONNXRUNTIME_CUDA_DEVICE_ID'] = '0'
                
                return True
            except Exception as e:
                logger.error(f"Error in GPU initialization: {str(e)}")
                return False
        return False
    
    @contextmanager
    def gpu_context(self):
        """Context manager for GPU operations with memory management"""
        if self.device.type == 'cuda':
            try:
                # Clear GPU memory before task
                torch.cuda.empty_cache()
                gc.collect()
                
                # Initialize GPU settings
                self.initialize_gpu()
                
                with torch.cuda.device(self._gpu_id):
                    try:
                        yield
                    except Exception as e:
                        logger.error(f"Error in GPU task execution: {str(e)}")
                        raise
                    finally:
                        # Clear GPU memory after task
                        torch.cuda.empty_cache()
                        gc.collect()
            finally:
                torch.cuda.empty_cache()
        else:
            yield
            
    def __call__(self, *args, **kwargs):
        """Execute task with GPU context management"""
        try:
            with self.gpu_context():
                return self.run(*args, **kwargs)
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            raise
        finally:
            if self.device.type == 'cuda':
                # Final cleanup
                torch.cuda.empty_cache()
                gc.collect()
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failures"""
        if self.device.type == 'cuda':
            # Cleanup on failure
            torch.cuda.empty_cache()
            gc.collect()
        super().on_failure(exc, task_id, args, kwargs, einfo)

def run_async(coro):
    """Helper function to run coroutines in a sync context"""
    try:
        return asyncio.get_event_loop().run_until_complete(coro)
    except RuntimeError:
        # Handle case where no event loop exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()