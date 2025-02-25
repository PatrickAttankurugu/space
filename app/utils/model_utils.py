import torch
import os
from typing import Optional, Dict, Any
import logging
from app.utils.device_utils import get_optimal_device
from pathlib import Path

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Get the appropriate device for model inference"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def verify_model_files(model_dir: str, required_files: list) -> dict:
    """Verify that all required model files exist"""
    status = {}
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        status[file] = os.path.exists(file_path)
    return status

class ModelLoader:
    @staticmethod
    def load_model(model: torch.nn.Module, weights_path: str, force_cpu: bool = False) -> torch.nn.Module:
        """
        Load a model with proper device handling
        """
        device = torch.device("cpu") if force_cpu else get_optimal_device()
        try:
            state_dict = torch.load(weights_path, map_location=device)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
            
            model = model.to(device)
            model.eval()
            logger.info(f"Model loaded successfully on {device}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
        """
        Get information about the model's device and memory usage
        """
        device = next(model.parameters()).device
        info = {
            "device": str(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        if device.type == "cuda":
            info.update({
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(device.index)/1024**3:.2f}GB",
                "gpu_memory_cached": f"{torch.cuda.memory_cached(device.index)/1024**3:.2f}GB"
            })
        
        return info 

def load_model_with_device_fallback(
    model: torch.nn.Module,
    weights_path: Path,
    force_cpu: bool = False
) -> torch.nn.Module:
    """
    Load a model with automatic device selection and fallback
    """
    device = torch.device("cpu") if force_cpu else get_optimal_device()
    
    try:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
            
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        if device.type == "cuda":
            logger.warning(f"Failed to load model on GPU: {e}. Falling back to CPU.")
            return load_model_with_device_fallback(model, weights_path, force_cpu=True)
        else:
            logger.error(f"Failed to load model: {e}")
            raise 