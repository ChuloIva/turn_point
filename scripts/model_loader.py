import torch
from transformer_lens import HookedTransformer
from typing import Optional, Union, Dict, Any
import logging
from utils.device_detection import get_device_manager

logger = logging.getLogger(__name__)


class ModelLoader:
    """Modular model loading for multiple transformer architectures with advanced device support."""
    
    SUPPORTED_MODELS = {
        "google/gemma-2-2b-it": "gemma-2-2b-it",
        "google/gemma-2-9b-it": "gemma-2-9b-it", 
        "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3"
    }
    
    def __init__(self, model_name: str = "google/gemma-2-2b-it", device: str = "auto"):
        self.model_name = model_name
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device(device)
        self.device_type = self.device_manager.optimal_device[0] if device == "auto" else device
        self.model = None
        self.tokenizer = None
        
        # Log device selection
        logger.info(f"ModelLoader initialized with device: {self.device} (type: {self.device_type})")
        
    def load_model(self, local_path: Optional[str] = None) -> HookedTransformer:
        """Load transformer model with TransformerLens and optimal device/dtype selection."""
        # Get optimal dtype for the selected device
        torch_dtype = self.device_manager.get_torch_dtype(self.device_type)
        
        logger.info(f"Loading model with device={self.device}, dtype={torch_dtype}")
        
        # Handle MLX-specific loading if available
        if self.device_type == 'mlx' and self.device_manager.device_info['mlx'].get('mlx_installed'):
            return self._load_model_mlx(local_path)
        
        # Standard PyTorch/TransformerLens loading
        if local_path:
            # Load from local path
            model = HookedTransformer.from_pretrained_no_processing(
                local_path,
                device=self.device,
                torch_dtype=torch_dtype
            )
        else:
            # Load from HuggingFace
            if self.model_name not in self.SUPPORTED_MODELS:
                raise ValueError(f"Model {self.model_name} not supported. "
                               f"Supported models: {list(self.SUPPORTED_MODELS.keys())}")
            
            tl_name = self.SUPPORTED_MODELS[self.model_name]
            model = HookedTransformer.from_pretrained(
                tl_name,
                device=self.device,
                torch_dtype=torch_dtype
            )
        
        self.model = model
        logger.info(f"Model loaded successfully on {self.device}")
        return model
    
    def _load_model_mlx(self, local_path: Optional[str] = None) -> HookedTransformer:
        """Load model with MLX optimizations for Apple Silicon."""
        logger.info("Loading model with MLX optimizations for Apple Silicon")
        
        try:
            import mlx.core as mx
            
            # For now, fall back to standard loading but with CPU/MPS
            # Future: implement native MLX model loading
            fallback_device = 'mps' if self.device_manager.device_info['mps']['available'] else 'cpu'
            
            logger.info(f"MLX optimization not yet fully implemented, using {fallback_device}")
            
            if local_path:
                model = HookedTransformer.from_pretrained_no_processing(
                    local_path,
                    device=fallback_device,
                    torch_dtype=torch.float16 if fallback_device == 'mps' else torch.float32
                )
            else:
                if self.model_name not in self.SUPPORTED_MODELS:
                    raise ValueError(f"Model {self.model_name} not supported. "
                                   f"Supported models: {list(self.SUPPORTED_MODELS.keys())}")
                
                tl_name = self.SUPPORTED_MODELS[self.model_name]
                model = HookedTransformer.from_pretrained(
                    tl_name,
                    device=fallback_device,
                    torch_dtype=torch.float16 if fallback_device == 'mps' else torch.float32
                )
            
            return model
            
        except ImportError:
            logger.warning("MLX not available, falling back to standard loading")
            return self.load_model(local_path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        return {
            "n_layers": self.model.cfg.n_layers,
            "d_model": self.model.cfg.d_model,
            "n_heads": self.model.cfg.n_heads,
            "d_head": self.model.cfg.d_head,
            "vocab_size": self.model.cfg.d_vocab,
            "context_length": self.model.cfg.n_ctx
        }