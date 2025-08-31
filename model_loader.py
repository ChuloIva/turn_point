import torch
from transformer_lens import HookedTransformer
from typing import Optional, Union, Dict, Any


class ModelLoader:
    """Modular model loading for multiple transformer architectures."""
    
    SUPPORTED_MODELS = {
        "google/gemma-2-2b-it": "gemma-2-2b-it",
        "google/gemma-2-9b-it": "gemma-2-9b-it", 
        "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3"
    }
    
    def __init__(self, model_name: str = "google/gemma-2-2b-it", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self, local_path: Optional[str] = None) -> HookedTransformer:
        """Load transformer model with TransformerLens."""
        if local_path:
            # Load from local path
            model = HookedTransformer.from_pretrained_no_processing(
                local_path,
                device=self.device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        self.model = model
        return model
    
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