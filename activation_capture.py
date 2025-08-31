import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from transformer_lens import HookedTransformer
from .model_loader import ModelLoader


class ActivationCapturer:
    """Capture and store model activations for cognitive pattern analysis."""
    
    def __init__(self, model_name: str = "google/gemma-2-2b-it", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model_loader = ModelLoader(model_name, device)
        self.model = None
        self.activations = {}
        
    def load_model(self, local_path: Optional[str] = None) -> None:
        """Load the model for activation capture."""
        self.model = self.model_loader.load_model(local_path)
        
    def capture_activations(
        self, 
        strings: List[str], 
        layer_nums: List[int] = [23, 29],
        cognitive_pattern: str = "default",
        position: str = "last"
    ) -> Dict[str, torch.Tensor]:
        """
        Capture activations for given strings at specified layers.
        
        Args:
            strings: List of input strings
            layer_nums: Layers to capture activations from
            cognitive_pattern: Pattern category for organization
            position: Token position to extract ('last', 'all', or int)
            
        Returns:
            Dictionary of activations organized by layer and pattern
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        all_activations = {}
        
        for string in strings:
            # Tokenize input
            tokens = self.model.to_tokens(string, prepend_bos=True)
            
            # Run with cache to capture activations
            logits, cache = self.model.run_with_cache(tokens, remove_batch_dim=True)
            
            for layer_num in layer_nums:
                layer_key = f"layer_{layer_num}"
                pattern_key = f"{cognitive_pattern}_{layer_key}"
                
                if pattern_key not in all_activations:
                    all_activations[pattern_key] = []
                
                # Get residual stream activations
                if f"blocks.{layer_num}.hook_resid_post" in cache:
                    activation = cache[f"blocks.{layer_num}.hook_resid_post"]
                else:
                    # Fallback to residual pre
                    activation = cache[f"blocks.{layer_num}.hook_resid_pre"]
                
                # Extract position-specific activations
                if position == "last":
                    pos_activation = activation[-1, :].detach()
                elif position == "all":
                    pos_activation = activation.detach()
                elif isinstance(position, int):
                    pos_activation = activation[position, :].detach()
                else:
                    pos_activation = activation[-1, :].detach()  # default to last
                
                all_activations[pattern_key].append(pos_activation)
        
        # Convert lists to tensors
        for key in all_activations:
            all_activations[key] = torch.stack(all_activations[key])
        
        # Store in instance
        self._update_stored_activations(all_activations, cognitive_pattern)
        
        return all_activations
    
    def _update_stored_activations(self, new_activations: Dict[str, torch.Tensor], pattern: str) -> None:
        """Update stored activations with new captures."""
        if pattern not in self.activations:
            self.activations[pattern] = {}
            
        for key, tensor in new_activations.items():
            if key in self.activations[pattern]:
                # Concatenate with existing
                self.activations[pattern][key] = torch.cat([
                    self.activations[pattern][key], tensor
                ], dim=0)
            else:
                self.activations[pattern][key] = tensor
    
    def get_activations(self, cognitive_pattern: str, layer_num: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Retrieve stored activations for a cognitive pattern."""
        if cognitive_pattern not in self.activations:
            return {}
            
        if layer_num is not None:
            layer_key = f"{cognitive_pattern}_layer_{layer_num}"
            return {layer_key: self.activations[cognitive_pattern].get(layer_key, torch.empty(0))}
        
        return self.activations[cognitive_pattern]
    
    def save_activations(self, filepath: str, cognitive_pattern: Optional[str] = None) -> None:
        """Save activations to disk."""
        data_to_save = self.activations
        if cognitive_pattern:
            data_to_save = {cognitive_pattern: self.activations.get(cognitive_pattern, {})}
            
        torch.save(data_to_save, filepath)
    
    def load_activations(self, filepath: str) -> None:
        """Load activations from disk."""
        self.activations = torch.load(filepath, map_location=self.device)
    
    def clear_activations(self, cognitive_pattern: Optional[str] = None) -> None:
        """Clear stored activations."""
        if cognitive_pattern:
            self.activations.pop(cognitive_pattern, None)
        else:
            self.activations.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get loaded model information."""
        return self.model_loader.get_model_info()