import torch
import numpy as np
import os
import hashlib
from typing import List, Dict, Optional, Union, Tuple, Any
from transformer_lens import HookedTransformer
from model_loader import ModelLoader
from utils.device_detection import get_device_manager
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ActivationCapturer:
    """Capture and store model activations for cognitive pattern analysis with multi-device support."""
    
    def __init__(self, model_name: str = "google/gemma-2-2b-it", device: str = "auto", cache_dir: str = "./activations/"):
        self.model_name = model_name
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device(device)
        self.device_type = self.device_manager.optimal_device[0] if device == "auto" else device
        self.model_loader = ModelLoader(model_name, device)
        self.model = None
        self.activations = {}
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"ActivationCapturer initialized with device: {self.device} (type: {self.device_type})")
        logger.info(f"Cache directory: {cache_dir}")
        
    def load_model(self, local_path: Optional[str] = None) -> None:
        """Load the model for activation capture."""
        self.model = self.model_loader.load_model(local_path)
        
    def _generate_cache_key(self, strings: List[str], layer_nums: List[int], cognitive_pattern: str, position: str) -> str:
        """Generate a unique cache key based on inputs."""
        # Create a hash of the relevant parameters
        content = f"{self.model_name}_{layer_nums}_{cognitive_pattern}_{position}_{len(strings)}"
        # Add a hash of the actual strings for uniqueness
        strings_hash = hashlib.md5(str(sorted(strings)).encode()).hexdigest()[:8]
        content += f"_{strings_hash}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cache_filepath(self, cache_key: str) -> str:
        """Get the full filepath for a cache key."""
        return os.path.join(self.cache_dir, f"activations_{cache_key}.pt")
    
    def _load_cached_activations(self, cache_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached activations if they exist."""
        cache_path = self._get_cache_filepath(cache_key)
        if os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached activations from {cache_path}")
                cached_data = torch.load(cache_path, map_location=self.device)
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached activations: {e}")
                return None
        return None
    
    def _save_cached_activations(self, cache_key: str, activations: Dict[str, torch.Tensor]) -> None:
        """Save activations to cache."""
        cache_path = self._get_cache_filepath(cache_key)
        try:
            torch.save(activations, cache_path)
            logger.info(f"Saved activations to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save activations to cache: {e}")
        
    def capture_activations(
        self, 
        strings: List[str], 
        layer_nums: List[int] = [17, 21],
        cognitive_pattern: str = "default",
        position: str = "last",
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Capture activations for given strings at specified layers.
        
        Args:
            strings: List of input strings
            layer_nums: Layers to capture activations from
            cognitive_pattern: Pattern category for organization
            position: Token position to extract ('last', 'all', 'last_10', or int)
            use_cache: Whether to use cached activations if available
            
        Returns:
            Dictionary of activations organized by layer and pattern
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Generate cache key and check for cached activations
        cache_key = self._generate_cache_key(strings, layer_nums, cognitive_pattern, position)
        
        if use_cache:
            cached_activations = self._load_cached_activations(cache_key)
            if cached_activations is not None:
                logger.info(f"Using cached activations for {cognitive_pattern} (cache key: {cache_key})")
                # Store in instance
                self._update_stored_activations(cached_activations, cognitive_pattern)
                return cached_activations
        
        logger.info(f"Computing activations for {cognitive_pattern} (cache key: {cache_key})")
        all_activations = {}
        
        for string in tqdm(strings, desc=f"Processing {cognitive_pattern}"):
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
                elif position == "last_10":
                    # Take last 10 tokens, or all tokens if sequence is shorter
                    seq_len = activation.shape[0]
                    start_pos = max(0, seq_len - 10)
                    pos_activation = activation[start_pos:, :].detach()
                elif isinstance(position, int):
                    pos_activation = activation[position, :].detach()
                else:
                    pos_activation = activation[-1, :].detach()  # default to last
                
                all_activations[pattern_key].append(pos_activation)
        
        # Convert lists to tensors
        for key in all_activations:
            if position == "all" or position == "last_10":
                # For variable length sequences, we need to handle padding
                max_seq_len = max(tensor.size(0) for tensor in all_activations[key])
                
                # Pad all sequences to same length
                padded_tensors = []
                for tensor in all_activations[key]:
                    if tensor.size(0) < max_seq_len:
                        # Pad with zeros at the beginning for last_10 to maintain "last" semantics
                        pad_size = max_seq_len - tensor.size(0)
                        if position == "last_10":
                            # Pad at the beginning to preserve the "last" tokens
                            padded = torch.cat([torch.zeros(pad_size, tensor.size(1), device=tensor.device), tensor], dim=0)
                        else:
                            # Pad at the end for "all"
                            padded = torch.cat([tensor, torch.zeros(pad_size, tensor.size(1), device=tensor.device)], dim=0)
                        padded_tensors.append(padded)
                    else:
                        padded_tensors.append(tensor)
                
                all_activations[key] = torch.stack(padded_tensors)
            else:
                # For single position extractions, shapes should already match
                all_activations[key] = torch.stack(all_activations[key])
        
        # Save to cache if requested
        if use_cache:
            self._save_cached_activations(cache_key, all_activations)
        
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
    
    def list_cached_activations(self) -> List[str]:
        """List all cached activation files."""
        if not os.path.exists(self.cache_dir):
            return []
        
        cache_files = []
        for filename in os.listdir(self.cache_dir):
            if filename.startswith("activations_") and filename.endswith(".pt"):
                cache_files.append(filename)
        return cache_files
    
    def clear_cache(self, pattern: Optional[str] = None) -> None:
        """Clear cached activation files."""
        if pattern:
            # Clear specific pattern caches (approximate matching)
            cache_files = self.list_cached_activations()
            for cache_file in cache_files:
                if pattern in cache_file:
                    cache_path = os.path.join(self.cache_dir, cache_file)
                    try:
                        os.remove(cache_path)
                        logger.info(f"Removed cache file: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        else:
            # Clear all cache files
            cache_files = self.list_cached_activations()
            for cache_file in cache_files:
                cache_path = os.path.join(self.cache_dir, cache_file)
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached activations."""
        cache_files = self.list_cached_activations()
        total_size = 0
        
        for cache_file in cache_files:
            cache_path = os.path.join(self.cache_dir, cache_file)
            try:
                total_size += os.path.getsize(cache_path)
            except Exception:
                pass
        
        return {
            "cache_directory": self.cache_dir,
            "cached_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_files": cache_files
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get loaded model information."""
        return self.model_loader.get_model_info()