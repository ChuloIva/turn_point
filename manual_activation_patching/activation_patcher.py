import torch
import json
from functools import partial
import sys
import os
from enum import Enum

sys.path.append('/Users/ivanculo/Desktop/Projects/turn_point/third_party/TransformerLens')

from transformer_lens import HookedTransformer
import transformer_lens.utils as utils


class TokenSelectionStrategy(Enum):
    """Token selection strategies for activation extraction."""
    KEYWORDS = "keywords"  # Original: extract from specific meaningful words
    MID_TOKENS = "mid_tokens"  # Extract from middle portion of the sequence
    LAST_COUPLE = "last_couple"  # Extract and average from last few tokens
    LAST_TOKEN = "last_token"  # Extract only from the very last token
    ALL_TOKENS = "all_tokens"  # Extract and average from all tokens


class ActivationPatcher:
    # Supported models with their configurations
    SUPPORTED_MODELS = {
        # GPT-2 family
        "gpt2-small": {"family": "gpt2", "size": "small"},
        "gpt2-medium": {"family": "gpt2", "size": "medium"}, 
        "gpt2-large": {"family": "gpt2", "size": "large"},
        "gpt2-xl": {"family": "gpt2", "size": "xl"},
        
        # GPT-J
        "EleutherAI/gpt-j-6b": {"family": "gptj", "size": "6b"},
        
        # GPT-Neo family
        "EleutherAI/gpt-neo-125m": {"family": "gpt-neo", "size": "125m"},
        "EleutherAI/gpt-neo-1.3b": {"family": "gpt-neo", "size": "1.3b"},
        "EleutherAI/gpt-neo-2.7b": {"family": "gpt-neo", "size": "2.7b"},
        
        # OPT family
        "facebook/opt-125m": {"family": "opt", "size": "125m"},
        "facebook/opt-1.3b": {"family": "opt", "size": "1.3b"},
        "facebook/opt-2.7b": {"family": "opt", "size": "2.7b"},
        "facebook/opt-6.7b": {"family": "opt", "size": "6.7b"},
        
        # Pythia family
        "EleutherAI/pythia-70m": {"family": "pythia", "size": "70m"},
        "EleutherAI/pythia-160m": {"family": "pythia", "size": "160m"},
        "EleutherAI/pythia-410m": {"family": "pythia", "size": "410m"},
        "EleutherAI/pythia-1b": {"family": "pythia", "size": "1b"},
        "EleutherAI/pythia-1.4b": {"family": "pythia", "size": "1.4b"},
        "EleutherAI/pythia-2.8b": {"family": "pythia", "size": "2.8b"},
        
        # Gemma (if available)
        "google/gemma-2b": {"family": "gemma", "size": "2b"},
        "google/gemma-7b": {"family": "gemma", "size": "7b"},
    }
    
    def __init__(self, model_name="gpt2-small", device="auto"):
        """Initialize the activation patcher with a specified model."""
        if model_name not in self.SUPPORTED_MODELS:
            print(f"Warning: {model_name} not in supported models list. Attempting to load anyway.")
            print(f"Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        # Handle device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA GPU")
            else:
                device = "cpu"
                print("Using CPU")
        
        try:
            print(f"Loading model: {model_name} on device: {device}")
            self.model = HookedTransformer.from_pretrained(model_name, device=device)
            self.model.eval()
            self.model_name = model_name
            self.model_config = self.SUPPORTED_MODELS.get(model_name, {"family": "unknown", "size": "unknown"})
            
            print(f"✓ Model loaded successfully")
            print(f"  - Family: {self.model_config['family']}")
            print(f"  - Size: {self.model_config['size']}")
            print(f"  - Layers: {self.model.cfg.n_layers}")
            print(f"  - Model dimension: {self.model.cfg.d_model}")
            print(f"  - Vocabulary size: {self.model.cfg.d_vocab}")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to gpt2-small")
            self.model = HookedTransformer.from_pretrained("gpt2-small", device=device)
            self.model.eval()
            self.model_name = "gpt2-small"
            self.model_config = self.SUPPORTED_MODELS["gpt2-small"]
        
    def load_dataset(self, jsonl_path):
        """Load the positive patterns dataset."""
        patterns = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                patterns.append(json.loads(line.strip()))
        return patterns
    
    def get_bos_token(self):
        """Get the BOS token for the current model."""
        if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'bos_token'):
            bos_token = self.model.tokenizer.bos_token
            if bos_token:
                return bos_token
        
        # Fallback based on model family
        model_family = self.model_config.get('family', '').lower()
        if any(family in model_family for family in ['gpt2', 'gpt-neo', 'gpt-j']):
            return "<|endoftext|>"
        elif 'opt' in model_family:
            return "</s>"
        else:
            # Default fallback to <bos>
            return "<bos>"
    
    def reset_hooks(self):
        """Reset all hooks on the model to clean state.
        
        This method should be called between experiments to ensure no residual
        hooks interfere with subsequent runs.
        """
        self.model.reset_hooks()
        print("✅ Model hooks reset to clean state")
    
    def _process_layer_specification(self, layer_spec):
        """Process layer specification and return list of layer indices.
        
        Args:
            layer_spec: Can be int, list of ints, 'all', or range object
            
        Returns:
            List of layer indices
        """
        if isinstance(layer_spec, int):
            # Single layer
            return [layer_spec if layer_spec >= 0 else self.model.cfg.n_layers + layer_spec]
        elif isinstance(layer_spec, (list, tuple)):
            # List of layers
            return [idx if idx >= 0 else self.model.cfg.n_layers + idx for idx in layer_spec]
        elif layer_spec == 'all':
            # All layers
            return list(range(self.model.cfg.n_layers))
        elif hasattr(layer_spec, '__iter__'):
            # Range or other iterable
            return [idx if idx >= 0 else self.model.cfg.n_layers + idx for idx in layer_spec]
        else:
            raise ValueError(f"Invalid layer specification: {layer_spec}")
    
    def prepare_texts(self, clean_text, corrupted_text, num_placeholder_tokens=5, bos_token="<bos>"):
        """Prepare clean and corrupted texts for patching."""
        placeholder = "<|endoftext|>"
        placeholder_sequence = " ".join([placeholder] * num_placeholder_tokens)
        
        if bos_token:
            full_corrupted_text = f"{bos_token}{placeholder_sequence} {corrupted_text}"
        else:
            full_corrupted_text = f"{placeholder_sequence} {corrupted_text}"
        
        clean_tokens = self.model.to_tokens(clean_text)
        corrupted_tokens = self.model.to_tokens(full_corrupted_text)
        
        return clean_tokens, corrupted_tokens, num_placeholder_tokens
    
    def get_clean_cache(self, clean_tokens):
        """Run model on clean tokens and return cache."""
        _, clean_cache = self.model.run_with_cache(clean_tokens)
        return clean_cache
    
    def create_patching_hook(self, clean_activation_vectors, patch_positions, activation_name):
        """Create a hook function for patching activations."""
        def patching_hook(corrupted_activation, hook, clean_vectors=clean_activation_vectors, positions=patch_positions):
            print(f"Patching at hook: {hook.name} for positions: {positions}")
            
            for i, pos in enumerate(positions):
                if i < len(clean_vectors) and pos < corrupted_activation.shape[1]:
                    corrupted_activation[0, pos, :] = clean_vectors[i]
            
            return corrupted_activation
        
        return patching_hook
    
    def extract_activations_by_strategy(self, clean_cache, clean_tokens, layer_idx=-1, 
                                        strategy=TokenSelectionStrategy.KEYWORDS, 
                                        target_words=None, num_tokens=3):
        """Extract activations using different token selection strategies.
        
        Args:
            clean_cache: Model cache from clean text
            clean_tokens: Tokenized clean text
            layer_idx: Layer index to extract from
            strategy: TokenSelectionStrategy enum value
            target_words: List of words for KEYWORDS strategy
            num_tokens: Number of tokens for LAST_COUPLE and MID_TOKENS strategies
        
        Returns:
            activation_vectors: List of activation tensors
            positions_found: List of token positions used
            activation_name: Name of the activation layer
        """
        activation_name = utils.get_act_name("resid_post", 
                                           layer_idx if layer_idx >= 0 else self.model.cfg.n_layers + layer_idx)
        
        activation_vectors = []
        positions_found = []
        seq_len = clean_tokens.shape[1]
        
        print(f"Using token selection strategy: {strategy.value}")
        print(f"Sequence length: {seq_len}")
        
        if strategy == TokenSelectionStrategy.KEYWORDS:
            # Original behavior: extract from specific meaningful words
            if target_words is None:
                target_words = self._extract_key_words(self.model.to_string(clean_tokens[0]))
            
            for word in target_words:
                try:
                    word_token = self.model.to_single_token(f" {word}")
                    positions = torch.where(clean_tokens[0] == word_token)[0]
                    
                    if len(positions) > 0:
                        pos = positions[0]
                        activation_vector = clean_cache[activation_name][0, pos, :]
                        activation_vectors.append(activation_vector)
                        positions_found.append(pos.item())
                        print(f"Found keyword '{word}' at position {pos.item()}")
                    else:
                        print(f"Keyword '{word}' not found in clean tokens")
                except ValueError:
                    print(f"Could not tokenize word '{word}'")
        
        elif strategy == TokenSelectionStrategy.MID_TOKENS:
            # Extract from middle portion of sequence
            start_idx = max(1, seq_len // 3)  # Skip BOS token, start at 1/3
            end_idx = min(seq_len - 1, (2 * seq_len) // 3)  # End at 2/3
            
            # Take num_tokens from the middle range
            mid_positions = list(range(start_idx, end_idx))
            if len(mid_positions) > num_tokens:
                # Sample evenly spaced positions
                step = len(mid_positions) // num_tokens
                mid_positions = mid_positions[::step][:num_tokens]
            
            for pos in mid_positions:
                activation_vector = clean_cache[activation_name][0, pos, :]
                activation_vectors.append(activation_vector)
                positions_found.append(pos)
            
            print(f"Selected {len(mid_positions)} mid tokens from positions: {mid_positions}")
        
        elif strategy == TokenSelectionStrategy.LAST_COUPLE:
            # Extract and average from last few tokens
            start_pos = max(1, seq_len - num_tokens)  # Don't include position 0 (BOS)
            last_positions = list(range(start_pos, seq_len))
            
            # Extract all last tokens
            last_activations = []
            for pos in last_positions:
                activation_vector = clean_cache[activation_name][0, pos, :]
                last_activations.append(activation_vector)
                positions_found.append(pos)
            
            # Average them into a single vector
            if last_activations:
                averaged_activation = torch.mean(torch.stack(last_activations), dim=0)
                activation_vectors.append(averaged_activation)
            
            print(f"Averaged last {len(last_positions)} tokens from positions: {last_positions}")
        
        elif strategy == TokenSelectionStrategy.LAST_TOKEN:
            # Extract only from the very last token
            last_pos = seq_len - 1
            activation_vector = clean_cache[activation_name][0, last_pos, :]
            activation_vectors.append(activation_vector)
            positions_found.append(last_pos)
            
            print(f"Selected last token at position: {last_pos}")
        
        elif strategy == TokenSelectionStrategy.ALL_TOKENS:
            # Extract and average from all tokens (excluding BOS if present)
            start_pos = 1 if seq_len > 1 else 0  # Skip BOS token
            all_positions = list(range(start_pos, seq_len))
            
            # Extract all token activations
            all_activations = []
            for pos in all_positions:
                activation_vector = clean_cache[activation_name][0, pos, :]
                all_activations.append(activation_vector)
                positions_found.append(pos)
            
            # Average them into a single vector
            if all_activations:
                averaged_activation = torch.mean(torch.stack(all_activations), dim=0)
                activation_vectors.append(averaged_activation)
            
            print(f"Averaged all {len(all_positions)} tokens from positions: {all_positions}")
        
        else:
            raise ValueError(f"Unknown token selection strategy: {strategy}")
        
        return activation_vectors, positions_found, activation_name

    def extract_activations_for_patching(self, clean_cache, clean_tokens, target_words, layer_idx=-1):
        """Extract activations from specific words/tokens for patching.
        
        This method maintains backward compatibility with existing code.
        For new code, consider using extract_activations_by_strategy() instead.
        """
        return self.extract_activations_by_strategy(
            clean_cache, clean_tokens, layer_idx, 
            TokenSelectionStrategy.KEYWORDS, target_words
        )
    
    def extract_batch_activations(self, texts, layer_idx=-1, aggregation="mean", target_words=None):
        """Extract activations from a batch of texts and aggregate them."""
        activation_name = utils.get_act_name("resid_post", 
                                           layer_idx if layer_idx >= 0 else self.model.cfg.n_layers + layer_idx)
        
        all_activations = []
        all_positions = []
        
        print(f"Processing {len(texts)} texts for batch activation extraction...")
        
        for i, text in enumerate(texts):
            try:
                tokens = self.model.to_tokens(text)
                _, cache = self.model.run_with_cache(tokens)
                
                # If specific words are provided, extract from those positions
                if target_words:
                    for word in target_words:
                        try:
                            word_token = self.model.to_single_token(f" {word}")
                            positions = torch.where(tokens[0] == word_token)[0]
                            
                            for pos in positions:
                                activation = cache[activation_name][0, pos, :]
                                all_activations.append(activation)
                                all_positions.append((i, pos.item(), word))
                        except ValueError:
                            continue
                else:
                    # Extract from all meaningful positions (skip special tokens)
                    seq_len = tokens.shape[1]
                    for pos in range(1, seq_len):  # Skip first token (usually BOS)
                        activation = cache[activation_name][0, pos, :]
                        all_activations.append(activation)
                        all_positions.append((i, pos, "position"))
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                continue
        
        if not all_activations:
            print("No activations extracted!")
            return None, None, activation_name
        
        # Stack activations
        activation_tensor = torch.stack(all_activations)
        print(f"Extracted {activation_tensor.shape[0]} activations from {len(texts)} texts")
        
        # Aggregate activations
        if aggregation == "mean":
            aggregated_activation = torch.mean(activation_tensor, dim=0)
        elif aggregation == "median":
            aggregated_activation = torch.median(activation_tensor, dim=0)[0]
        elif aggregation == "max":
            aggregated_activation = torch.max(activation_tensor, dim=0)[0]
        elif aggregation == "random":
            idx = torch.randint(0, activation_tensor.shape[0], (1,))
            aggregated_activation = activation_tensor[idx[0]]
        else:
            print(f"Unknown aggregation method: {aggregation}. Using mean.")
            aggregated_activation = torch.mean(activation_tensor, dim=0)
        
        print(f"Aggregated {activation_tensor.shape[0]} activations using {aggregation}")
        print(f"Final activation shape: {aggregated_activation.shape}")
        
        return aggregated_activation, all_positions, activation_name
    
    def batch_patch_and_generate(self, clean_texts, corrupted_text, capture_layer_idx=-1, 
                                patch_layer_idx=None, aggregation="mean", target_words=None, 
                                num_placeholder_tokens=5, max_new_tokens=50, bos_token="<bos>"):
        """Patch activations from multiple texts and generate with multi-layer support."""
        
        # Default patch_layer_idx to capture_layer_idx if not specified
        if patch_layer_idx is None:
            patch_layer_idx = capture_layer_idx
            
        # Reset hooks to ensure clean state
        self.reset_hooks()
            
        print(f"Batch patching with {len(clean_texts)} clean texts")
        print(f"Aggregation method: {aggregation}")
        print(f"Target words: {target_words}")
        print(f"Capture layer: {capture_layer_idx}")
        print(f"Patch layer: {patch_layer_idx}")
        
        # Extract and aggregate activations from batch
        aggregated_activation, positions_info, capture_activation_name = self.extract_batch_activations(
            clean_texts, capture_layer_idx, aggregation, target_words
        )
        
        if aggregated_activation is None:
            print("Failed to extract activations from batch")
            return None, None
        
        # Prepare corrupted tokens
        placeholder = "<|endoftext|>"
        placeholder_sequence = " ".join([placeholder] * num_placeholder_tokens)
        full_corrupted_text = f"{bos_token}{placeholder_sequence} {corrupted_text}"
        corrupted_tokens = self.model.to_tokens(full_corrupted_text)
        
        print(f"Corrupted text: {full_corrupted_text}")
        
        # Create patch activation name for the layer we want to patch into
        # For now, support single layer patching in batch mode (can be extended later)
        if isinstance(patch_layer_idx, (list, tuple)) and len(patch_layer_idx) > 1:
            print("Warning: Batch patching currently supports single patch layer. Using first layer from list.")
            patch_layer_idx = patch_layer_idx[0]
        elif patch_layer_idx == 'all':
            patch_layer_idx = -1  # Default to last layer for 'all'
        elif isinstance(patch_layer_idx, (list, tuple)):
            patch_layer_idx = patch_layer_idx[0]
            
        patch_activation_name = utils.get_act_name("resid_post", 
                                                  patch_layer_idx if patch_layer_idx >= 0 else self.model.cfg.n_layers + patch_layer_idx)
        
        # Create hook function using the aggregated activation
        def batch_patching_hook(corrupted_activation, hook, agg_activation=aggregated_activation):
            print(f"Batch patching at hook: {hook.name}")
            # Patch all placeholder positions with the same aggregated activation
            for pos in range(min(num_placeholder_tokens, corrupted_activation.shape[1])):
                corrupted_activation[0, pos, :] = agg_activation
            return corrupted_activation
        
        # Generate text with patched activations
        try:
            patched_logits = self.model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(patch_activation_name, batch_patching_hook)]
            )
            
            last_token_logits = patched_logits[0, -1, :]
            predicted_token_id = torch.argmax(last_token_logits).item()
            predicted_token_str = self.model.to_string([predicted_token_id])
            
            print(f"Next token prediction: '{predicted_token_str}'")
            
            generated_tokens = self.model.generate(
                corrupted_tokens,
                max_new_tokens=max_new_tokens,
                fwd_hooks=[(patch_activation_name, batch_patching_hook)],
                temperature=0.7,
                do_sample=True
            )
            generated_text = self.model.to_string(generated_tokens[0])
            
            return predicted_token_str, generated_text
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return None, None
    
    def patch_and_generate(self, clean_text, corrupted_text, target_words=None, 
                          num_placeholder_tokens=5, capture_layer_idx=-1, patch_layer_idx=None, 
                          max_new_tokens=50, bos_token="<bos>", 
                          token_selection_strategy=TokenSelectionStrategy.KEYWORDS,
                          num_strategy_tokens=3):
        """Main method to perform activation patching and generate text.
        
        Args:
            clean_text: Clean text to extract activations from
            corrupted_text: Corrupted text to patch activations into
            target_words: List of words for KEYWORDS strategy (auto-extracted if None)
            num_placeholder_tokens: Number of placeholder tokens in corrupted text
            capture_layer_idx: Layer(s) to capture clean activations from. 
                              Can be: int, list of ints, 'all', or range(start, end)
            patch_layer_idx: Layer(s) to patch activations into. 
                            Can be: int, list of ints, 'all', or range(start, end)
                            Defaults to capture_layer_idx if None
            max_new_tokens: Maximum number of tokens to generate
            bos_token: Beginning of sequence token
            token_selection_strategy: TokenSelectionStrategy enum for how to select tokens
            num_strategy_tokens: Number of tokens for LAST_COUPLE and MID_TOKENS strategies
        """
        # Reset hooks to ensure clean state
        self.reset_hooks()
        
        # Process layer specifications
        capture_layers = self._process_layer_specification(capture_layer_idx)
        
        # Default patch_layer_idx to capture_layer_idx if not specified
        if patch_layer_idx is None:
            patch_layer_idx = capture_layer_idx
        patch_layers = self._process_layer_specification(patch_layer_idx)
        
        clean_tokens, corrupted_tokens, num_placeholders = self.prepare_texts(
            clean_text, corrupted_text, num_placeholder_tokens, bos_token
        )
        
        print(f"Clean text: {clean_text}")
        print(f"Corrupted text: {self.model.to_string(corrupted_tokens[0])}")
        
        clean_cache = self.get_clean_cache(clean_tokens)
        
        # Collect activations from all specified capture layers
        all_activation_vectors = []
        all_patch_hooks = []
        
        print(f"Capture layers: {capture_layers}")
        print(f"Patch layers: {patch_layers}")
        
        # For each capture-patch layer pair
        for capture_layer in capture_layers:
            activation_vectors, positions_found, _ = self.extract_activations_by_strategy(
                clean_cache, clean_tokens, capture_layer, 
                token_selection_strategy, target_words, num_strategy_tokens
            )
            
            if not activation_vectors:
                print(f"No activations found for capture layer {capture_layer}")
                continue
                
            all_activation_vectors.extend(activation_vectors)
            
            # Create hooks for each patch layer using activations from this capture layer
            for patch_layer in patch_layers:
                patch_activation_name = utils.get_act_name("resid_post", patch_layer)
                
                patch_positions = list(range(num_placeholders))
                num_patches = min(len(activation_vectors), len(patch_positions))
                patch_positions = patch_positions[:num_patches]
                current_vectors = activation_vectors[:num_patches]
                
                hook_fn = self.create_patching_hook(current_vectors, patch_positions, patch_activation_name)
                all_patch_hooks.append((patch_activation_name, hook_fn))
        
        if not all_activation_vectors:
            print("No activations found for patching from any layer!")
            return None, None
        
        print(f"\nPatching with {len(all_activation_vectors)} total activation vectors")
        print(f"Using {len(all_patch_hooks)} hooks across layers: {capture_layers} -> {patch_layers}")
        
        patched_logits = self.model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=all_patch_hooks
        )
        
        last_token_logits = patched_logits[0, -1, :]
        predicted_token_id = torch.argmax(last_token_logits).item()
        predicted_token_str = self.model.to_string([predicted_token_id])
        
        print(f"\nNext token prediction after patching: '{predicted_token_str}'")
        
        generated_tokens = self.model.generate(
            corrupted_tokens,
            max_new_tokens=max_new_tokens,
            fwd_hooks=all_patch_hooks,
            temperature=0.7,
            do_sample=True
        )
        generated_text = self.model.to_string(generated_tokens[0])
        
        return predicted_token_str, generated_text
    
    def experiment_with_dataset(self, dataset_path, num_samples=3, max_new_tokens=60):
        """Run experiments using samples from the dataset."""
        patterns = self.load_dataset(dataset_path)
        
        print(f"Loaded {len(patterns)} patterns from dataset")
        print("="*80)
        
        for i in range(min(num_samples, len(patterns))):
            pattern = patterns[i]
            
            print(f"\n--- Experiment {i+1}/{num_samples} ---")
            print(f"Pattern: {pattern['cognitive_pattern_name']}")
            
            clean_text = pattern['positive_thought_pattern']
            
            corrupted_prompts = [
                "I feel overwhelmed and stuck. Everything seems",
                "My thoughts are spiraling and I can't focus on",
                "I'm struggling with negative thoughts about"
            ]
            
            corrupted_text = corrupted_prompts[i % len(corrupted_prompts)]
            
            target_words = self._extract_key_words(clean_text)
            
            print(f"Clean text (truncated): {clean_text[:100]}...")
            print(f"Corrupted prompt: {corrupted_text}")
            print(f"Target words for patching: {target_words}")
            
            try:
                predicted_token, generated_text = self.patch_and_generate(
                    clean_text, corrupted_text, target_words, 
                    num_placeholder_tokens=5, layer_idx=-1, max_new_tokens=max_new_tokens
                )
                
                if generated_text:
                    print(f"\n--- Generated Text ---")
                    print(generated_text)
                    print("="*80)
                else:
                    print("Failed to generate text for this sample")
                    
            except Exception as e:
                print(f"Error in experiment {i+1}: {e}")
                continue
    
    def _extract_key_words(self, text):
        """Extract key meaningful words from text for patching."""
        words = text.split()
        
        key_words = []
        skip_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        
        for word in words[:20]:  # Look at first 20 words
            clean_word = word.strip('.,!?";()').lower()
            if len(clean_word) > 3 and clean_word not in skip_words:
                key_words.append(clean_word)
                if len(key_words) >= 5:  # Limit to 5 key words
                    break
        
        return key_words[:5]
    
    @classmethod
    def list_supported_models(cls):
        """List all supported models grouped by family."""
        print("Supported Models:")
        print("="*50)
        
        families = {}
        for model_name, config in cls.SUPPORTED_MODELS.items():
            family = config['family']
            if family not in families:
                families[family] = []
            families[family].append((model_name, config['size']))
        
        for family, models in families.items():
            print(f"\n{family.upper()}:")
            for model_name, size in models:
                print(f"  - {model_name} ({size})")
    
    def get_model_info(self):
        """Get detailed information about the current model."""
        return {
            'name': self.model_name,
            'family': self.model_config['family'],
            'size': self.model_config['size'],
            'n_layers': self.model.cfg.n_layers,
            'd_model': self.model.cfg.d_model,
            'd_vocab': self.model.cfg.d_vocab,
            'n_heads': getattr(self.model.cfg, 'n_heads', 'N/A'),
            'd_head': getattr(self.model.cfg, 'd_head', 'N/A'),
            'device': str(next(self.model.parameters()).device)
        }


if __name__ == "__main__":
    patcher = ActivationPatcher("gpt2-small")
    
    dataset_path = "/Users/ivanculo/Desktop/Projects/turn_point/data/final/positive_patterns.jsonl"
    
    patcher.experiment_with_dataset(dataset_path, num_samples=3, max_new_tokens=60)