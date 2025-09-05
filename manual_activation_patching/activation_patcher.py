import torch
import json
from functools import partial
import sys
import os
from enum import Enum
import re
import importlib.util
import random

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
        placeholder = "<eos>"
        placeholder_sequence = " ".join([placeholder] * num_placeholder_tokens)
        
        if bos_token:
            full_corrupted_text = f"{bos_token}{placeholder_sequence} {corrupted_text}"
        else:
            full_corrupted_text = f"{placeholder_sequence} {corrupted_text}"
        
        clean_tokens = self.model.to_tokens(clean_text)
        corrupted_tokens = self.model.to_tokens(full_corrupted_text)
        
        return clean_tokens, corrupted_tokens, num_placeholder_tokens

    # -------------------------
    # Zero-placeholder utilities
    # -------------------------
    def _load_interpretation_templates(self):
        """Load INTERPRETATION_TEMPLATES from interpretation_templates.py regardless of package state."""
        templates_path = "/Users/ivanculo/Desktop/Projects/turn_point/manual_activation_patching/interpretation_templates.py"
        try:
            spec = importlib.util.spec_from_file_location("interpretation_templates", templates_path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader is not None
            spec.loader.exec_module(module)
            return getattr(module, "INTERPRETATION_TEMPLATES", {})
        except Exception as e:
            print(f"Could not load interpretation templates: {e}")
            return {}

    def get_interpretation_template(self, template_name):
        """Public helper to fetch a template tuple by name."""
        templates = self._load_interpretation_templates()
        if template_name not in templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(templates.keys())}")
        return templates[template_name]

    def _render_zero_prompt_from_template(self, template_obj, bos_token, placeholder_token_string):
        """Render a prompt string from a template tuple where integer 0 marks patch points.

        The expected template format is ("<bos>", 0, 0, ..., "continuation text").
        We ignore the first element and use the provided bos_token.
        """
        if not isinstance(template_obj, (list, tuple)):
            raise ValueError("Template object must be a list/tuple for zero-placeholder rendering.")

        parts = []
        # Start with BOS (no trailing space to keep parity with original prepare_texts)
        if bos_token:
            parts.append(bos_token)

        # Render rest: replace int 0 with placeholder string
        for elem in template_obj[1:]:
            if isinstance(elem, int) and elem == 0:
                parts.append(f" {placeholder_token_string}")
            elif isinstance(elem, str):
                # Ensure a space before normal text if not already present
                if len(parts) == 0:
                    parts.append(elem)
                else:
                    parts.append(f" {elem}")
            else:
                # Unknown type, convert to string conservatively
                parts.append(f" {str(elem)}")

        return "".join(parts)

    def _render_zero_prompt_from_string(self, text_with_zeros, bos_token, placeholder_token_string):
        """Render a prompt string from a manual string where '0' marks patch points.

        We replace standalone '0' with the placeholder token string and prefix BOS.
        """
        if text_with_zeros is None:
            text_with_zeros = ""

        # Remove any literal "<bos>"-like markers at the start; we'll inject our BOS correctly
        cleaned = text_with_zeros.strip()
        cleaned = re.sub(r"^(<\|endoftext\|>|</s>|<bos>|<eos>)", "", cleaned).lstrip()

        # Replace standalone 0s (not part of larger numbers) with placeholder marker
        # Surround with spaces to encourage token separation
        replaced = re.sub(r"(?<!\d)0(?!\d)", f" {placeholder_token_string} ", cleaned)

        # Collapse multiple spaces
        replaced = re.sub(r"\s+", " ", replaced).strip()

        if bos_token:
            return f"{bos_token} {replaced}" if len(replaced) > 0 else f"{bos_token}"
        return replaced

    def _find_placeholder_token_positions(self, tokens_tensor, placeholder_token_string):
        """Find starting token indices where the placeholder string occurs as a subsequence.

        Works even if the placeholder string tokenizes into multiple tokens.
        """
        # Tokenize placeholder the same way we rendered it (with leading space, but skip BOS token)
        # The rendered placeholder appears as " <eos>" which tokenizes to [BOS, space_token, eos_token]
        # We want just [space_token, eos_token]
        full_tokenized = self.model.to_tokens(f" {placeholder_token_string}")[0]
        # Skip the first token if it's BOS (which to_tokens always adds)
        if len(full_tokenized) > 1 and full_tokenized[0] == self.model.tokenizer.bos_token_id:
            placeholder_tokens = full_tokenized[1:]
        else:
            placeholder_tokens = full_tokenized
        token_ids = tokens_tensor[0].tolist()
        needle = placeholder_tokens.tolist()

        positions = []
        if not needle:
            return positions

        n = len(token_ids)
        m = len(needle)
        # Sliding window subsequence match
        for i in range(0, n - m + 1):
            if token_ids[i:i + m] == needle:
                positions.append(i)
        return positions
    
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
        placeholder = "<eos>"
        placeholder = "<eos>"
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
            
            # Add hook and then generate (TransformerLens approach)
            self.model.add_hook(patch_activation_name, batch_patching_hook)
            
            try:
                generated_tokens = self.model.generate(
                    corrupted_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True
                )
            finally:
                # Always reset hooks after generation
                self.model.reset_hooks()
            generated_text = self.model.to_string(generated_tokens[0])
            
            return predicted_token_str, generated_text
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return None, None
    
    def patch_and_generate(self, clean_text, corrupted_text,
                          target_words=None,
                          num_placeholder_tokens=5,
                          capture_layer_idx=-1,
                          patch_layer_idx=None,
                          max_new_tokens=50,
                          bos_token="<bos>",
                          token_selection_strategy=TokenSelectionStrategy.KEYWORDS,
                          num_strategy_tokens=5,
                          zero_placeholder_mode=False,
                          prompt_input=None,
                          template_name=None,
                          placeholder_token_string="<eos>"):
        """Main method to perform activation patching and generate text.
        
        Args:
            clean_text: Clean text to extract activations from
            corrupted_text: Corrupted text to patch activations into. Ignored if zero_placeholder_mode and prompt_input/template_name are provided.
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
            zero_placeholder_mode: If True, detect '0' placeholders (manual) or template zeros and patch at those positions
            prompt_input: If string, treat as manual prompt with '0' markers; if tuple/list, treat as template-like object
            template_name: If provided, load template by name from interpretation_templates.py
            placeholder_token_string: Unique string used to mark placeholder positions after rendering (default '<eos>')
        """
        # Reset hooks to ensure clean state
        self.reset_hooks()
        
        # Process layer specifications
        capture_layers = self._process_layer_specification(capture_layer_idx)
        
        # Default patch_layer_idx to capture_layer_idx if not specified
        if patch_layer_idx is None:
            patch_layer_idx = capture_layer_idx
        patch_layers = self._process_layer_specification(patch_layer_idx)

        # Prepare tokens and patch positions depending on mode
        if zero_placeholder_mode:
            # Determine prompt source
            effective_prompt = None
            if template_name is not None:
                try:
                    tmpl = self.get_interpretation_template(template_name)
                    effective_prompt = tmpl
                except Exception as e:
                    print(f"Failed to load template '{template_name}': {e}")
            if prompt_input is not None:
                effective_prompt = prompt_input
            if effective_prompt is None:
                # Fallback to corrupted_text if provided
                effective_prompt = corrupted_text if corrupted_text is not None else ""

            # Render final text with placeholder markers
            if isinstance(effective_prompt, (list, tuple)):
                full_corrupted_text = self._render_zero_prompt_from_template(
                    effective_prompt, bos_token, placeholder_token_string
                )
            elif isinstance(effective_prompt, str):
                full_corrupted_text = self._render_zero_prompt_from_string(
                    effective_prompt, bos_token, placeholder_token_string
                )
            else:
                raise ValueError("Unsupported prompt_input type for zero_placeholder_mode. Use str or tuple/list.")

            clean_tokens = self.model.to_tokens(clean_text)
            corrupted_tokens = self.model.to_tokens(full_corrupted_text)

            # Find placeholder token positions
            patch_positions = self._find_placeholder_token_positions(
                corrupted_tokens, placeholder_token_string
            )

            if not patch_positions:
                print("Warning: No placeholder positions found in the rendered prompt. Nothing to patch.")
        else:
            clean_tokens, corrupted_tokens, num_placeholders = self.prepare_texts(
                clean_text, corrupted_text, num_placeholder_tokens, bos_token
            )
            patch_positions = list(range(num_placeholders))
        
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
                
                local_patch_positions = patch_positions
                num_patches = min(len(activation_vectors), len(local_patch_positions))
                local_patch_positions = local_patch_positions[:num_patches]
                current_vectors = activation_vectors[:num_patches]
                
                hook_fn = self.create_patching_hook(current_vectors, local_patch_positions, patch_activation_name)
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
        
        # Add hooks and then generate (TransformerLens approach)
        for hook_name, hook_fn in all_patch_hooks:
            self.model.add_hook(hook_name, hook_fn)
        
        try:
            generated_tokens = self.model.generate(
                corrupted_tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
        finally:
            # Always reset hooks after generation
            self.model.reset_hooks()
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

    # Data loading and pattern utilities
    @staticmethod
    def load_cognitive_patterns(dataset_path="/Users/ivanculo/Desktop/Projects/turn_point/data/final/positive_patterns.jsonl", 
                               max_examples_per_type=None):
        """Load the cognitive patterns dataset with all text variants (positive, negative, transition).
        
        Args:
            dataset_path: Path to the JSONL dataset file
            max_examples_per_type: Maximum number of examples to load per cognitive pattern type (None = load all)
        
        Returns:
            patterns: List of all patterns
            pattern_types: Dictionary grouping patterns by type
        """
        all_patterns = []
        all_pattern_types = {}
        
        # First pass: load all patterns and group by type
        with open(dataset_path, 'r') as f:
            for line in f:
                pattern = json.loads(line.strip())
                all_patterns.append(pattern)
                
                # Group by cognitive pattern type
                pattern_type = pattern['cognitive_pattern_type']
                if pattern_type not in all_pattern_types:
                    all_pattern_types[pattern_type] = []
                all_pattern_types[pattern_type].append(pattern)
        
        # If no limit specified, return all patterns
        if max_examples_per_type is None:
            return all_patterns, all_pattern_types
        
        # Second pass: limit examples per type
        limited_patterns = []
        limited_pattern_types = {}
        
        for pattern_type, type_patterns in all_pattern_types.items():
            # Take only the first max_examples_per_type examples
            limited_type_patterns = type_patterns[:max_examples_per_type]
            limited_pattern_types[pattern_type] = limited_type_patterns
            limited_patterns.extend(limited_type_patterns)
        
        print(f"📊 Limited dataset: {max_examples_per_type} examples per type")
        print(f"📈 Total patterns: {len(limited_patterns)} (was {len(all_patterns)})")
        
        return limited_patterns, limited_pattern_types

    @staticmethod
    def get_pattern_by_index(patterns, index):
        """Get a pattern by index with bounds checking."""
        if 0 <= index < len(patterns):
            return patterns[index]
        else:
            raise IndexError(f"Index {index} out of range. Dataset has {len(patterns)} patterns.")

    @staticmethod
    def get_pattern_by_type(pattern_types, pattern_type, index=0):
        """
        Get a pattern by cognitive pattern type and optionally by index within that type.
        
        Args:
            pattern_types: The pattern_types dictionary from load_cognitive_patterns()
            pattern_type: The cognitive pattern type string
            index: Index within the pattern type (default: 0 for first example)
        """
        if pattern_type in pattern_types:
            patterns_of_type = pattern_types[pattern_type]
            if 0 <= index < len(patterns_of_type):
                return patterns_of_type[index]
            else:
                raise IndexError(f"Index {index} out of range. Pattern type '{pattern_type}' has {len(patterns_of_type)} examples.")
        else:
            available_types = list(pattern_types.keys())
            raise KeyError(f"Pattern type '{pattern_type}' not found. Available types: {available_types}")

    @staticmethod
    def get_random_pattern_by_type(pattern_types, pattern_type):
        """Get a random pattern from a specific cognitive pattern type."""
        patterns_of_type = ActivationPatcher.get_patterns_by_type(pattern_types, pattern_type)
        return random.choice(patterns_of_type)

    @staticmethod
    def get_patterns_by_type(pattern_types, pattern_type):
        """Get all patterns for a specific cognitive pattern type."""
        if pattern_type in pattern_types:
            return pattern_types[pattern_type]
        else:
            available_types = list(pattern_types.keys())
            raise KeyError(f"Pattern type '{pattern_type}' not found. Available types: {available_types}")

    @staticmethod
    def list_available_pattern_types(pattern_types):
        """List all available pattern types with counts."""
        print("Available cognitive pattern types:")
        for i, (pattern_type, examples) in enumerate(pattern_types.items(), 1):
            print(f"{i:2d}. {pattern_type} ({len(examples)} examples)")

    @staticmethod
    def get_pattern_text(pattern, text_type="positive"):
        """
        Get specific text variant from a pattern.
        
        Args:
            pattern: The pattern dictionary
            text_type: "positive", "negative", or "transition"
        
        Returns:
            The requested text string
        """
        text_map = {
            "positive": "positive_thought_pattern",
            "negative": "reference_negative_example", 
            "transition": "reference_transformed_example"
        }
        
        if text_type not in text_map:
            raise ValueError(f"text_type must be one of: {list(text_map.keys())}")
        
        field_name = text_map[text_type]
        if field_name not in pattern:
            raise KeyError(f"Pattern missing field: {field_name}")
        
        return pattern[field_name]

    @staticmethod
    def get_template(template_name):
        """Get an interpretation template by name."""
        # Load templates dynamically
        try:
            templates_path = "/Users/ivanculo/Desktop/Projects/turn_point/manual_activation_patching/interpretation_templates.py"
            spec = importlib.util.spec_from_file_location("interpretation_templates", templates_path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader is not None
            spec.loader.exec_module(module)
            templates = getattr(module, "INTERPRETATION_TEMPLATES", {})
            
            if template_name in templates:
                return templates[template_name]
            else:
                available_templates = list(templates.keys())
                raise KeyError(f"Template '{template_name}' not found. Available templates: {available_templates}")
        except Exception as e:
            print(f"Could not load interpretation templates: {e}")
            raise

    @staticmethod
    def show_pattern_info(pattern):
        """Display detailed information about a pattern."""
        print(f"🧠 Pattern: {pattern['cognitive_pattern_name']}")
        print(f"🔄 Type: {pattern['cognitive_pattern_type']}")
        print(f"📝 Description: {pattern['pattern_description']}")
        print(f"❓ Source Question: {pattern['source_question']}")
        print(f"\n✅ Positive Text: {pattern['positive_thought_pattern']}")
        print(f"\n❌ Negative Text: {pattern['reference_negative_example']}")
        print(f"\n🔄 Transition Text: {pattern['reference_transformed_example']}")

    def check_model_info(self):
        """Check and display model status information."""
        print("📊 MODEL STATUS:")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {next(self.model.parameters()).device}")
        print(f"  Layers: {self.model.cfg.n_layers}")
        print(f"  D_model: {self.model.cfg.d_model}")
        print(f"  Vocab size: {self.model.cfg.d_vocab}")
    
    @staticmethod
    def filter_patterns_by_count(pattern_types, num_examples_per_type):
        """Filter patterns at experiment time to use only specified number of examples per type.
        
        Args:
            pattern_types: The full pattern_types dictionary
            num_examples_per_type: Number of examples to use per type (1-40)
            
        Returns:
            filtered_patterns: List of filtered patterns
            filtered_pattern_types: Filtered pattern_types dict
        """
        if num_examples_per_type < 1 or num_examples_per_type > 40:
            print(f"⚠️  Warning: num_examples_per_type should be between 1-40. Using 40 (all examples).")
            num_examples_per_type = 40
        
        filtered_patterns = []
        filtered_pattern_types = {}
        
        for pattern_type, type_patterns in pattern_types.items():
            # Take only the first num_examples_per_type examples
            filtered_type_patterns = type_patterns[:num_examples_per_type]
            filtered_pattern_types[pattern_type] = filtered_type_patterns
            filtered_patterns.extend(filtered_type_patterns)
        
        return filtered_patterns, filtered_pattern_types
    
    @staticmethod
    def get_filtered_patterns_by_type(pattern_types, pattern_type, num_examples):
        """Get filtered patterns for a specific cognitive pattern type.
        
        Args:
            pattern_types: The full pattern_types dictionary
            pattern_type: The cognitive pattern type string
            num_examples: Number of examples to return (1-40)
            
        Returns:
            List of patterns for the specified type, limited to num_examples
        """
        if pattern_type not in pattern_types:
            available_types = list(pattern_types.keys())
            raise KeyError(f"Pattern type '{pattern_type}' not found. Available types: {available_types}")
        
        if num_examples < 1 or num_examples > 40:
            print(f"⚠️  Warning: num_examples should be between 1-40. Using all available examples.")
            num_examples = len(pattern_types[pattern_type])
        
        return pattern_types[pattern_type][:num_examples]
    
    @staticmethod
    def clear_memory():
        """Clear GPU/system memory."""
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        print("🧹 Memory cleared!")


if __name__ == "__main__":
    patcher = ActivationPatcher("gpt2-small")
    
    dataset_path = "/Users/ivanculo/Desktop/Projects/turn_point/data/final/positive_patterns.jsonl"
    
    patcher.experiment_with_dataset(dataset_path, num_samples=3, max_new_tokens=60)