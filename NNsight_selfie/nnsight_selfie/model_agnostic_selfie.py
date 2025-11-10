"""
Model-agnostic implementation of selfie-like functionality using NNsight.

This module provides a unified interface for neural network interpretation
that works across different transformer architectures (GPT, LLaMA, BERT, etc.)
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

try:
    import nnsight
except ImportError:
    raise ImportError("NNsight is required. Install it with: pip install nnsight")

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from .interpretation_prompt import InterpretationPrompt
from .utils import get_model_layers, get_layer_by_path
from .device_utils import get_optimal_device, get_device_map, ensure_device_compatibility


class ModelAgnosticSelfie:
    """
    Model-agnostic implementation of selfie functionality using NNsight.
    
    This class provides the ability to:
    1. Extract activations from any transformer model
    2. Inject activations at different layers
    3. Generate interpretations using activation steering
    4. Compute relevancy scores for interpretations
    
    Args:
        model_name_or_path: HuggingFace model identifier or path to local model (optional if model_instance is provided)
        tokenizer: Optional tokenizer (will be auto-loaded if not provided)
        device_map: Device mapping for model loading (default: "auto")
        model_instance: Optional existing nnsight.LanguageModel instance to reuse
        **kwargs: Additional arguments passed to nnsight.LanguageModel
    """
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        tokenizer=None,
        device_map: Optional[str] = None,
        device: Optional[str] = None,
        model_instance=None,
        **kwargs
    ):
        # Use existing model instance if provided
        if model_instance is not None:
            print("Using existing model instance...")
            self.model = model_instance
            self.device = str(model_instance.device) if hasattr(model_instance, 'device') else device or get_optimal_device()
            
            # Set up tokenizer
            if tokenizer is None:
                self.tokenizer = self.model.tokenizer
            else:
                self.tokenizer = tokenizer
                
            # Set model name for potential Gemma filtering (try to get from model config)
            if hasattr(model_instance, 'config') and hasattr(model_instance.config, '_name_or_path'):
                self.model_name = model_instance.config._name_or_path
            elif model_name_or_path:
                self.model_name = model_name_or_path
            else:
                self.model_name = "unknown"
                
        else:
            # Load new model instance
            if model_name_or_path is None:
                raise ValueError("Either model_name_or_path or model_instance must be provided")
                
            # Determine optimal device if not specified
            if device is None:
                device = get_optimal_device()
            
            self._load_new_model(model_name_or_path, tokenizer, device_map, device, **kwargs)
        
        # Common post-processing for both new and existing models
        self.model.eval()
        self.layer_paths = get_model_layers(self.model)
        
        # Filter out vision components for Gemma 3 4B models
        if hasattr(self, 'model_name') and self._is_gemma_3_4b():
            self.layer_paths = self._filter_vision_components(self.layer_paths)
            print(f"Filtered out vision components for Gemma 3 4B model.")
        
        print(f"Model loaded successfully with {len(self.layer_paths)} layers detected.")
        
        # Import DeviceManager here to avoid circular import issues
        from .device_utils import DeviceManager
        self.device_manager = DeviceManager()
        
    def _load_new_model(self, model_name_or_path, tokenizer, device_map, device, **kwargs):
        """Load a new model instance."""
        # Get appropriate device mapping
        if device_map is None:
            device_map = get_device_map(device)
        
        # Store device info
        self.device = device
        self.model_name = model_name_or_path
        
        # Initialize model with device-aware settings and quantization
        print(f"Initializing model on device: {device}")
        
        # Setup quantization config (only for CUDA devices)
        quantization_config = None
        # Check if quantization is explicitly disabled via kwargs
        load_in_8bit = kwargs.pop('load_in_8bit', False)  # Default to False - no quantization
        if BitsAndBytesConfig is not None and device == "cuda" and load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        
        # Extract dtype from kwargs to avoid duplicate parameter
        model_kwargs = kwargs.copy()
        model_dtype = model_kwargs.pop('dtype', torch.bfloat16)
        
        try:
            
            self.model = nnsight.LanguageModel(
                model_name_or_path,
                tokenizer=tokenizer,
                device_map=device_map,
                quantization_config=quantization_config,
                dtype=model_dtype,
                low_cpu_mem_usage=False,  # Avoid meta device issues
                **model_kwargs
            )
            
            # Apply device-specific optimizations if needed
            if device == "mps":
                self._apply_mps_optimizations()
            
        except Exception as e:
            if device == "mps":
                warnings.warn(
                    f"Failed to initialize model on MPS: {e}. "
                    "Falling back to CPU. This may happen with some models that "
                    "have operations not yet supported on MPS."
                )
                self.device = "cpu"
                self.model = nnsight.LanguageModel(
                    model_name_or_path,
                    tokenizer=tokenizer,
                    device_map="cpu",
                    **model_kwargs
                )
            else:
                raise e
        
        # Post-processing will be done in __init__
    
    def _is_gemma_3_4b(self) -> bool:
        """Check if the loaded model is Gemma 3 4B."""
        model_name_lower = self.model_name.lower()
        return "gemma" in model_name_lower and "3" in model_name_lower and "4b" in model_name_lower
    
    def _filter_vision_components(self, layer_paths: List[str]) -> List[str]:
        """Filter out vision components from layer paths."""
        return [path for path in layer_paths if 'vision_tower' not in path and 'vision_model' not in path]
    
    def _apply_mps_optimizations(self):
        """Apply MPS-specific optimizations and workarounds."""
        try:
            # Some operations might not be supported on MPS yet
            # Add any MPS-specific optimizations here
            if hasattr(self.model, 'config'):
                # Disable features that might cause issues on MPS
                pass
        except Exception as e:
            warnings.warn(f"MPS optimizations failed: {e}")
    
    def _ensure_tensor_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device."""
        return ensure_device_compatibility(tensor, self.device)
    
    def _should_use_chat_template(self) -> bool:
        """Heuristically decide whether to use a chat template for the current model."""
        tokenizer = getattr(self.model, 'tokenizer', None)
        has_apply = hasattr(tokenizer, 'apply_chat_template')
        model_name_lower = self.model_name.lower() if self.model_name else ""
        is_instruct_family = any(name in model_name_lower for name in ['gemma', 'llama', 'mistral', 'qwen', 'phi'])
        return bool(has_apply or is_instruct_family)
    
    def apply_chat_template(self, user_text: str, add_generation_prompt: bool = True) -> str:
        """
        Format a single-turn chat with the given user text using the model's chat template if available.
        Falls back to a simple "User: ...\nAssistant:" preamble if no template is exposed.

        Args:
            user_text: The user's input text
            add_generation_prompt: Whether to add generation prompt for model response

        Returns:
            Formatted text string
        """
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template') and callable(tokenizer.apply_chat_template):
            try:
                messages = [{"role": "user", "content": user_text}]
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            except Exception:
                # Fall through to simple formatting on any failure
                pass

        # Simple generic chat-style fallback (no special tokens)
        if add_generation_prompt:
            return f"User: {user_text}\nAssistant:"
        else:
            return f"User: {user_text}"

    def _find_model_response_start_position(self, formatted_text: str) -> Optional[int]:
        """
        Find the token position where the model's response begins in a chat-templated string.
        This is typically right after special tokens like <start_of_turn>model or <|assistant|>.

        Args:
            formatted_text: The chat-templated text

        Returns:
            Token index where model response starts, or None if not found
        """
        # Tokenize the formatted text
        tokens = self.model.tokenizer.encode(formatted_text)

        # Common model response markers in different chat templates
        model_markers = [
            '<start_of_turn>model',  # Gemma
            '<|assistant|>',          # Llama 3
            '<|im_start|>assistant',  # Qwen
            'Assistant:',             # Generic
            '\nAssistant:',          # Alternative
        ]

        # Try to find the marker in the text
        for marker in model_markers:
            if marker in formatted_text:
                # Find where this marker ends in the token sequence
                marker_tokens = self.model.tokenizer.encode(marker, add_special_tokens=False)

                # Search for the marker sequence in the full token list
                for i in range(len(tokens) - len(marker_tokens) + 1):
                    if tokens[i:i+len(marker_tokens)] == marker_tokens:
                        # Return position right after the marker
                        return i + len(marker_tokens)

        # Fallback: look for common special token IDs
        # This is model-specific and might need adjustment
        tokenizer = self.model.tokenizer

        # Try to find special tokens that indicate model turn
        special_token_candidates = []
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            special_token_candidates.append(tokenizer.bos_token_id)

        # For Gemma models, look for the last BOS token (indicates model turn)
        if 'gemma' in self.model_name.lower():
            bos_positions = [i for i, tok in enumerate(tokens) if tok == tokenizer.bos_token_id]
            if len(bos_positions) > 0:
                # Return the position right after the last BOS token
                return bos_positions[-1] + 1

        return None

    def get_concept_activations(
        self,
        concepts: Union[str, List[str]],
        layer_indices: Optional[List[int]] = None,
        use_chat_template: bool = False,
        prompt_template: str = "think about the {word}"
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Extract activations for specific concepts/words using optional chat template formatting.

        This method formats prompts as "think about the {word}" (or custom template),
        optionally applies chat templates, and extracts activations from where the word appears
        in the model's response section.

        Args:
            concepts: Single word/phrase or list of words/phrases to extract activations for
            layer_indices: List of layer indices to extract from (default: all layers)
            use_chat_template: Whether to apply chat template and capture from word in model response
            prompt_template: Template string with {word} placeholder (default: "think about the {word}")

        Returns:
            Dictionary mapping concept -> {layer_idx -> activation tensor}

        Example:
            >>> # Without chat template: captures from "think about the happiness"
            >>> # With chat template: formats as:
            >>> #   User: think about the happiness
            >>> #   Model: happiness <-- captures here
            >>>
            >>> activations = selfie.get_concept_activations(
            ...     ["happiness", "sadness", "joy"],
            ...     layer_indices=[10, 15, 20],
            ...     use_chat_template=True
            ... )
            >>> happy_vec = activations["happiness"][15]  # Layer 15 activation for "happiness"
        """
        # Normalize to list
        if isinstance(concepts, str):
            concepts = [concepts]

        if layer_indices is None:
            layer_indices = list(range(len(self.layer_paths)))

        results = {}

        for concept in concepts:
            # Format the prompt with the concept
            user_prompt = prompt_template.format(word=concept)

            # Apply chat template if requested
            if use_chat_template:
                # Create chat with the word in the assistant response
                tokenizer = self.model.tokenizer

                if hasattr(tokenizer, 'apply_chat_template') and callable(tokenizer.apply_chat_template):
                    try:
                        # Format with user message and partial assistant response containing the word
                        messages = [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": concept}
                        ]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False  # Don't add generation prompt since we have assistant content
                        )
                    except Exception:
                        # Fallback to simple formatting
                        formatted_prompt = f"User: {user_prompt}\nAssistant: {concept}"
                else:
                    # Simple fallback
                    formatted_prompt = f"User: {user_prompt}\nAssistant: {concept}"

                # Tokenize to find where the concept word appears
                tokens = tokenizer.encode(formatted_prompt)

                # Find the position of the concept word in the model response section
                # We look for it starting from the end since it's in the assistant response
                concept_tokens = tokenizer.encode(concept, add_special_tokens=False)

                # Search for the concept token sequence in the full prompt
                capture_pos = None
                for i in range(len(tokens) - len(concept_tokens), -1, -1):
                    if tokens[i:i+len(concept_tokens)] == concept_tokens:
                        # Found it! Capture at the first token of the concept
                        capture_pos = i
                        break

                if capture_pos is None:
                    warnings.warn(
                        f"Could not find concept '{concept}' tokens in formatted prompt. "
                        f"Falling back to last token position."
                    )
                    capture_pos = len(tokens) - 1

                # Extract activations at the concept position
                activations = self.get_activations(
                    formatted_prompt,
                    layer_indices=layer_indices,
                    token_indices=[capture_pos]
                )

                # Reshape to single tensor per layer
                concept_activations = {}
                for layer_idx in activations:
                    if isinstance(activations[layer_idx], list):
                        concept_activations[layer_idx] = activations[layer_idx][0]
                    else:
                        concept_activations[layer_idx] = activations[layer_idx][:, capture_pos, :]

            else:
                # Without chat template: extract from last token of the prompt
                tokens = self.model.tokenizer.encode(user_prompt)
                last_token_pos = len(tokens) - 1

                activations = self.get_activations(
                    user_prompt,
                    layer_indices=layer_indices,
                    token_indices=[last_token_pos]
                )

                # Reshape to single tensor per layer
                concept_activations = {}
                for layer_idx in activations:
                    if isinstance(activations[layer_idx], list):
                        concept_activations[layer_idx] = activations[layer_idx][0]
                    else:
                        concept_activations[layer_idx] = activations[layer_idx][:, last_token_pos, :]

            results[concept] = concept_activations

        return results

    def get_activations(
        self,
        prompt: str,
        layer_indices: Optional[List[int]] = None,
        token_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from specified layers and tokens.
        
        Args:
            prompt: Input text prompt
            layer_indices: List of layer indices to extract from (default: all layers)
            token_indices: List of token positions to extract (default: all positions)
            
        Returns:
            Dictionary mapping layer_idx -> activations tensor
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.layer_paths)))
            
        activations = {}
        
        with self.model.generate(prompt, max_new_tokens=1) as tracer:
            for layer_idx in layer_indices:
                layer = get_layer_by_path(self.model, self.layer_paths[layer_idx])
                
                if token_indices is not None:
                    # Extract specific token positions
                    layer_activations = []
                    for token_idx in token_indices:
                        activation = layer.output[0][:, token_idx, :].save()
                        layer_activations.append(activation)
                    activations[layer_idx] = layer_activations
                else:
                    # Extract all token positions
                    activations[layer_idx] = layer.output[0].save()
        
        # Ensure activations are on the correct device
        for layer_idx in activations:
            if isinstance(activations[layer_idx], list):
                activations[layer_idx] = [
                    self._ensure_tensor_device(act) for act in activations[layer_idx]
                ]
            else:
                activations[layer_idx] = self._ensure_tensor_device(activations[layer_idx])
                    
        return activations
    
    def inject_activation(
        self,
        prompt: str,
        activation: torch.Tensor,
        injection_layer: int,
        injection_positions: List[int],
        overlay_strength: float = 1.0,
        replacing_mode: str = 'normalized',
        max_new_tokens: int = 30,
        use_chat_template: bool = False
    ) -> torch.Tensor:
        """
        Generate text with injected activations.
        
        Args:
            prompt: Input prompt for generation
            activation: Activation tensor to inject
            injection_layer: Layer index to inject at
            injection_positions: Token positions to inject at
            overlay_strength: Strength of intervention (0-1)
            replacing_mode: 'normalized' or 'addition'
            max_new_tokens: Maximum new tokens to generate
            use_chat_template: Whether to apply chat template formatting
            
        Returns:
            Generated token IDs
        """
        # Note: Chat template handling is now done by InterpretationPrompt.with_chat_template()
        # This ensures proper token position alignment
        
        # Ensure activation is on the correct device
        activation = self._ensure_tensor_device(activation)
        
        with self.model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
            layer = get_layer_by_path(self.model, self.layer_paths[injection_layer])
            
            # Get original activations
            original_output = layer.output[0]
            
            # Prepare injection - ensure device compatibility
            batch_size, seq_len, hidden_size = original_output.shape
            try:
                activation_expanded = activation.expand(batch_size, len(injection_positions), hidden_size)
            except Exception as e:
                if self.device == "mps":
                    # MPS might have issues with expand, try alternative
                    activation_expanded = activation.repeat(batch_size, len(injection_positions), 1)
                else:
                    raise e
            
            # Apply intervention
            for i, pos in enumerate(injection_positions):
                if replacing_mode == 'normalized':
                    original_output[:, pos, :] = (
                        overlay_strength * activation_expanded[:, i, :] +
                        (1 - overlay_strength) * original_output[:, pos, :]
                    )
                elif replacing_mode == 'addition':
                    original_output[:, pos, :] += overlay_strength * activation_expanded[:, i, :]
            
            output_ids = self.model.generator.output.save()
            
        return output_ids
    
    def interpret(
        self,
        original_prompt: str,
        interpretation_prompt: InterpretationPrompt,
        tokens_to_interpret: List[Tuple[int, int]],  # [(layer, token), ...]
        injection_layer: int = 3,
        batch_size: int = 8,
        max_new_tokens: int = 30,
        overlay_strength: float = 1.0,
        replacing_mode: str = 'normalized',
        use_chat_template: bool = False
    ) -> Dict[str, Any]:
        """
        Interpret specific tokens using activation injection.
        
        Args:
            original_prompt: The prompt containing tokens to interpret
            interpretation_prompt: InterpretationPrompt object with template and positions
            tokens_to_interpret: List of (layer_idx, token_idx) tuples
            injection_layer: Layer to inject activations at
            batch_size: Batch size for processing
            max_new_tokens: Max tokens to generate for each interpretation
            overlay_strength: Strength of intervention
            replacing_mode: Mode for replacing activations
            use_chat_template: Whether to apply chat template formatting (default: False for compatibility)
            
        Returns:
            Dictionary containing interpretation results
        """
        print(f"Interpreting '{original_prompt}' with '{interpretation_prompt.interpretation_prompt}'")
        
        # Apply chat template to interpretation prompt if requested
        if use_chat_template:
            interpretation_prompt.with_chat_template(True)
        else:
            interpretation_prompt.with_chat_template(False)
        
        # Get original activations (always from raw text, no chat template)
        original_activations = self.get_activations(original_prompt)
        
        interpretation_df = {
            'prompt': [],
            'interpretation': [],
            'layer': [],
            'token': [],
            'token_decoded': [],
            'relevancy_score': [],
        }
        
        # Process in batches
        for batch_start in tqdm(range(0, len(tokens_to_interpret), batch_size)):
            batch_tokens = tokens_to_interpret[batch_start:batch_start + batch_size]
            batch_interpretations = []
            
            for retrieve_layer, retrieve_token in batch_tokens:
                # Get activation for this token
                activation = original_activations[retrieve_layer][:, retrieve_token, :].unsqueeze(0)
                
                # Generate interpretation
                output_ids = self.inject_activation(
                    interpretation_prompt.interpretation_prompt,
                    activation,
                    injection_layer,
                    interpretation_prompt.insert_locations,
                    overlay_strength,
                    replacing_mode,
                    max_new_tokens,
                    False  # Chat template already applied to interpretation_prompt
                )
                
                # Decode interpretation
                prompt_len = len(interpretation_prompt.interpretation_prompt_inputs['input_ids'][0])
                interpretation_tokens = output_ids[0, prompt_len:]
                interpretation_text = self.model.tokenizer.decode(
                    interpretation_tokens, 
                    skip_special_tokens=True
                )
                
                batch_interpretations.append(interpretation_text)
                
                # Store results
                interpretation_df['prompt'].append(original_prompt)
                interpretation_df['interpretation'].append(interpretation_text)
                interpretation_df['layer'].append(retrieve_layer)
                interpretation_df['token'].append(retrieve_token)
                
                # Decode original token
                original_inputs = self.model.tokenizer(original_prompt, return_tensors="pt")
                if retrieve_token < len(original_inputs['input_ids'][0]):
                    token_text = self.model.tokenizer.decode(
                        original_inputs['input_ids'][0][retrieve_token]
                    )
                else:
                    token_text = "<out_of_range>"
                interpretation_df['token_decoded'].append(token_text)
                
            # Compute relevancy scores (placeholder - can be enhanced)
            interpretation_df['relevancy_score'].extend([[1.0] * max_new_tokens] * len(batch_tokens))
        
        return interpretation_df
    
    def interpret_vectors(
        self,
        vectors: List[torch.Tensor],
        interpretation_prompt: InterpretationPrompt,
        injection_layer: int = 3,
        batch_size: int = 8,
        max_new_tokens: int = 30,
        overlay_strength: float = 1.0,
        use_chat_template: bool = False
    ) -> List[str]:
        """
        Interpret arbitrary activation vectors.
        
        Args:
            vectors: List of activation tensors to interpret
            interpretation_prompt: InterpretationPrompt object
            injection_layer: Layer to inject at
            batch_size: Batch size for processing
            max_new_tokens: Max tokens to generate
            overlay_strength: Intervention strength
            use_chat_template: Whether to apply chat template formatting (default: False for compatibility)
            
        Returns:
            List of interpretation strings
        """
        interpretations = []
        
        # Apply chat template to interpretation prompt if requested
        if use_chat_template:
            interpretation_prompt.with_chat_template(True)
        else:
            interpretation_prompt.with_chat_template(False)
        
        for i in tqdm(range(0, len(vectors), batch_size)):
            batch_vectors = vectors[i:i + batch_size]
            
            for vector in batch_vectors:
                output_ids = self.inject_activation(
                    interpretation_prompt.interpretation_prompt,
                    vector.unsqueeze(0),
                    injection_layer,
                    interpretation_prompt.insert_locations,
                    overlay_strength,
                    'normalized',
                    max_new_tokens,
                    False  # Chat template already applied to interpretation_prompt
                )
                
                prompt_len = len(interpretation_prompt.interpretation_prompt_inputs['input_ids'][0])
                interpretation_tokens = output_ids[0, prompt_len:]
                interpretation_text = self.model.tokenizer.decode(
                    interpretation_tokens,
                    skip_special_tokens=True
                )
                
                interpretations.append(interpretation_text)
                
        return interpretations
    
    def compute_relevancy_scores(
        self,
        original_prompt: str,
        interpretation_outputs: torch.Tensor,
        intervention_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevancy scores by comparing model outputs with/without intervention.
        
        Args:
            original_prompt: Original input prompt
            interpretation_outputs: Model outputs with intervention
            intervention_outputs: Model outputs without intervention
            
        Returns:
            Relevancy scores tensor
        """
        # Get logits for both conditions
        with self.model.generate(original_prompt, max_new_tokens=1) as tracer:
            original_logits = self.model.lm_head.output.save()
            
        # Compute differences (simplified version)
        # This can be enhanced with more sophisticated metrics
        diff = torch.abs(interpretation_outputs - intervention_outputs)
        return diff.mean(dim=-1)