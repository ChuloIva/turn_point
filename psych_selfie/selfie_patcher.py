"""
SelfIE Psychology Wrapper

A wrapper class that provides a similar interface to ActivationPatcher but uses
the SelfIE technique for self-interpretation of embeddings instead of activation patching.

This allows for interpretable analysis of cognitive patterns using the model's own
ability to understand and describe its internal representations.
"""

import sys
import os
import torch
import json
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum
import warnings
import importlib.util

# Import multi-text aggregation functionality
from multi_text_aggregation import MultiTextActivationExtractor, AggregationStrategy

# Add the selfie library to path
selfie_path = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'selfie')
sys.path.insert(0, selfie_path)

# Add activation_patcher path for dataset utilities
activation_patcher_path = os.path.join(os.path.dirname(__file__), '..', 'manual_activation_patching')
sys.path.insert(0, activation_patcher_path)

# Patch missing function in huggingface_hub for transformers compatibility
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'split_torch_state_dict_into_shards'):
        def split_torch_state_dict_into_shards(*args, **kwargs):
            raise NotImplementedError('This function is not available in older huggingface_hub versions')
        huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
except Exception:
    pass  # Ignore if patching fails

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from selfie.interpret import InterpretationPrompt, interpret, interpret_vectors
    SELFIE_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to import SelfIE dependencies: {e}")
    warnings.warn("Make sure you have installed the requirements and selfie library")
    SELFIE_AVAILABLE = False


class TokenSelectionStrategy(Enum):
    """Token selection strategies for SelfIE interpretation"""
    LAST_TOKEN = "last_token"
    LAST_COUPLE = "last_couple"
    MID_TOKENS = "mid_tokens"
    ALL_TOKENS = "all_tokens"
    KEYWORDS = "keywords"


class SelfIEPatcher:
    """
    SelfIE-based interpretation wrapper that mimics ActivationPatcher interface
    but uses self-interpretation instead of activation patching.
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the SelfIE patcher with a model.
        
        Args:
            model_name: HuggingFace model name (must be LLaMA-compatible for SelfIE)
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        if not SELFIE_AVAILABLE:
            raise ImportError("SelfIE library is not available. Check installation.")
            
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_config = {}
        self.multi_text_extractor = None  # Initialize after model loading
        
        # Default interpretation templates
        self.interpretation_templates = {
            'cognitive_pattern': ("[INST]", 0, 0, 0, 0, 0, "[/INST] This represents a cognitive pattern related to:"),
            'emotional_state': ("[INST]", 0, 0, 0, 0, 0, "[/INST] This indicates an emotional state of:"),
            'general_concept': ("[INST]", 0, 0, 0, 0, 0, "[/INST] This concept can be described as:"),
            'decision_making': ("[INST]", 0, 0, 0, 0, 0, "[/INST] This decision-making process involves:"),
            'psychological_state': ("[INST]", 0, 0, 0, 0, 0, "[/INST] This psychological state reflects:")
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading SelfIE-compatible model: {self.model_name}")
        
        # Handle device selection like activation_patcher.py
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("Using CUDA GPU")
            else:
                self.device = "cpu"
                print("Using CPU")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token if not present (required for SelfIE)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device if self.device != "auto" else "auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.model.eval()
            
            # Set model configuration
            self.model_config = {
                'family': self.model_name.split('/')[0] if '/' in self.model_name else 'unknown',
                'size': 'unknown',  # Could extract from model name
                'layers': getattr(self.model.config, 'num_hidden_layers', 'unknown')
            }
            
            print(f"✓ Model loaded successfully")
            print(f"  - Model: {self.model_name}")
            print(f"  - Layers: {self.model_config['layers']}")
            print(f"  - Device: {next(self.model.parameters()).device}")
            print(f"  - Vocab size: {len(self.tokenizer)}")
            
            # Initialize multi-text extractor
            self.multi_text_extractor = MultiTextActivationExtractor(
                self.model, self.tokenizer, self.device
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def interpret_text(
        self,
        text: str,
        layers_to_interpret: List[int] = None,
        token_positions: List[int] = None,
        interpretation_template: str = 'cognitive_pattern',
        max_new_tokens: int = 30,
        batch_size: int = 2,
        k: int = 1
    ) -> pd.DataFrame:
        """
        Interpret internal representations of text using SelfIE.
        
        Args:
            text: Input text to interpret
            layers_to_interpret: Which layers to extract representations from
            token_positions: Which token positions to interpret
            interpretation_template: Template for interpretation prompt
            max_new_tokens: Maximum tokens to generate for interpretation
            batch_size: Batch size for processing
            k: Layer to insert interpreted representations into
            
        Returns:
            DataFrame with interpretation results
        """
        if layers_to_interpret is None:
            layers_to_interpret = [-1]  # Default to last layer
            
        # Tokenize input to determine valid positions
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        seq_len = inputs['input_ids'].shape[-1]
        
        if token_positions is None:
            # Default to last few tokens
            token_positions = list(range(max(0, seq_len-3), seq_len))
        
        # Create interpretation prompt
        if interpretation_template in self.interpretation_templates:
            template_tuple = self.interpretation_templates[interpretation_template]
        else:
            # Use provided string as custom template
            template_tuple = ("[INST]", 0, 0, 0, 0, 0, f"[/INST] {interpretation_template}")
            
        interpretation_prompt = InterpretationPrompt(self.tokenizer, template_tuple)
        
        # Create tokens to interpret list
        tokens_to_interpret = []
        for layer in layers_to_interpret:
            for pos in token_positions:
                if pos < seq_len:  # Ensure position is valid
                    tokens_to_interpret.append((layer, pos))
        
        # Perform interpretation
        results = interpret(
            original_prompt=text,
            tokens_to_interpret=tokens_to_interpret,
            model=self.model,
            interpretation_prompt=interpretation_prompt,
            bs=batch_size,
            max_new_tokens=max_new_tokens,
            k=k,
            tokenizer=self.tokenizer
        )
        
        return pd.DataFrame(results)
    
    def interpret_and_compare(
        self,
        clean_text: str,
        corrupted_text: str,
        layer_idx: int = -1,
        token_selection_strategy: TokenSelectionStrategy = TokenSelectionStrategy.LAST_TOKEN,
        interpretation_template: str = 'cognitive_pattern',
        max_new_tokens: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Interpret and compare representations between clean and corrupted text.
        This mimics the patch_and_generate interface but uses interpretation instead.
        
        Args:
            clean_text: Clean/positive text to interpret
            corrupted_text: Corrupted/negative text to interpret  
            layer_idx: Layer to extract representations from
            token_selection_strategy: How to select tokens for interpretation
            interpretation_template: Template for interpretation
            max_new_tokens: Maximum tokens for interpretation
            
        Returns:
            Dictionary containing interpretations and comparison results
        """
        # Get token positions based on strategy
        clean_positions = self._get_token_positions(clean_text, token_selection_strategy)
        corrupted_positions = self._get_token_positions(corrupted_text, token_selection_strategy)
        
        # Interpret clean text
        clean_results = self.interpret_text(
            clean_text,
            layers_to_interpret=[layer_idx],
            token_positions=clean_positions,
            interpretation_template=interpretation_template,
            max_new_tokens=max_new_tokens
        )
        
        # Interpret corrupted text
        corrupted_results = self.interpret_text(
            corrupted_text,
            layers_to_interpret=[layer_idx],
            token_positions=corrupted_positions,
            interpretation_template=interpretation_template,
            max_new_tokens=max_new_tokens
        )
        
        return {
            'clean_interpretations': clean_results,
            'corrupted_interpretations': corrupted_results,
            'clean_text': clean_text,
            'corrupted_text': corrupted_text,
            'layer': layer_idx,
            'strategy': token_selection_strategy.value
        }
    
    def _get_token_positions(self, text: str, strategy: TokenSelectionStrategy) -> List[int]:
        """Get token positions based on selection strategy"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        seq_len = inputs['input_ids'].shape[-1]
        
        if strategy == TokenSelectionStrategy.LAST_TOKEN:
            return [seq_len - 1]
        elif strategy == TokenSelectionStrategy.LAST_COUPLE:
            return list(range(max(0, seq_len-3), seq_len))
        elif strategy == TokenSelectionStrategy.MID_TOKENS:
            mid = seq_len // 2
            return list(range(max(0, mid-2), min(seq_len, mid+2)))
        elif strategy == TokenSelectionStrategy.ALL_TOKENS:
            return list(range(seq_len))
        elif strategy == TokenSelectionStrategy.KEYWORDS:
            # Simple keyword detection - in practice could be more sophisticated
            tokens = self.tokenizer.tokenize(text)
            # Focus on longer tokens that might be keywords
            positions = [i for i, token in enumerate(tokens) if len(token) > 3]
            return positions[:5]  # Limit to first 5 keywords
        else:
            return [seq_len - 1]  # Default to last token
    
    # Placeholder methods for future advanced features
    def supervised_control(self, target_concept: str, control_strength: float = 1.0):
        """
        Placeholder for supervised control using SelfIE.
        This would implement the supervised control method from the SelfIE paper.
        
        TODO: Implement supervised control for editing concepts in hidden representations
        """
        raise NotImplementedError("Supervised control not yet implemented")
    
    def reinforcement_control(self, harmful_concepts: List[str]):
        """
        Placeholder for reinforcement control using SelfIE.
        This would implement RLHF on hidden embeddings to erase harmful knowledge.
        
        TODO: Implement reinforcement control for removing harmful knowledge
        """
        raise NotImplementedError("Reinforcement control not yet implemented")
    
    def batch_interpret_patterns(
        self, 
        patterns: List[Dict], 
        text_type: str = "negative",
        aggregation_strategy: AggregationStrategy = AggregationStrategy.ATTENTION_WEIGHTED,
        layers_to_interpret: List[int] = None,
        interpretation_template: str = 'cognitive_pattern',
        max_new_tokens: int = 30,
        batch_size: int = 2,
        k: int = 1
    ) -> pd.DataFrame:
        """
        Process multiple cognitive patterns and aggregate their activations for interpretation.
        
        This method extracts activations from multiple text examples, aggregates them using 
        advanced semantic preservation techniques, then generates interpretations.
        
        Args:
            patterns: List of cognitive pattern dictionaries
            text_type: Which text variant to use ("positive", "negative", "transition")
            aggregation_strategy: How to aggregate activations across patterns
            layers_to_interpret: Which layers to extract from
            interpretation_template: Template for interpretation prompt
            max_new_tokens: Maximum tokens for interpretation
            batch_size: Batch size for processing interpretations
            k: Layer to insert aggregated activations into
            
        Returns:
            DataFrame with aggregated interpretation results
        """
        if layers_to_interpret is None:
            layers_to_interpret = [-1]  # Default to last layer
            
        # Import text extraction utility
        try:
            from activation_patcher import ActivationPatcher
            get_pattern_text = ActivationPatcher.get_pattern_text
        except ImportError:
            raise ImportError("Could not import pattern text utilities")
        
        # Extract texts from patterns
        texts = []
        pattern_names = []
        for pattern in patterns:
            try:
                text = get_pattern_text(pattern, text_type)
                texts.append(text)
                pattern_names.append(pattern.get('cognitive_pattern_name', 'Unknown'))
            except Exception as e:
                print(f"Warning: Could not extract text from pattern: {e}")
                continue
        
        if not texts:
            raise ValueError("No valid texts could be extracted from patterns")
            
        print(f"🔄 Processing {len(texts)} patterns with {aggregation_strategy.value} aggregation")
        print(f"📝 Text type: {text_type}")
        print(f"🏗️ Layers: {layers_to_interpret}")
        print(f"🎯 Aggregation: {aggregation_strategy.value}")
        
        # Extract and aggregate activations across all texts
        aggregated_activations = self.multi_text_extractor.extract_multi_text_activations(
            texts=texts,
            layers_to_extract=layers_to_interpret,
            token_positions=None,  # Use default (last few tokens)
            aggregation_strategy=aggregation_strategy
        )
        
        print(f"✓ Extracted and aggregated {len(aggregated_activations)} activation combinations")
        
        # Create interpretation prompt
        if interpretation_template in self.interpretation_templates:
            template_tuple = self.interpretation_templates[interpretation_template]
        else:
            template_tuple = ("[INST]", 0, 0, 0, 0, 0, f"[/INST] {interpretation_template}")
            
        from selfie.interpret import InterpretationPrompt, generate_interpret
        interpretation_prompt = InterpretationPrompt(self.tokenizer, template_tuple)
        
        # Prepare interpretation tasks using aggregated activations
        all_insert_infos = []
        interpretation_df = {
            'prompt': [],
            'interpretation': [],
            'layer': [],
            'token': [],
            'token_decoded': [],
            'aggregation_strategy': [],
            'num_patterns': [],
            'pattern_names': []
        }
        
        for (retrieve_layer, retrieve_token), aggregated_activation in aggregated_activations.items():
            insert_info = {}
            insert_info['replacing_mode'] = 'normalized'
            insert_info['overlay_strength'] = 1
            insert_info['retrieve_layer'] = retrieve_layer
            insert_info['retrieve_token'] = retrieve_token
            
            # Insert aggregated activation at specified layer k
            for layer_idx, layer in enumerate(self.model.model.layers):
                if layer_idx == k:
                    insert_locations = interpretation_prompt.insert_locations
                    # Use aggregated activation instead of single text activation
                    insert_info[layer_idx] = (
                        insert_locations, 
                        aggregated_activation.unsqueeze(0).repeat(1, len(insert_locations), 1)
                    )
            all_insert_infos.append(insert_info)
        
        # Generate interpretations in batches
        from tqdm import tqdm
        
        for batch_start_idx in tqdm(range(0, len(all_insert_infos), batch_size), 
                                  desc="Generating interpretations"):
            with torch.no_grad():
                batch_insert_infos = all_insert_infos[batch_start_idx:min(batch_start_idx+batch_size, len(all_insert_infos))]
                
                # Create batched interpretation prompts
                batched_interpretation_prompt_model_inputs = self.tokenizer(
                    [interpretation_prompt.interpretation_prompt] * len(batch_insert_infos), 
                    return_tensors="pt"
                ).to(self.device)
                
                repeat_prompt_n_tokens = interpretation_prompt.interpretation_prompt_model_inputs['input_ids'].shape[-1]
                
                # Generate interpretations with aggregated activations
                output = generate_interpret(
                    **batched_interpretation_prompt_model_inputs, 
                    model=self.model, 
                    max_new_tokens=max_new_tokens, 
                    insert_info=batch_insert_infos, 
                    pad_token_id=self.tokenizer.eos_token_id, 
                    output_attentions=False
                )
                
                # Decode interpretations
                cropped_interpretation_tokens = output[:, repeat_prompt_n_tokens:]
                cropped_interpretation = self.tokenizer.batch_decode(
                    cropped_interpretation_tokens, skip_special_tokens=True
                )
                
                # Store results
                for i, interpretation in enumerate(cropped_interpretation):
                    interpretation_df['prompt'].append(f"Aggregated from {len(texts)} patterns")
                    interpretation_df['interpretation'].append(interpretation.strip())
                    interpretation_df['layer'].append(batch_insert_infos[i]['retrieve_layer'])
                    interpretation_df['token'].append(batch_insert_infos[i]['retrieve_token'])
                    interpretation_df['token_decoded'].append(f"aggregated_token_{batch_insert_infos[i]['retrieve_token']}")
                    interpretation_df['aggregation_strategy'].append(aggregation_strategy.value)
                    interpretation_df['num_patterns'].append(len(texts))
                    interpretation_df['pattern_names'].append("; ".join(pattern_names[:5]))  # First 5 names
        
        print(f"🎊 Generated {len(interpretation_df['interpretation'])} aggregated interpretations")
        return pd.DataFrame(interpretation_df)
    
    def visualize_interpretations(self, results: pd.DataFrame):
        """
        Placeholder for visualization of interpretation results.
        
        TODO: Implement visualization tools for interpretation analysis
        """
        raise NotImplementedError("Interpretation visualization not yet implemented")
    
    def export_interpretations(self, results: pd.DataFrame, format: str = 'json'):
        """
        Placeholder for exporting interpretation results.
        
        TODO: Implement export functionality for results
        """
        raise NotImplementedError("Export functionality not yet implemented")
    
    # Utility methods to maintain interface compatibility
    def reset_hooks(self):
        """Compatibility method - SelfIE doesn't use hooks"""
        print("✓ SelfIE doesn't use hooks - model state is clean")
    
    def check_model_info(self):
        """Display model information"""
        print(f"SelfIE Model Information:")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {next(self.model.parameters()).device}")
        print(f"  Layers: {self.model_config['layers']}")
        print(f"  Family: {self.model_config['family']}")
        
    @staticmethod
    def clear_memory():
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU memory cleared")
        else:
            print("✓ No GPU memory to clear")


# Dataset loading utilities - import from activation_patcher.py
def _import_dataset_utilities():
    """Import dataset utilities from activation_patcher.py"""
    try:
        # Import the activation_patcher module  
        spec = importlib.util.spec_from_file_location(
            "activation_patcher", 
            os.path.join(activation_patcher_path, "activation_patcher.py")
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader is not None
        spec.loader.exec_module(module)
        
        # Get the ActivationPatcher class and extract static methods
        ActivationPatcher = module.ActivationPatcher
        
        return {
            'load_cognitive_patterns': ActivationPatcher.load_cognitive_patterns,
            'get_pattern_by_index': ActivationPatcher.get_pattern_by_index,
            'get_pattern_by_type': ActivationPatcher.get_pattern_by_type,
            'get_pattern_text': ActivationPatcher.get_pattern_text,
            'filter_patterns_by_count': ActivationPatcher.filter_patterns_by_count,
            'get_filtered_patterns_by_type': ActivationPatcher.get_filtered_patterns_by_type,
            'list_available_pattern_types': ActivationPatcher.list_available_pattern_types,
            'show_pattern_info': ActivationPatcher.show_pattern_info,
            'get_random_pattern_by_type': ActivationPatcher.get_random_pattern_by_type,
            'get_patterns_by_type': ActivationPatcher.get_patterns_by_type
        }
    except Exception as e:
        warnings.warn(f"Could not import dataset utilities from activation_patcher.py: {e}")
        return {}

# Import dataset utilities
try:
    _dataset_utils = _import_dataset_utilities()
    
    # Expose dataset utilities at module level for easy import
    load_cognitive_patterns = _dataset_utils.get('load_cognitive_patterns')
    get_pattern_by_index = _dataset_utils.get('get_pattern_by_index') 
    get_pattern_by_type = _dataset_utils.get('get_pattern_by_type')
    get_pattern_text = _dataset_utils.get('get_pattern_text')
    filter_patterns_by_count = _dataset_utils.get('filter_patterns_by_count')
    get_filtered_patterns_by_type = _dataset_utils.get('get_filtered_patterns_by_type')
    list_available_pattern_types = _dataset_utils.get('list_available_pattern_types')
    show_pattern_info = _dataset_utils.get('show_pattern_info')
    get_random_pattern_by_type = _dataset_utils.get('get_random_pattern_by_type')
    get_patterns_by_type = _dataset_utils.get('get_patterns_by_type')
    
    print("✓ Dataset utilities imported successfully from activation_patcher.py")
    
except Exception as e:
    warnings.warn(f"Failed to import dataset utilities: {e}")
    # Provide None values for missing functions
    load_cognitive_patterns = None
    get_pattern_by_index = None
    get_pattern_by_type = None
    get_pattern_text = None
    filter_patterns_by_count = None
    get_filtered_patterns_by_type = None
    list_available_pattern_types = None
    show_pattern_info = None
    get_random_pattern_by_type = None
    get_patterns_by_type = None
    
