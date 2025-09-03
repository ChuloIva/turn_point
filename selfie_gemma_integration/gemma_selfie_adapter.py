#!/usr/bin/env python3
"""
Gemma SelfIE Adapter

This module adapts the SelfIE library to work with pre-captured Gemma-2-2b activations
from the cognitive pattern analysis pipeline. It provides utilities to:

1. Load pre-captured activations from the activations cache
2. Convert activation format to be compatible with SelfIE
3. Generate natural language interpretations of specific layer activations
4. Integrate with the cognitive pattern analysis workflow

Key differences from standard SelfIE usage:
- Works with pre-captured activations instead of live model inference
- Adapted for Gemma-2-2b architecture (not LLaMA)
- Focuses on cognitive pattern analysis use case
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import sys
import os

# Add the parent directory to path to import from the main project
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Add selfie to path
selfie_path = parent_dir / "selfie"
sys.path.insert(0, str(selfie_path))

# Import project modules
from model_loader import ModelLoader
from activation_capture import ActivationCapturer

# Import SelfIE components
try:
    from selfie.interpret import InterpretationPrompt, interpret_vectors
    from transformers import AutoTokenizer, AutoModelForCausalLM
    SELFIE_AVAILABLE = True
    print("✅ SelfIE imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SelfIE: {e}")
    SELFIE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaSelfieAdapter:
    """
    Adapter class that bridges pre-captured Gemma activations with SelfIE interpretations.
    
    This class allows you to:
    1. Load cached Gemma activations from your cognitive pattern analysis
    2. Generate natural language interpretations using SelfIE
    3. Analyze activations at different layers and token positions
    4. Export results in various formats for further analysis
    """
    
    def __init__(self, base_path: str = None, device: str = "auto"):
        """
        Initialize the Gemma SelfIE adapter.
        
        Args:
            base_path: Base directory path for the project
            device: Device to use for computation ("auto", "cuda", "cpu", "mps")
        """
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.activations_cache = {}
        self.cognitive_patterns = []
        
        # Paths
        self.activations_dir = self.base_path / "activations"
        self.output_dir = self.base_path / "selfie_gemma_integration" / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Initialized Gemma SelfIE Adapter")
        print(f"   Device: {self.device}")
        print(f"   Activations directory: {self.activations_dir}")
        print(f"   Output directory: {self.output_dir}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def load_model_and_tokenizer(self, model_name: str = "google/gemma-2-2b-it") -> bool:
        """
        Load the Gemma model and tokenizer for SelfIE interpretation.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n🤖 Loading model and tokenizer: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                device_map="auto"
            )
            print(f"   ✅ Tokenizer loaded")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            print(f"   ✅ Model loaded")
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            return False
    
    def load_cached_activations(self, cache_file: str = None) -> Dict[str, Any]:
        """
        Load pre-cached activations from the cognitive pattern analysis.
        
        Args:
            cache_file: Specific cache file to load, or None to load all available
            
        Returns:
            dict: Loaded activation data
        """
        print(f"\n📊 Loading cached activations...")
        
        if cache_file:
            # Load specific file
            cache_files = [cache_file]
        else:
            # Load all available cache files
            cache_files = list(self.activations_dir.glob("activations_*.pt"))
        
        if not cache_files:
            raise FileNotFoundError(f"No activation cache files found in {self.activations_dir}")
        
        loaded_data = {}
        
        for file_path in cache_files:
            if isinstance(file_path, str):
                file_path = Path(file_path)
            elif not file_path.is_absolute():
                file_path = self.activations_dir / file_path
            
            print(f"   Loading: {file_path.name}")
            
            try:
                data = torch.load(file_path, map_location='cpu')
                
                # Extract key information - handle different activation file formats
                cache_key = file_path.stem.replace("activations_", "")
                
                # Separate activation tensors from metadata
                activations = {}
                metadata = {}
                enriched_metadata = []
                
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        # This is an activation tensor
                        activations[key] = value
                    elif key == 'enriched_metadata':
                        enriched_metadata = value
                    elif key in ['metadata', 'metadata_info']:
                        metadata = value
                    else:
                        # Other metadata
                        metadata[key] = value
                
                loaded_data[cache_key] = {
                    'activations': activations,
                    'metadata': metadata,
                    'enriched_metadata': enriched_metadata,
                    'file_path': str(file_path),
                    'load_time': datetime.now().isoformat()
                }
                
                # Log summary
                print(f"     ✅ Loaded {len(activations)} activation tensors")
                for key, tensor in activations.items():
                    print(f"       {key}: {tensor.shape}")
                
            except Exception as e:
                print(f"     ❌ Failed to load {file_path.name}: {e}")
        
        self.activations_cache = loaded_data
        print(f"   📋 Total cache entries loaded: {len(loaded_data)}")
        return loaded_data
    
    def extract_cognitive_patterns(self) -> List[Dict[str, Any]]:
        """
        Extract cognitive patterns from loaded activation cache.
        
        Returns:
            list: List of cognitive patterns with associated data
        """
        print(f"\n🧠 Extracting cognitive patterns...")
        
        if not self.activations_cache:
            raise ValueError("No activations loaded. Call load_cached_activations() first.")
        
        patterns = []
        
        for cache_key, cache_data in self.activations_cache.items():
            enriched_metadata = cache_data.get('enriched_metadata', [])
            
            for entry in enriched_metadata:
                pattern_name = entry.get('cognitive_pattern_name')
                if pattern_name:
                    patterns.append({
                        'name': pattern_name,
                        'cache_key': cache_key,
                        'metadata': entry,
                        'activations_available': list(cache_data['activations'].keys())
                    })
        
        # Remove duplicates based on pattern name
        unique_patterns = {}
        for pattern in patterns:
            name = pattern['name']
            if name not in unique_patterns:
                unique_patterns[name] = pattern
        
        self.cognitive_patterns = list(unique_patterns.values())
        
        print(f"   ✅ Found {len(self.cognitive_patterns)} unique cognitive patterns:")
        for i, pattern in enumerate(self.cognitive_patterns):
            print(f"     {i+1}. {pattern['name']}")
        
        return self.cognitive_patterns
    
    def prepare_activations_for_selfie(self, 
                                     cache_key: str, 
                                     layer: int = 17,
                                     max_patterns: int = None) -> Dict[str, Any]:
        """
        Prepare activations from cache for SelfIE interpretation.
        
        Args:
            cache_key: Cache key to extract activations from
            layer: Layer number to analyze
            max_patterns: Maximum number of patterns to process
            
        Returns:
            dict: Prepared data for SelfIE
        """
        print(f"\n🔧 Preparing activations for SelfIE...")
        print(f"   Cache key: {cache_key}")
        print(f"   Target layer: {layer}")
        
        if cache_key not in self.activations_cache:
            raise KeyError(f"Cache key '{cache_key}' not found in loaded data")
        
        cache_data = self.activations_cache[cache_key]
        activations = cache_data['activations']
        
        # Find the layer activation tensor - handle different naming formats
        layer_key = None
        possible_patterns = [
            f'layer_{layer}',
            f'negative_layer_{layer}',
            f'positive_layer_{layer}', 
            f'transition_layer_{layer}',
            f'test_last10_layer_{layer}',
            f'test_multi_layer_{layer}',
            f'cognitive_patterns_depressed_layer_{layer}'
        ]
        
        for pattern in possible_patterns:
            for key in activations.keys():
                if pattern in key:
                    layer_key = key
                    break
            if layer_key:
                break
        
        if not layer_key:
            # List available keys for debugging
            available_keys = list(activations.keys())
            raise KeyError(f"No activations found for layer {layer}. Available keys: {available_keys}")
        
        activation_tensor = activations[layer_key]
        print(f"   Found activations: {layer_key} with shape {activation_tensor.shape}")
        
        # Get associated metadata
        enriched_metadata = cache_data.get('enriched_metadata', [])
        
        # Limit patterns if specified
        if max_patterns:
            enriched_metadata = enriched_metadata[:max_patterns]
            activation_tensor = activation_tensor[:max_patterns]
            print(f"   Limited to {max_patterns} patterns")
        
        # Extract text examples for each pattern
        pattern_texts = []
        pattern_names = []
        
        for i, metadata in enumerate(enriched_metadata):
            name = metadata.get('cognitive_pattern_name', f'Pattern_{i}')
            
            # Extract text - try multiple fields
            text = None
            if 'bad_good_narratives_match' in metadata:
                text = metadata['bad_good_narratives_match'].get('original_thought_pattern')
            elif 'states' in metadata and 'depressed' in metadata['states']:
                text = metadata['states']['depressed']
            elif 'text' in metadata:
                text = metadata['text']
            
            if text:
                pattern_texts.append(text)
                pattern_names.append(name)
        
        return {
            'activation_tensor': activation_tensor,
            'layer': layer,
            'layer_key': layer_key,
            'pattern_names': pattern_names,
            'pattern_texts': pattern_texts,
            'enriched_metadata': enriched_metadata,
            'cache_key': cache_key
        }
    
    def interpret_activations_with_selfie(self, 
                                        prepared_data: Dict[str, Any],
                                        interpretation_template: str = None,
                                        position: Union[int, str] = -1,
                                        batch_size: int = 4,
                                        max_new_tokens: int = 30) -> List[Dict[str, Any]]:
        """
        Generate SelfIE interpretations for prepared activations.
        
        Args:
            prepared_data: Data prepared by prepare_activations_for_selfie()
            interpretation_template: Custom interpretation prompt template
            position: Token position to interpret (-1 for last token, 'mean' for average)
            batch_size: Batch size for processing
            max_new_tokens: Maximum tokens to generate per interpretation
            
        Returns:
            list: Interpretation results
        """
        if not SELFIE_AVAILABLE:
            raise ImportError("SelfIE not available. Please install it first.")
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        print(f"\n🔮 Generating SelfIE interpretations...")
        
        # Default interpretation template for Gemma
        if interpretation_template is None:
            interpretation_template = (
                "<bos>", 0, 0, 0, 0, 0, 
                "\n\nThis neural activation pattern represents:"
            )
        
        # Create interpretation prompt
        interpretation_prompt = InterpretationPrompt(self.tokenizer, interpretation_template)
        
        # Extract vectors from activations
        activation_tensor = prepared_data['activation_tensor']
        pattern_names = prepared_data['pattern_names']
        pattern_texts = prepared_data['pattern_texts']
        
        print(f"   Processing {len(pattern_names)} patterns")
        print(f"   Activation tensor shape: {activation_tensor.shape}")
        
        # Handle position selection
        if isinstance(position, int):
            if activation_tensor.dim() == 3:  # [batch, seq_len, hidden_size]
                vectors = activation_tensor[:, position, :]  # [batch, hidden_size]
            else:  # [batch, hidden_size]
                vectors = activation_tensor
        elif position == 'mean' and activation_tensor.dim() == 3:
            vectors = activation_tensor.mean(dim=1)  # [batch, hidden_size]
        else:
            vectors = activation_tensor
        
        print(f"   Selected vectors shape: {vectors.shape}")
        
        # Ensure vectors have correct shape for SelfIE
        if vectors.dim() == 2:
            # Add dimension for SelfIE: [batch, 1, hidden_size]
            vectors = vectors.unsqueeze(1)
        
        # Convert to list of tensors for SelfIE
        vector_list = [vectors[i:i+1] for i in range(vectors.shape[0])]
        
        # Generate interpretations
        print(f"   Generating interpretations with SelfIE...")
        
        try:
            interpretations = interpret_vectors(
                vecs=vector_list,
                model=self.model,
                interpretation_prompt=interpretation_prompt,
                tokenizer=self.tokenizer,
                bs=batch_size,
                max_new_tokens=max_new_tokens
            )
            
            print(f"   ✅ Generated {len(interpretations)} interpretations")
            
        except Exception as e:
            print(f"   ❌ SelfIE interpretation failed: {e}")
            # Fallback: return empty interpretations
            interpretations = ["Failed to interpret"] * len(pattern_names)
        
        # Package results
        results = []
        for i, (name, text, interpretation) in enumerate(zip(pattern_names, pattern_texts, interpretations)):
            results.append({
                'pattern_name': name,
                'pattern_text': text[:200] + "..." if len(text) > 200 else text,
                'interpretation': interpretation.strip(),
                'layer': prepared_data['layer'],
                'position': position,
                'activation_shape': str(vectors[i].shape),
                'cache_key': prepared_data['cache_key']
            })
        
        return results
    
    def save_interpretation_results(self, results: List[Dict[str, Any]], 
                                  filename: str = None) -> str:
        """
        Save interpretation results to files.
        
        Args:
            results: Results from interpret_activations_with_selfie()
            filename: Custom filename (without extension)
            
        Returns:
            str: Path to saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"selfie_interpretations_{timestamp}"
        
        # Save JSON
        json_path = self.output_dir / f"{filename}.json"
        
        save_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_interpretations': len(results),
                'model_used': "google/gemma-2-2b-it",
                'interpretation_method': 'SelfIE'
            },
            'interpretations': results
        }
        
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Create markdown report
        md_path = self.output_dir / f"{filename}.md"
        self._create_markdown_report(md_path, results)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 JSON: {json_path}")
        print(f"   📋 Report: {md_path}")
        
        return str(json_path)
    
    def _create_markdown_report(self, report_path: Path, results: List[Dict[str, Any]]):
        """Create a markdown report of interpretation results."""
        with open(report_path, 'w') as f:
            f.write("# SelfIE Cognitive Pattern Interpretations\n\n")
            
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Total Patterns:** {len(results)}\n\n")
            f.write(f"**Model:** google/gemma-2-2b-it\n\n")
            f.write("---\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"## {i}. {result['pattern_name']}\n\n")
                f.write(f"**Layer:** {result['layer']}\n\n")
                f.write(f"**Position:** {result['position']}\n\n")
                f.write(f"**Original Text:**\n")
                f.write(f"```\n{result['pattern_text']}\n```\n\n")
                f.write(f"**SelfIE Interpretation:**\n")
                f.write(f"```\n{result['interpretation']}\n```\n\n")
                f.write("---\n\n")
    
    def run_full_analysis(self, 
                         cache_key: str = None, 
                         layer: int = 17,
                         max_patterns: int = 5,
                         position: Union[int, str] = -1) -> str:
        """
        Run complete SelfIE analysis on cached activations.
        
        Args:
            cache_key: Specific cache to analyze (None for first available)
            layer: Layer to analyze
            max_patterns: Maximum patterns to process
            position: Token position to interpret
            
        Returns:
            str: Path to results file
        """
        print("🚀 Running full SelfIE analysis on Gemma activations")
        print("=" * 60)
        
        try:
            # Step 1: Load activations if not already loaded
            if not self.activations_cache:
                self.load_cached_activations()
            
            # Step 2: Load model if not already loaded
            if not self.model:
                if not self.load_model_and_tokenizer():
                    raise RuntimeError("Failed to load model")
            
            # Step 3: Select cache key
            if cache_key is None:
                cache_key = list(self.activations_cache.keys())[0]
                print(f"   Using first available cache: {cache_key}")
            
            # Step 4: Prepare activations
            prepared_data = self.prepare_activations_for_selfie(
                cache_key=cache_key,
                layer=layer,
                max_patterns=max_patterns
            )
            
            # Step 5: Generate interpretations
            results = self.interpret_activations_with_selfie(
                prepared_data=prepared_data,
                position=position,
                batch_size=2,  # Conservative batch size
                max_new_tokens=50
            )
            
            # Step 6: Save results
            results_file = self.save_interpretation_results(results)
            
            # Summary
            print(f"\n📊 Analysis Complete!")
            print(f"   Patterns analyzed: {len(results)}")
            print(f"   Layer analyzed: {layer}")
            print(f"   Results saved to: {results_file}")
            
            # Show sample interpretations
            print(f"\n🔍 Sample interpretations:")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. {result['pattern_name']}:")
                print(f"      \"{result['interpretation'][:60]}...\"")
            
            return results_file
            
        except Exception as e:
            print(f"\n💥 Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function to demonstrate the adapter."""
    print("🎯 Gemma SelfIE Adapter Demo")
    print("=" * 40)
    
    # Initialize adapter
    adapter = GemmaSelfieAdapter()
    
    # Run full analysis
    try:
        results_file = adapter.run_full_analysis(
            layer=17,
            max_patterns=3,  # Start small for demo
            position=-1  # Last token
        )
        print(f"\n✅ Demo completed successfully!")
        print(f"   Check results at: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()