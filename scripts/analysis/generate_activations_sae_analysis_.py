#!/usr/bin/env python3
"""
Generate activations and run SAE analysis for cognitive patterns.

This script:
1. Loads enriched metadata from cached activations to get text examples
2. Generates fresh activations for one example from each cognitive pattern
3. Focuses on 'depressed' state only
4. Runs SAE analysis with Neuronpedia interpretations
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import requests
from datetime import datetime
import sys
import os
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the SAELens directory to Python path
sae_lens_path = Path(__file__).parent / "SAELens"
if str(sae_lens_path) not in sys.path:
    sys.path.insert(0, str(sae_lens_path))

# Import SAELens components
try:
    from sae_lens import SAE
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    SAE_AVAILABLE = True
    print("✅ SAELens imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SAELens: {e}")
    SAE_AVAILABLE = False

# Import local modules
from model_loader import ModelLoader
from activation_capture import ActivationCapturer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuronpediaClient:
    """Client for fetching feature explanations from Neuronpedia API."""
    
    def __init__(self, base_url: str = "https://www.neuronpedia.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Add API key authentication if available
        api_key = os.getenv("NEURONPEDIA_KEY")
        if api_key:
            self.session.headers.update({"x-api-key": api_key})
            print(f"   🔑 Using Neuronpedia API key authentication")
        else:
            print(f"   ⚠️ No Neuronpedia API key found, using unauthenticated requests")
    
    def get_feature_explanation(self, model_id: str, layer: str, feature_idx: int) -> Dict[str, Any]:
        """Fetch individual feature explanation from Neuronpedia API."""
        url = f"{self.base_url}/api/feature/{model_id}/{layer}/{feature_idx}"
        try:
            response = self.fetch_with_retry(url)
            return response
        except Exception as e:
            logger.warning(f"Failed to fetch feature {feature_idx}: {e}")
            return {}
    
    def fetch_with_retry(self, url: str, max_retries: int = 3, delay: float = 1.0) -> Dict[str, Any]:
        """Fetch with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay * (2 ** attempt))


class ActivationSAEAnalyzer:
    """Generate activations and run SAE analysis for cognitive patterns."""
    
    def __init__(self, base_path: str = None, device: str = "auto"):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.device = self._get_device(device)
        self.sae = None
        self.model = None
        self.activation_capturer = None
        self.cognitive_patterns = []
        
        # Create output directory
        self.output_dir = self.base_path / "sae_generated_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Initialized analyzer with device: {self.device}")
        print(f"📁 Output directory: {self.output_dir}")
    
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
    
    def load_cognitive_patterns_from_cache(self) -> List[Dict[str, Any]]:
        """Load cognitive patterns from cached activation data."""
        print(f"\n📊 Loading cognitive patterns from cached data...")
        
        # Load from one of the cached files to get enriched metadata
        cache_files = [
            'activations_8ff00d963316212d.pt',  # negative
            'activations_e5ad16e9b3c33c9b.pt',  # positive  
            'activations_332f24de2a3f82ff.pt'   # transition
        ]
        
        enriched_metadata = None
        for filename in cache_files:
            file_path = self.base_path / "activations" / filename
            if file_path.exists():
                print(f"   Loading metadata from {filename}")
                data = torch.load(file_path, map_location='cpu')
                if 'enriched_metadata' in data:
                    enriched_metadata = data['enriched_metadata']
                    break
        
        if not enriched_metadata:
            raise ValueError("No enriched metadata found in cached files")
        
        # Extract unique cognitive patterns
        unique_patterns = {}
        for entry in enriched_metadata:
            pattern_name = entry.get('cognitive_pattern_name')
            if pattern_name and pattern_name not in unique_patterns:
                unique_patterns[pattern_name] = entry
        
        self.cognitive_patterns = list(unique_patterns.values())
        print(f"✅ Found {len(self.cognitive_patterns)} unique cognitive patterns")
        for i, pattern in enumerate(self.cognitive_patterns):
            print(f"   {i+1}. {pattern.get('cognitive_pattern_name', 'Unknown')}")
        
        return self.cognitive_patterns
    
    def load_model_and_capturer(self) -> bool:
        """Load the model and activation capturer."""
        print(f"\n🔧 Loading model and activation capturer...")
        
        try:
            # Initialize activation capturer (this loads the model)
            self.activation_capturer = ActivationCapturer(
                model_name="google/gemma-2-2b-it",
                device=str(self.device),
                cache_dir=str(self.base_path / "generated_activations")
            )
            
            # Load model from HuggingFace (will be cached automatically)
            print(f"   Loading model from HuggingFace: google/gemma-2-2b-it")
            self.activation_capturer.load_model()
            
            self.model = self.activation_capturer.model
            print(f"✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def load_sae(self) -> bool:
        """Load the SAE for analysis."""
        if not SAE_AVAILABLE:
            print("❌ SAELens not available")
            return False
        
        print(f"\n⚡ Loading SAE...")
        
        # Use the same SAE as in the original script
        release = "gemma-scope-2b-pt-res"
        sae_id = "layer_17/width_65k/average_l0_125"
        
        try:
            self.sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=str(self.device)
            )
            
            print(f"✅ SAE loaded successfully!")
            print(f"   Input dims: {self.sae.cfg.d_in}")
            print(f"   SAE dims: {self.sae.cfg.d_sae}")
            print(f"   Device: {self.sae.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading SAE: {e}")
            return False
    
    def generate_activations_for_patterns(self, target_state: str = "depressed") -> Dict[str, torch.Tensor]:
        """Generate activations for one example from each cognitive pattern."""
        print(f"\n🧠 Generating activations for {target_state} state...")
        
        if not self.activation_capturer:
            raise ValueError("Model not loaded")
        
        # Collect texts for the target state from each pattern
        pattern_texts = []
        pattern_names = []
        
        for pattern in self.cognitive_patterns:
            # Get the depressed state text
            depressed_text = None
            
            # Try different possible field names for depressed state
            if 'states' in pattern:
                depressed_text = pattern['states'].get(target_state)
            elif f'{target_state}_text' in pattern:
                depressed_text = pattern[f'{target_state}_text']
            elif target_state in pattern:
                depressed_text = pattern[target_state]
            elif 'bad_good_narratives_match' in pattern:
                # This seems to be the negative/depressed state
                depressed_text = pattern['bad_good_narratives_match'].get('original_thought_pattern')
            
            if depressed_text:
                pattern_texts.append(depressed_text)
                pattern_names.append(pattern.get('cognitive_pattern_name', 'Unknown'))
                print(f"   Added: {pattern.get('cognitive_pattern_name', 'Unknown')}")
                print(f"     Text: {depressed_text[:60]}...")
        
        if not pattern_texts:
            raise ValueError(f"No texts found for state: {target_state}")
        
        print(f"   Total texts to process: {len(pattern_texts)}")
        
        # Generate activations for all texts at once
        activations = self.activation_capturer.capture_activations(
            strings=pattern_texts,
            layer_nums=[17],  # Focus on layer 17 for SAE analysis
            cognitive_pattern=f"cognitive_patterns_{target_state}",
            position="all",  # Capture all token positions
            use_cache=True
        )
        
        print(f"✅ Generated activations for {len(pattern_texts)} patterns")
        print(f"   Activation keys: {list(activations.keys())}")
        
        # Store pattern names for later reference
        self.pattern_names = pattern_names
        
        return activations
    
    def run_sae_analysis(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run SAE analysis on generated activations."""
        if not self.sae:
            raise ValueError("SAE not loaded")
        
        print(f"\n🔬 Running SAE analysis...")
        
        # Get the layer 17 activations
        layer_17_key = None
        for key in activations.keys():
            if 'layer_17' in key:
                layer_17_key = key
                break
        
        if not layer_17_key:
            raise ValueError("No layer 17 activations found")
        
        acts = activations[layer_17_key]
        print(f"   Processing activations: {acts.shape}")
        
        # Move to SAE device
        acts = acts.to(self.sae.device)
        
        # Set SAE to eval mode
        self.sae.eval()
        
        results = {}
        
        with torch.no_grad():
            # For each pattern (first dimension), analyze across all tokens
            for pattern_idx in range(acts.shape[0]):
                pattern_name = self.pattern_names[pattern_idx] if pattern_idx < len(self.pattern_names) else f"Pattern_{pattern_idx}"
                
                # Get activations for this pattern [seq_len, d_model]
                pattern_acts = acts[pattern_idx]
                
                # Run through SAE
                feature_acts = self.sae.encode(pattern_acts)
                sae_out = self.sae.decode(feature_acts)
                
                # Calculate metrics
                l0_norm = (feature_acts > 0).sum(dim=-1).float()  # [seq_len]
                reconstruction_mse = torch.nn.functional.mse_loss(pattern_acts, sae_out)
                
                # Get top features across all tokens for this pattern
                # Sum activations across sequence length to get overall feature importance
                total_feature_acts = feature_acts.sum(dim=0)  # [n_features]
                top_values, top_indices = torch.topk(total_feature_acts, k=20)
                
                results[pattern_name] = {
                    'feature_activations': feature_acts.cpu(),
                    'avg_l0_norm': l0_norm.mean().item(),
                    'reconstruction_mse': reconstruction_mse.item(),
                    'top_features': {
                        'values': top_values.detach().cpu().numpy().tolist(),
                        'indices': top_indices.detach().cpu().numpy().tolist()
                    },
                    'pattern_text': self.get_pattern_text(pattern_name)
                }
                
                print(f"   {pattern_name}:")
                print(f"     L0 sparsity: {l0_norm.mean().item():.2f}")
                print(f"     Reconstruction MSE: {reconstruction_mse.item():.6f}")
                print(f"     Top feature: {top_indices[0].item()} (total activation: {top_values[0].item():.4f})")
        
        return results
    
    def get_pattern_text(self, pattern_name: str) -> str:
        """Get the text for a given pattern name."""
        for pattern in self.cognitive_patterns:
            if pattern.get('cognitive_pattern_name') == pattern_name:
                # Try to get the depressed/negative text
                if 'bad_good_narratives_match' in pattern:
                    return pattern['bad_good_narratives_match'].get('original_thought_pattern', '')
                elif 'depressed_text' in pattern:
                    return pattern['depressed_text']
                elif 'states' in pattern and 'depressed' in pattern['states']:
                    return pattern['states']['depressed']
        return ""
    
    def fetch_feature_descriptions(self, feature_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch descriptions for a list of feature indices."""
        print(f"\n📖 Fetching descriptions for {len(feature_indices)} features...")
        
        # Use confirmed working API format for Gemma-2-2B layer 17
        client = NeuronpediaClient()
        descriptions = {}
        
        model_id = "gemma-2-2b"
        sae_id = "17-gemmascope-res-65k"
        
        for i, feature_idx in enumerate(feature_indices):
            print(f"   Fetching {i+1}/{len(feature_indices)}: Feature {feature_idx}")
            
            try:
                feature_data = client.get_feature_explanation(model_id, sae_id, feature_idx)
                
                if feature_data:
                    # Extract explanation
                    explanation = ""
                    explanation_score = 0.0
                    
                    if 'explanations' in feature_data and feature_data['explanations']:
                        first_explanation = feature_data['explanations'][0]
                        explanation = first_explanation.get('description', '')
                        scores_array = first_explanation.get('scores', [])
                        explanation_score = scores_array[0] if scores_array else 0.0
                    
                    if not explanation:
                        explanation = feature_data.get('explanation', feature_data.get('description', ''))
                        explanation_score = feature_data.get('explanationScore', 0.0)
                    
                    # Extract tokens
                    pos_tokens = feature_data.get('pos_str', [])
                    neg_tokens = feature_data.get('neg_str', [])
                    
                    description = explanation if explanation else f"Activates on: {', '.join(pos_tokens[:5])}" if pos_tokens else "No description available"
                    
                    descriptions[feature_idx] = {
                        'description': description,
                        'autointerp_explanation': explanation,
                        'autointerp_score': explanation_score,
                        'neuronpedia_url': f"https://neuronpedia.org/{model_id}/{sae_id}/{feature_idx}",
                        'pos_tokens': pos_tokens[:10],
                        'neg_tokens': neg_tokens[:10]
                    }
                    
                    print(f"     ✅ Success: {explanation[:50]}..." if explanation else f"     ✅ Basic data retrieved")
                
                else:
                    descriptions[feature_idx] = {
                        'description': 'No data returned from API',
                        'autointerp_explanation': '',
                        'autointerp_score': 0.0,
                        'neuronpedia_url': f"https://neuronpedia.org/{model_id}/{sae_id}/{feature_idx}"
                    }
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                descriptions[feature_idx] = {
                    'description': f'Failed to fetch: {str(e)}',
                    'autointerp_explanation': '',
                    'autointerp_score': 0.0,
                    'neuronpedia_url': f"https://neuronpedia.org/{model_id}/{sae_id}/{feature_idx}"
                }
            
            # Rate limiting
            time.sleep(0.1)
        
        successful_fetches = sum(1 for desc in descriptions.values() if 'Failed to fetch' not in desc['description'] and 'No data returned' not in desc['description'])
        print(f"✅ Successfully fetched descriptions for {successful_fetches}/{len(feature_indices)} features")
        
        return descriptions
    
    def create_analysis_dataset(self, sae_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured dataset with analysis results and descriptions."""
        print(f"\n📊 Creating analysis dataset...")
        
        # Collect all top features
        all_feature_indices = set()
        for pattern_name, results in sae_results.items():
            for feature_idx in results['top_features']['indices'][:10]:  # Top 10 per pattern
                all_feature_indices.add(feature_idx)
        
        print(f"   Found {len(all_feature_indices)} unique features across all patterns")
        
        # Fetch descriptions
        descriptions = self.fetch_feature_descriptions(list(all_feature_indices))
        
        # Create dataset
        dataset = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_patterns': len(sae_results),
                'total_unique_features': len(all_feature_indices),
                'sae_config': {
                    'release': 'gemma-scope-2b-pt-res',
                    'sae_id': 'layer_17/width_65k/average_l0_125',
                    'layer': 17
                },
                'target_state': 'depressed'
            },
            'patterns': {}
        }
        
        # Process each pattern
        for pattern_name, results in sae_results.items():
            pattern_features = []
            
            for i, (feature_idx, activation) in enumerate(zip(
                results['top_features']['indices'][:10],
                results['top_features']['values'][:10]
            )):
                feature_info = {
                    'rank': i + 1,
                    'feature_index': feature_idx,
                    'total_activation': activation,
                    'description': descriptions.get(feature_idx, {}).get('description', 'No description'),
                    'autointerp_explanation': descriptions.get(feature_idx, {}).get('autointerp_explanation', ''),
                    'autointerp_score': descriptions.get(feature_idx, {}).get('autointerp_score', 0.0),
                    'neuronpedia_url': descriptions.get(feature_idx, {}).get('neuronpedia_url', ''),
                    'pos_tokens': descriptions.get(feature_idx, {}).get('pos_tokens', []),
                    'neg_tokens': descriptions.get(feature_idx, {}).get('neg_tokens', [])
                }
                pattern_features.append(feature_info)
            
            dataset['patterns'][pattern_name] = {
                'avg_l0_norm': results['avg_l0_norm'],
                'reconstruction_mse': results['reconstruction_mse'],
                'pattern_text': results['pattern_text'],
                'top_features': pattern_features
            }
        
        return dataset
    
    def save_results(self, dataset: Dict[str, Any]) -> str:
        """Save analysis results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON dataset
        json_file = self.output_dir / f"cognitive_patterns_sae_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Create summary report
        report_file = self.output_dir / f"cognitive_patterns_report_{timestamp}.md"
        self._create_report(report_file, dataset)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Dataset: {json_file}")
        print(f"   📋 Report: {report_file}")
        
        return str(json_file)
    
    def _create_report(self, report_file: Path, dataset: Dict[str, Any]) -> None:
        """Create a markdown report of the results."""
        with open(report_file, 'w') as f:
            f.write("# Cognitive Patterns SAE Analysis Report\\n\\n")
            
            # Metadata
            metadata = dataset['analysis_metadata']
            f.write("## Analysis Overview\\n")
            f.write(f"**Generated:** {metadata['timestamp']}\\n\\n")
            f.write(f"**Target State:** {metadata['target_state']}\\n\\n")
            f.write(f"**Patterns Analyzed:** {metadata['total_patterns']}\\n\\n")
            f.write(f"**Unique SAE Features:** {metadata['total_unique_features']}\\n\\n")
            f.write(f"**SAE:** {metadata['sae_config']['release']}/{metadata['sae_config']['sae_id']}\\n\\n")
            
            # Pattern results
            f.write("## Pattern Analysis Results\\n\\n")
            
            for pattern_name, pattern_data in dataset['patterns'].items():
                f.write(f"### {pattern_name}\\n\\n")
                f.write(f"**L0 Sparsity:** {pattern_data['avg_l0_norm']:.2f}\\n\\n")
                f.write(f"**Reconstruction MSE:** {pattern_data['reconstruction_mse']:.6f}\\n\\n")
                f.write(f"**Text Example:** {pattern_data['pattern_text'][:200]}...\\n\\n")
                
                f.write("**Top SAE Features:**\\n\\n")
                for feature in pattern_data['top_features'][:5]:  # Top 5
                    f.write(f"{feature['rank']}. **Feature {feature['feature_index']}** (activation: {feature['total_activation']:.4f})\\n")
                    f.write(f"   - *Description:* {feature['description']}\\n")
                    if feature['autointerp_explanation'] and feature['autointerp_explanation'] != feature['description']:
                        f.write(f"   - *AutoInterp:* {feature['autointerp_explanation']} (score: {feature['autointerp_score']:.2f})\\n")
                    f.write(f"   - *Dashboard:* [{feature['neuronpedia_url']}]({feature['neuronpedia_url']})\\n")
                    if feature['pos_tokens']:
                        f.write(f"   - *Activates on:* {', '.join(feature['pos_tokens'][:5])}\\n")
                    f.write("\\n")
                
                f.write("---\\n\\n")
            
            f.write(f"*Generated on {metadata['timestamp']}*\\n")


def main():
    """Main function to run the analysis."""
    print("🚀 Starting Cognitive Patterns SAE Analysis with Generated Activations")
    print("="*70)
    
    analyzer = ActivationSAEAnalyzer()
    
    try:
        # Step 1: Load cognitive patterns from cached data
        cognitive_patterns = analyzer.load_cognitive_patterns_from_cache()
        
        # Step 2: Load model and activation capturer
        if not analyzer.load_model_and_capturer():
            return
        
        # Step 3: Load SAE
        if not analyzer.load_sae():
            return
        
        # Step 4: Generate activations for depressed state
        activations = analyzer.generate_activations_for_patterns(target_state="depressed")
        
        # Step 5: Run SAE analysis
        sae_results = analyzer.run_sae_analysis(activations)
        
        # Step 6: Create analysis dataset with Neuronpedia descriptions
        dataset = analyzer.create_analysis_dataset(sae_results)
        
        # Step 7: Save results
        results_file = analyzer.save_results(dataset)
        
        # Summary
        print(f"\\n📊 Analysis Summary:")
        print(f"   Patterns analyzed: {len(sae_results)}")
        print(f"   Total unique features: {dataset['analysis_metadata']['total_unique_features']}")
        print(f"   Results file: {results_file}")
        
        # Show top features for each pattern
        for pattern_name, pattern_data in dataset['patterns'].items():
            print(f"\\n   {pattern_name}:")
            top_feature = pattern_data['top_features'][0]
            print(f"     Top feature: {top_feature['feature_index']} (activation: {top_feature['total_activation']:.4f})")
            print(f"     Description: {top_feature['description'][:60]}...")
        
        print(f"\\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\\n💥 Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()