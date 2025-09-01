#!/usr/bin/env python3
"""
Simple test script to run a single cognitive pattern through SAE analysis with Neuronpedia interpretation.

This script:
1. Takes one cognitive pattern from the enriched metadata
2. Uses its negative, positive, and transition activations  
3. Runs them through SAE using SAELens
4. Interprets the results with Neuronpedia integration
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

class SinglePatternSAETest:
    """Test class for running a single cognitive pattern through SAE analysis."""
    
    def __init__(self, base_path: str = None, device: str = "auto"):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.device = self._get_device(device)
        self.sae = None
        self.model = None
        self.pattern_data = None
        self.activations = None
        
        # Create output directory
        self.output_dir = self.base_path / "sae_test_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Initialized SAE test with device: {self.device}")
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
    
    def load_pattern_data(self, pattern_index: int = 0) -> Dict[str, Any]:
        """Load data for a specific cognitive pattern."""
        print(f"\n📊 Loading pattern data (index: {pattern_index})...")
        
        # Load enriched metadata
        metadata_path = self.base_path / "data" / "final" / "enriched_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if pattern_index >= len(metadata):
            raise IndexError(f"Pattern index {pattern_index} out of range (max: {len(metadata) - 1})")
        
        self.pattern_data = metadata[pattern_index]
        pattern_name = self.pattern_data['bad_good_narratives_match']['cognitive_pattern_name_from_bad_good']
        
        print(f"✅ Loaded pattern: {pattern_name}")
        print(f"   Type: {self.pattern_data['cognitive_pattern_type']}")
        print(f"   Description: {self.pattern_data['pattern_description']}")
        
        return self.pattern_data
    
    def generate_single_pattern_activations(self) -> Dict[str, torch.Tensor]:
        """Generate activations for the current pattern using the activation capturer."""
        if not self.pattern_data:
            raise ValueError("No pattern data loaded. Call load_pattern_data() first.")
        
        print(f"\n🧠 Generating activations for pattern...")
        
        # Load model
        model_loader = ModelLoader("google/gemma-2-2b-it", device=str(self.device))
        self.model = model_loader.load_model()
        
        # Initialize activation capturer
        capturer = ActivationCapturer("google/gemma-2-2b-it", device=str(self.device))
        capturer.model = self.model
        
        # Extract the different thought patterns
        negative_text = self.pattern_data['bad_good_narratives_match']['original_thought_pattern']
        positive_text = self.pattern_data['positive_thought_pattern']
        transition_text = self.pattern_data['bad_good_narratives_match']['transformed_thought_pattern']
        
        print(f"   Negative: {negative_text[:100]}...")
        print(f"   Positive: {positive_text[:100]}...")
        print(f"   Transition: {transition_text[:100]}...")
        
        # Capture activations for each state
        layer_nums = [17, 21]  # Focus on these layers based on existing code
        
        activations = {}
        
        # Negative activations
        neg_acts = capturer.capture_activations(
            strings=[negative_text],
            layer_nums=layer_nums,
            cognitive_pattern="negative",
            position="last"
        )
        
        # Positive activations  
        pos_acts = capturer.capture_activations(
            strings=[positive_text],
            layer_nums=layer_nums,
            cognitive_pattern="positive", 
            position="last"
        )
        
        # Transition activations
        trans_acts = capturer.capture_activations(
            strings=[transition_text],
            layer_nums=layer_nums,
            cognitive_pattern="transition",
            position="last"
        )
        
        # Organize activations
        for layer in layer_nums:
            activations[f'negative_layer_{layer}'] = neg_acts[f'negative_layer_{layer}']
            activations[f'positive_layer_{layer}'] = pos_acts[f'positive_layer_{layer}']
            activations[f'transition_layer_{layer}'] = trans_acts[f'transition_layer_{layer}']
        
        self.activations = activations
        print(f"✅ Generated activations for {len(layer_nums)} layers")
        
        return activations
    
    def discover_available_saes(self) -> List[Dict[str, Any]]:
        """Discover available SAEs for the model."""
        if not SAE_AVAILABLE:
            print("❌ SAELens not available")
            return []
        
        print(f"\n🔍 Discovering available SAEs...")
        
        try:
            # Get all available SAEs
            saes_directory = get_pretrained_saes_directory()
            
            # Filter for Gemma-2 SAEs
            gemma_saes = []
            for release, sae_info in saes_directory.items():
                if "gemma" in sae_info.model.lower() and "2" in sae_info.model:
                    for sae_id in sae_info.saes_map.keys():
                        gemma_saes.append({
                            "release": release,
                            "sae_id": sae_id,
                            "model": sae_info.model,
                            "repo_id": sae_info.repo_id,
                            "path": sae_info.saes_map[sae_id],
                            "var_explained": sae_info.expected_var_explained.get(sae_id, "unknown"),
                            "l0": sae_info.expected_l0.get(sae_id, "unknown"),
                            "neuronpedia_id": sae_info.neuronpedia_id.get(sae_id, "unknown")
                        })
            
            print(f"✅ Found {len(gemma_saes)} Gemma-2 SAEs")
            for i, sae in enumerate(gemma_saes[:5]):  # Show first 5
                print(f"   {i+1}. {sae['release']}/{sae['sae_id']} - {sae['model']}")
            
            return gemma_saes
            
        except Exception as e:
            print(f"❌ Error discovering SAEs: {e}")
            return []
    
    def load_sae(self, release: str = None, sae_id: str = None) -> bool:
        """Load a specific SAE."""
        if not SAE_AVAILABLE:
            print("❌ SAELens not available")
            return False
        
        # Default to a known working SAE for Gemma-2-2B
        if not release or not sae_id:
            # Try to find a suitable default
            available_saes = self.discover_available_saes()
            if available_saes:
                sae_info = available_saes[0]  # Use the first available
                release = sae_info['release']
                sae_id = sae_info['sae_id']
                print(f"🎯 Using default SAE: {release}/{sae_id}")
            else:
                print("❌ No suitable SAEs found")
                return False
        
        print(f"\n⚡ Loading SAE: {release}/{sae_id}...")
        
        try:
            self.sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=str(self.device)
            )
            
            print(f"✅ SAE loaded successfully!")
            print(f"   Hook point: {self.sae.cfg.hook_name}")
            print(f"   Input dims: {self.sae.cfg.d_in}")
            print(f"   SAE dims: {self.sae.cfg.d_sae}")
            print(f"   Device: {self.sae.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading SAE: {e}")
            return False
    
    def run_sae_analysis(self, layer: int = 17) -> Dict[str, Any]:
        """Run SAE analysis on the activations."""
        if not self.sae:
            raise ValueError("SAE not loaded. Call load_sae() first.")
        
        if not self.activations:
            raise ValueError("No activations available. Call generate_single_pattern_activations() first.")
        
        print(f"\n🔬 Running SAE analysis on layer {layer}...")
        
        # Get activations for the specified layer
        neg_acts = self.activations[f'negative_layer_{layer}']
        pos_acts = self.activations[f'positive_layer_{layer}']
        trans_acts = self.activations[f'transition_layer_{layer}']
        
        print(f"   Negative shape: {neg_acts.shape}")
        print(f"   Positive shape: {pos_acts.shape}")
        print(f"   Transition shape: {trans_acts.shape}")
        
        # Ensure activations are on the same device as SAE
        neg_acts = neg_acts.to(self.sae.device)
        pos_acts = pos_acts.to(self.sae.device)
        trans_acts = trans_acts.to(self.sae.device)
        
        results = {}
        
        # Process each activation type through SAE
        for act_type, activations in [("negative", neg_acts), ("positive", pos_acts), ("transition", trans_acts)]:
            print(f"   Processing {act_type} activations...")
            
            # Get feature activations
            feature_acts = self.sae.encode(activations)
            
            # Get reconstructions
            reconstructions = self.sae.decode(feature_acts)
            
            # Calculate metrics
            reconstruction_error = torch.nn.functional.mse_loss(activations, reconstructions, reduction='none')
            l0_norm = (feature_acts > 0).sum(dim=-1).float()
            
            # Find top features
            if len(feature_acts.shape) == 2:  # [batch, features]
                feature_vector = feature_acts[0, :]
            else:  # Handle other shapes
                feature_vector = feature_acts.flatten()
            
            top_k = 10
            values, indices = torch.topk(feature_vector, k=top_k)
            
            results[act_type] = {
                'original_activations': activations.cpu(),
                'feature_activations': feature_acts.cpu(),
                'reconstructions': reconstructions.cpu(),
                'reconstruction_error': reconstruction_error.cpu(),
                'l0_norm': l0_norm.cpu(),
                'top_features': {
                    'values': values.cpu(),
                    'indices': indices.cpu()
                }
            }
            
            print(f"     L0 sparsity: {l0_norm.mean().item():.2f}")
            print(f"     Reconstruction MSE: {reconstruction_error.mean().item():.6f}")
            print(f"     Top feature: {indices[0].item()} (activation: {values[0].item():.4f})")
        
        return results
    
    def analyze_feature_differences(self, sae_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze differences between cognitive states in feature space."""
        print(f"\n🔍 Analyzing feature differences...")
        
        # Get feature activations for each state
        neg_features = sae_results['negative']['feature_activations'][0]  # Remove batch dim
        pos_features = sae_results['positive']['feature_activations'][0]
        trans_features = sae_results['transition']['feature_activations'][0]
        
        # Calculate differences
        neg_to_pos_diff = pos_features - neg_features
        neg_to_trans_diff = trans_features - neg_features
        trans_to_pos_diff = pos_features - trans_features
        
        # Find top differential features
        def get_top_diff_features(diff_tensor, name, k=10):
            abs_diff = torch.abs(diff_tensor)
            values, indices = torch.topk(abs_diff, k=k)
            directions = torch.sign(diff_tensor[indices])
            
            print(f"   Top {k} differential features ({name}):")
            feature_info = []
            for i, (val, idx, direction) in enumerate(zip(values, indices, directions)):
                direction_str = "↑" if direction > 0 else "↓"
                print(f"     {i+1}. Feature {idx.item()}: {direction_str} {val.item():.4f}")
                feature_info.append({
                    'rank': i+1,
                    'feature_idx': idx.item(),
                    'abs_diff': val.item(),
                    'direction': direction_str,
                    'raw_diff': diff_tensor[idx].item()
                })
            
            return feature_info
        
        analysis = {
            'negative_to_positive': get_top_diff_features(neg_to_pos_diff, "Negative → Positive"),
            'negative_to_transition': get_top_diff_features(neg_to_trans_diff, "Negative → Transition"), 
            'transition_to_positive': get_top_diff_features(trans_to_pos_diff, "Transition → Positive")
        }
        
        return analysis
    
    def generate_neuronpedia_links(self, feature_indices: List[int], release: str, sae_id: str) -> List[str]:
        """Generate Neuronpedia URLs for specific features."""
        base_url = "https://neuronpedia.org"
        
        # Convert release and sae_id to Neuronpedia format
        # This might need adjustment based on the specific SAE used
        model_id = release.replace("-", "_")
        layer_id = sae_id.replace("blocks.", "").replace(".hook_resid_pre", "")
        
        urls = []
        for feature_idx in feature_indices:
            url = f"{base_url}/{model_id}/{layer_id}/{feature_idx}"
            urls.append(url)
        
        return urls
    
    def create_neuronpedia_dashboard(self, differential_analysis: Dict[str, Any], 
                                   release: str, sae_id: str) -> Dict[str, List[str]]:
        """Create Neuronpedia dashboard URLs for top differential features."""
        print(f"\n🌐 Generating Neuronpedia links...")
        
        dashboard_urls = {}
        
        for transition_type, features in differential_analysis.items():
            print(f"   {transition_type}:")
            feature_indices = [f['feature_idx'] for f in features[:5]]  # Top 5 features
            urls = self.generate_neuronpedia_links(feature_indices, release, sae_id)
            
            dashboard_urls[transition_type] = []
            for i, (feature_info, url) in enumerate(zip(features[:5], urls)):
                dashboard_urls[transition_type].append({
                    'feature_idx': feature_info['feature_idx'],
                    'direction': feature_info['direction'],
                    'diff': feature_info['abs_diff'],
                    'url': url
                })
                print(f"     Feature {feature_info['feature_idx']}: {url}")
        
        return dashboard_urls
    
    def save_results(self, sae_results: Dict[str, Any], differential_analysis: Dict[str, Any],
                    neuronpedia_urls: Dict[str, List[str]], release: str, sae_id: str) -> str:
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern_name = self.pattern_data['bad_good_narratives_match']['cognitive_pattern_name_from_bad_good']
        safe_pattern_name = "".join(c for c in pattern_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_pattern_name = safe_pattern_name.replace(' ', '_')
        
        # Save comprehensive results
        results_data = {
            'pattern_info': {
                'name': pattern_name,
                'type': self.pattern_data['cognitive_pattern_type'],
                'description': self.pattern_data['pattern_description'],
                'negative_text': self.pattern_data['bad_good_narratives_match']['original_thought_pattern'],
                'positive_text': self.pattern_data['positive_thought_pattern'],
                'transition_text': self.pattern_data['bad_good_narratives_match']['transformed_thought_pattern']
            },
            'sae_info': {
                'release': release,
                'sae_id': sae_id,
                'device': str(self.device)
            },
            'sae_results': {k: {
                'l0_norm': float(v['l0_norm'].mean()),
                'reconstruction_mse': float(v['reconstruction_error'].mean()),
                'top_features': {
                    'indices': v['top_features']['indices'].tolist(),
                    'values': v['top_features']['values'].tolist()
                }
            } for k, v in sae_results.items()},
            'differential_analysis': differential_analysis,
            'neuronpedia_urls': neuronpedia_urls,
            'timestamp': timestamp
        }
        
        # Save to JSON
        results_file = self.output_dir / f"sae_test_results_{safe_pattern_name}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save raw tensors
        tensor_file = self.output_dir / f"sae_test_tensors_{safe_pattern_name}_{timestamp}.pt"
        torch.save({
            'sae_results': sae_results,
            'activations': self.activations
        }, tensor_file)
        
        # Create summary report
        report_file = self.output_dir / f"sae_test_report_{safe_pattern_name}_{timestamp}.md"
        self._create_report(report_file, results_data)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Results: {results_file}")
        print(f"   🔢 Tensors: {tensor_file}")
        print(f"   📋 Report: {report_file}")
        
        return str(report_file)
    
    def _create_report(self, report_file: Path, results_data: Dict[str, Any]) -> None:
        """Create a markdown report of the results."""
        with open(report_file, 'w') as f:
            f.write("# SAE Analysis Report - Single Pattern Test\n\n")
            
            # Pattern info
            f.write("## Cognitive Pattern\n")
            f.write(f"**Name:** {results_data['pattern_info']['name']}\n\n")
            f.write(f"**Type:** {results_data['pattern_info']['type']}\n\n")
            f.write(f"**Description:** {results_data['pattern_info']['description']}\n\n")
            
            # Texts
            f.write("### Text Examples\n")
            f.write(f"**Negative:** {results_data['pattern_info']['negative_text'][:200]}...\n\n")
            f.write(f"**Positive:** {results_data['pattern_info']['positive_text'][:200]}...\n\n")
            f.write(f"**Transition:** {results_data['pattern_info']['transition_text'][:200]}...\n\n")
            
            # SAE info
            f.write("## SAE Configuration\n")
            f.write(f"**Release:** {results_data['sae_info']['release']}\n\n")
            f.write(f"**SAE ID:** {results_data['sae_info']['sae_id']}\n\n")
            f.write(f"**Device:** {results_data['sae_info']['device']}\n\n")
            
            # Results summary
            f.write("## SAE Results Summary\n")
            for state, metrics in results_data['sae_results'].items():
                f.write(f"### {state.title()} State\n")
                f.write(f"- **L0 Sparsity:** {metrics['l0_norm']:.2f}\n")
                f.write(f"- **Reconstruction MSE:** {metrics['reconstruction_mse']:.6f}\n")
                f.write(f"- **Top Feature:** {metrics['top_features']['indices'][0]} (activation: {metrics['top_features']['values'][0]:.4f})\n\n")
            
            # Differential analysis
            f.write("## Feature Differences Analysis\n")
            for transition, features in results_data['differential_analysis'].items():
                f.write(f"### {transition.replace('_', ' ').title()}\n")
                for feature in features[:5]:
                    f.write(f"- Feature {feature['feature_idx']}: {feature['direction']} {feature['abs_diff']:.4f}\n")
                f.write("\n")
            
            # Neuronpedia links
            f.write("## Neuronpedia Dashboard Links\n")
            for transition, url_data in results_data['neuronpedia_urls'].items():
                f.write(f"### {transition.replace('_', ' ').title()}\n")
                for data in url_data:
                    f.write(f"- [Feature {data['feature_idx']} ({data['direction']} {data['diff']:.4f})]({data['url']})\n")
                f.write("\n")
            
            f.write(f"---\n*Generated on {results_data['timestamp']}*\n")


def main():
    """Main function to run the SAE test."""
    print("🚀 Starting Single Pattern SAE Test")
    print("="*50)
    
    # Initialize test
    test = SinglePatternSAETest()
    
    try:
        # Step 1: Load pattern data
        pattern_data = test.load_pattern_data(pattern_index=0)  # Use first pattern
        
        # Step 2: Generate activations
        activations = test.generate_single_pattern_activations()
        
        # Step 3: Discover and load SAE
        available_saes = test.discover_available_saes()
        if not available_saes:
            print("❌ No suitable SAEs found. Exiting.")
            return
        
        # Load the first available SAE
        sae_info = available_saes[0]
        success = test.load_sae(sae_info['release'], sae_info['sae_id'])
        if not success:
            print("❌ Failed to load SAE. Exiting.")
            return
        
        # Step 4: Run SAE analysis
        sae_results = test.run_sae_analysis(layer=17)
        
        # Step 5: Analyze feature differences
        differential_analysis = test.analyze_feature_differences(sae_results)
        
        # Step 6: Generate Neuronpedia links
        neuronpedia_urls = test.create_neuronpedia_dashboard(
            differential_analysis, sae_info['release'], sae_info['sae_id']
        )
        
        # Step 7: Save results
        report_path = test.save_results(
            sae_results, differential_analysis, neuronpedia_urls,
            sae_info['release'], sae_info['sae_id']
        )
        
        print(f"\n🎉 SAE analysis completed successfully!")
        print(f"📋 Full report available at: {report_path}")
        print(f"\n🔗 Key Neuronpedia links:")
        for transition, url_data in neuronpedia_urls.items():
            if url_data:  # Check if not empty
                print(f"   {transition}: {url_data[0]['url']}")
        
    except Exception as e:
        print(f"\n💥 Error during SAE test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
