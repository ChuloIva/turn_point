#!/usr/bin/env python3
"""
Cluster-Filtered SAE Analysis

This script:
1. Loads cluster analysis results from the UMAP clustering
2. Extracts the highest activating cluster activations 
3. Runs them through SAE analysis using SAELens
4. Interprets the results with Neuronpedia integration

Key difference from test_single_pattern_sae.py:
- Uses only the activations from the highest activating clusters
- Filters activations based on cluster membership
- Focuses analysis on the most significant neural patterns
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

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

class ClusterFilteredSAEAnalysis:
    """SAE analysis using cluster-filtered activations from UMAP clustering results."""
    
    def __init__(self, base_path: str = None, device: str = "auto"):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.device = self._get_device(device)
        self.sae = None
        self.model = None
        self.pattern_data = None
        self.filtered_activations = None
        self.cluster_analysis = None
        
        # Create output directory
        self.output_dir = self.base_path / "cluster_filtered_sae_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Initialized Cluster-Filtered SAE Analysis with device: {self.device}")
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
    
    def load_cluster_analysis(self, cluster_analysis_file: str) -> Dict[str, Any]:
        """Load cluster analysis results from pickle file."""
        print(f"\n📊 Loading cluster analysis from {cluster_analysis_file}...")
        
        cluster_file_path = Path(cluster_analysis_file)
        if not cluster_file_path.exists():
            raise FileNotFoundError(f"Cluster analysis file not found: {cluster_analysis_file}")
        
        with open(cluster_file_path, 'rb') as f:
            self.cluster_analysis = pickle.load(f)
        
        print(f"✅ Loaded cluster analysis for pattern: {self.cluster_analysis['analysis_config']['pattern_name']}")
        print(f"   Configuration: {self.cluster_analysis['analysis_config']}")
        
        # Extract pattern info
        self.pattern_name = self.cluster_analysis['analysis_config']['pattern_name']
        self.layer = self.cluster_analysis['analysis_config']['layer']
        self.all_tokens = self.cluster_analysis['analysis_config']['all_tokens']
        
        return self.cluster_analysis
    
    def load_original_activations(self) -> Dict[str, torch.Tensor]:
        """Load the original activation files."""
        print(f"\n📊 Loading original activations...")
        
        # Load all three activation files
        activation_files = {
            'negative': 'activations_8ff00d963316212d.pt',
            'positive': 'activations_e5ad16e9b3c33c9b.pt', 
            'transition': 'activations_332f24de2a3f82ff.pt'
        }
        
        all_activations = {}
        
        for state_name, filename in activation_files.items():
            file_path = self.base_path / "activations" / filename
            if not file_path.exists():
                print(f"   ⚠️ Missing {state_name} file: {filename}")
                continue
                
            print(f"   Loading {state_name} activations from {filename}")
            data = torch.load(file_path, map_location='cpu')
            
            # Store metadata from first file
            if state_name == 'negative':
                self.enriched_metadata = data.get('enriched_metadata', [])
                self.metadata_info = data.get('metadata_info', {})
            
            # Get activation tensors and rename them
            for key, tensor in data.items():
                if key not in ['enriched_metadata', 'metadata_info']:
                    # Only load the specified layer activations
                    if f'layer_{self.layer}' in key:
                        new_key = f"{state_name}_{key.split('_', 1)[1]}"
                        all_activations[new_key] = tensor
        
        print(f"✅ Loaded activations for {len(activation_files)} states")
        print(f"   Available patterns: {len(self.enriched_metadata)}")
        print(f"   Tensor keys: {list(all_activations.keys())}")
        
        return all_activations
    
    def extract_cluster_filtered_activations(self, original_activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract activations corresponding to the highest activating clusters."""
        print(f"\n🎯 Extracting cluster-filtered activations...")
        
        if not self.cluster_analysis:
            raise ValueError("No cluster analysis loaded. Call load_cluster_analysis() first.")
        
        # Get the activation filter from cluster analysis
        activation_filter = self.cluster_analysis['activation_filter']
        
        filtered_activations = {}
        
        for state_name, filter_info in activation_filter.items():
            if filter_info['activation_count'] == 0:
                print(f"   {state_name}: No activations to extract (empty filter)")
                continue
            
            print(f"\n   Processing {state_name} state:")
            print(f"     Total cluster activations to extract: {filter_info['activation_count']}")
            print(f"     From {len(filter_info['sample_ids'])} unique samples")
            
            # Get the original activation tensor for this state
            state_key = f"{state_name.lower()}_layer_{self.layer}"
            if state_key not in original_activations:
                print(f"     ⚠️ Missing original activations for {state_key}")
                continue
            
            original_tensor = original_activations[state_key]
            print(f"     Original tensor shape: {original_tensor.shape}")
            
            # Extract the specific activations based on sample_token_pairs
            sample_token_pairs = filter_info['sample_token_pairs']
            extracted_activations = []
            
            for sample_id, token_position in sample_token_pairs:
                if sample_id >= original_tensor.shape[0]:
                    print(f"     ⚠️ Sample {sample_id} out of range (max: {original_tensor.shape[0]-1})")
                    continue
                
                if token_position == -1:
                    # Last token
                    activation = original_tensor[sample_id, -1, :]
                else:
                    # Specific token position
                    if token_position >= original_tensor.shape[1]:
                        print(f"     ⚠️ Token {token_position} out of range for sample {sample_id}")
                        continue
                    activation = original_tensor[sample_id, token_position, :]
                
                extracted_activations.append(activation)
            
            if extracted_activations:
                # Stack all extracted activations
                filtered_tensor = torch.stack(extracted_activations)
                filtered_activations[state_key] = filtered_tensor
                print(f"     Extracted tensor shape: {filtered_tensor.shape}")
                print(f"     Successfully extracted {len(extracted_activations)} activations")
            else:
                print(f"     ❌ No valid activations extracted for {state_name}")
        
        print(f"\n✅ Cluster filtering complete. States with filtered activations: {list(filtered_activations.keys())}")
        self.filtered_activations = filtered_activations
        return filtered_activations
    
    def load_neuronpedia_sae(self) -> bool:
        """Load the specific SAE from HuggingFace."""
        if not SAE_AVAILABLE:
            print("❌ SAELens not available")
            return False
        
        # The specific SAE from HuggingFace URL
        release = "gemma-scope-2b-pt-res"
        sae_id = "layer_17/width_65k/average_l0_125"
        
        print(f"\n⚡ Loading HuggingFace SAE: {release}/{sae_id}...")
        
        try:
            self.sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=str(self.device)
            )
            
            print(f"✅ SAE loaded successfully!")
            print(f"   Hook point: {getattr(self.sae.cfg, 'hook_name', 'Unknown')}")
            print(f"   Input dims: {self.sae.cfg.d_in}")
            print(f"   SAE dims: {self.sae.cfg.d_sae}")
            print(f"   Device: {self.sae.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading SAE: {e}")
            return False
    
    def run_sae_analysis_on_filtered_activations(self) -> Dict[str, Any]:
        """Run SAE analysis on the cluster-filtered activations."""
        if not self.sae:
            raise ValueError("SAE not loaded. Call load_neuronpedia_sae() first.")
        
        if not self.filtered_activations:
            raise ValueError("No filtered activations available. Call extract_cluster_filtered_activations() first.")
        
        print(f"\n🔬 Running SAE analysis on cluster-filtered activations...")
        
        results = {}
        
        # Set SAE to eval mode
        self.sae.eval()
        
        with torch.no_grad():
            # Process each filtered activation type
            for key in self.filtered_activations.keys():
                state_name = key.split('_layer_')[0]  # Extract 'negative', 'positive', 'transition'
                activations = self.filtered_activations[key]
                
                print(f"\n   Processing {state_name} filtered activations: {activations.shape}")
                
                # Move to SAE device
                activations = activations.to(self.sae.device)
                
                # Run through SAE
                feature_acts = self.sae.encode(activations)
                sae_out = self.sae.decode(feature_acts)
                
                # Calculate metrics
                l0_norm = (feature_acts > 0).sum(dim=-1).float()
                reconstruction_mse = torch.nn.functional.mse_loss(activations, sae_out)
                
                # Top 20 features by average activation
                avg_feature_acts = feature_acts.mean(dim=0)
                top_values, top_indices = torch.topk(avg_feature_acts, k=20)
                
                results[state_name] = {
                    'feature_activations': feature_acts.cpu(),
                    'avg_l0_norm': l0_norm.mean().item(),
                    'reconstruction_mse': reconstruction_mse.item(),
                    'top_features': {
                        'values': top_values.detach().cpu().numpy().tolist(),
                        'indices': top_indices.detach().cpu().numpy().tolist()
                    },
                    'num_filtered_activations': activations.shape[0]
                }
                
                print(f"     L0 sparsity: {l0_norm.mean().item():.2f}")
                print(f"     Reconstruction MSE: {reconstruction_mse.item():.6f}")
                print(f"     Top feature: {top_indices[0].item()} (activation: {top_values[0].item():.4f})")
                print(f"     Analyzed {activations.shape[0]} cluster-filtered activations")
        
        return results
    
    def fetch_feature_descriptions(self, feature_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch descriptions for a list of feature indices."""
        print(f"\n📖 Fetching descriptions for {len(feature_indices)} features...")
        
        # Use the confirmed working API format
        identifier_combinations = [
            ("gemma-2-2b", "17-gemmascope-res-65k"),  # Confirmed working API format for layer 17
        ]
        
        print(f"   Will try identifier combinations: {identifier_combinations}")
        
        # Initialize client
        client = NeuronpediaClient()
        
        descriptions = {}
        
        # Try each feature with different identifier combinations
        for i, feature_idx in enumerate(feature_indices):
            print(f"   Fetching {i+1}/{len(feature_indices)}: Feature {feature_idx}")
            
            feature_fetched = False
            
            # Try each identifier combination until one works
            for model_id, sae_id in identifier_combinations:
                try:
                    print(f"     Trying {model_id}/{sae_id}")
                    feature_data = client.get_feature_explanation(model_id, sae_id, feature_idx)
                    
                    # Check if we got actual data
                    if feature_data:
                        # Extract explanation from explanations array if available
                        explanation = ""
                        explanation_score = 0.0
                        
                        if 'explanations' in feature_data and feature_data['explanations']:
                            first_explanation = feature_data['explanations'][0]
                            explanation = first_explanation.get('description', '')
                            scores_array = first_explanation.get('scores', [])
                            explanation_score = scores_array[0] if scores_array else 0.0
                        
                        # Also check for direct fields
                        if not explanation:
                            explanation = feature_data.get('explanation', feature_data.get('description', ''))
                            explanation_score = feature_data.get('explanationScore', 0.0)
                        
                        # Extract activating tokens
                        pos_tokens = feature_data.get('pos_str', [])
                        neg_tokens = feature_data.get('neg_str', [])
                        
                        # Create description
                        description = explanation if explanation else f"Activates on: {', '.join(pos_tokens[:5])}" if pos_tokens else "No description available"
                        
                        descriptions[feature_idx] = {
                            'description': description,
                            'autointerp_explanation': explanation,
                            'autointerp_score': explanation_score,
                            'neuronpedia_url': f"https://neuronpedia.org/{model_id}/{sae_id}/{feature_idx}",
                            'pos_tokens': pos_tokens[:10],
                            'neg_tokens': neg_tokens[:10]
                        }
                        
                        print(f"     ✅ Success with {model_id}/{sae_id}")
                        if explanation:
                            print(f"     Explanation: {explanation[:50]}...")
                        elif pos_tokens:
                            print(f"     Activates on: {', '.join(pos_tokens[:3])}...")
                        
                        feature_fetched = True
                        break
                    
                except Exception as e:
                    print(f"     Failed {model_id}/{sae_id}: {e}")
                    continue
                
                # Rate limiting between attempts
                time.sleep(0.1)
            
            # If all combinations failed, store a failure record
            if not feature_fetched:
                descriptions[feature_idx] = {
                    'description': 'Failed to fetch from any identifier combination',
                    'autointerp_explanation': '',
                    'autointerp_score': 0.0,
                    'neuronpedia_url': f"https://neuronpedia.org/{identifier_combinations[0][0]}/{identifier_combinations[0][1]}/{feature_idx}"
                }
        
        successful_fetches = sum(1 for desc in descriptions.values() if 'Failed to fetch' not in desc['description'])
        print(f"✅ Successfully fetched descriptions for {successful_fetches}/{len(feature_indices)} features")
        return descriptions
    
    def create_analysis_report(self, sae_results: Dict[str, Any], descriptions: Dict[int, Dict[str, Any]]) -> str:
        """Create comprehensive analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern_name_safe = self.pattern_name.replace(' ', '_').replace('&', 'and')
        
        # Create comprehensive results
        results_data = {
            'analysis_metadata': {
                'pattern_name': self.pattern_name,
                'analysis_type': 'cluster_filtered_sae',
                'layer': self.layer,
                'all_tokens': self.all_tokens,
                'timestamp': timestamp,
                'cluster_filter_info': {
                    state: {
                        'activation_count': info['activation_count'],
                        'unique_samples': len(info['sample_ids'])
                    }
                    for state, info in self.cluster_analysis['activation_filter'].items()
                    if info['activation_count'] > 0
                }
            },
            'cluster_info': {
                'top_clusters_per_state': {
                    state: [
                        {
                            'cluster_id': cluster['cluster_id'],
                            'size': cluster['size'],
                            'activation_magnitude': cluster['activation_magnitude']
                        }
                        for cluster in clusters
                    ]
                    for state, clusters in self.cluster_analysis['top_clusters'].items()
                }
            },
            'sae_results': {},
            'feature_descriptions': descriptions
        }
        
        # Process SAE results
        for state_name, results in sae_results.items():
            state_features = []
            for i, (feature_idx, activation) in enumerate(zip(
                results['top_features']['indices'][:20],
                results['top_features']['values'][:20]
            )):
                desc_data = descriptions.get(feature_idx, {})
                
                state_features.append({
                    'rank': i + 1,
                    'feature_idx': feature_idx,
                    'activation_value': activation,
                    'description': desc_data.get('description', ''),
                    'autointerp_explanation': desc_data.get('autointerp_explanation', ''),
                    'autointerp_score': desc_data.get('autointerp_score', 0.0),
                    'neuronpedia_url': desc_data.get('neuronpedia_url', '')
                })
            
            results_data['sae_results'][state_name] = {
                'l0_sparsity': results['avg_l0_norm'],
                'reconstruction_mse': results['reconstruction_mse'],
                'num_filtered_activations': results['num_filtered_activations'],
                'top_features': state_features
            }
        
        # Save JSON results
        json_file = self.output_dir / f"cluster_filtered_sae_results_{pattern_name_safe}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create markdown report
        report_file = self.output_dir / f"cluster_filtered_sae_report_{pattern_name_safe}_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write("# Cluster-Filtered SAE Analysis Report\n\n")
            
            # Metadata
            f.write("## Analysis Configuration\n")
            f.write(f"**Pattern:** {self.pattern_name}\n\n")
            f.write(f"**Layer:** {self.layer}\n\n")
            f.write(f"**Token Strategy:** {'All tokens' if self.all_tokens else 'Last token only'}\n\n")
            f.write(f"**Analysis Type:** Cluster-filtered (only highest activating clusters)\n\n")
            
            # Cluster filter info
            f.write("## Cluster Filter Summary\n")
            for state_name, filter_info in self.cluster_analysis['activation_filter'].items():
                if filter_info['activation_count'] > 0:
                    f.write(f"**{state_name} State:**\n")
                    f.write(f"- Filtered activations: {filter_info['activation_count']}\n")
                    f.write(f"- Unique samples: {len(filter_info['sample_ids'])}\n\n")
            
            # SAE results
            f.write("## SAE Analysis Results\n")
            for state_name, results in results_data['sae_results'].items():
                f.write(f"### {state_name.title()} State\n")
                f.write(f"- **Filtered Activations Analyzed:** {results['num_filtered_activations']}\n")
                f.write(f"- **L0 Sparsity:** {results['l0_sparsity']:.2f}\n")
                f.write(f"- **Reconstruction MSE:** {results['reconstruction_mse']:.6f}\n\n")
                
                f.write("#### Top Features:\n")
                for feature in results['top_features'][:10]:
                    f.write(f"{feature['rank']}. **Feature {feature['feature_idx']}** (activation: {feature['activation_value']:.4f})\n")
                    f.write(f"   - {feature['description']}\n")
                    if feature['neuronpedia_url']:
                        f.write(f"   - [Neuronpedia Link]({feature['neuronpedia_url']})\n")
                    f.write("\n")
            
            f.write(f"---\n*Generated on {timestamp}*\n")
        
        print(f"\n💾 Analysis results saved:")
        print(f"   📊 JSON: {json_file}")
        print(f"   📋 Report: {report_file}")
        
        return str(report_file)


def main():
    """Main function to run cluster-filtered SAE analysis."""
    print("🚀 Starting Cluster-Filtered SAE Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ClusterFilteredSAEAnalysis()
    
    try:
        # Step 1: Load cluster analysis results
        # You need to specify the path to your cluster analysis pickle file
        cluster_analysis_files = list(Path(".").glob("cluster_analysis_*.pkl"))
        if not cluster_analysis_files:
            print("❌ No cluster analysis files found. Please run the UMAP clustering first.")
            return
        
        # Use the most recent cluster analysis file
        cluster_file = sorted(cluster_analysis_files, key=lambda x: x.stat().st_mtime)[-1]
        print(f"📁 Using cluster analysis file: {cluster_file}")
        
        analyzer.load_cluster_analysis(str(cluster_file))
        
        # Step 2: Load original activations
        original_activations = analyzer.load_original_activations()
        
        # Step 3: Extract cluster-filtered activations
        filtered_activations = analyzer.extract_cluster_filtered_activations(original_activations)
        
        # Step 4: Load SAE
        success = analyzer.load_neuronpedia_sae()
        if not success:
            print("❌ Failed to load SAE. Exiting.")
            return
        
        # Step 5: Run SAE analysis on filtered activations
        sae_results = analyzer.run_sae_analysis_on_filtered_activations()
        
        # Step 6: Fetch feature descriptions
        all_feature_indices = set()
        for state_results in sae_results.values():
            all_feature_indices.update(state_results['top_features']['indices'][:20])
        
        descriptions = analyzer.fetch_feature_descriptions(list(all_feature_indices))
        
        # Step 7: Create analysis report
        report_file = analyzer.create_analysis_report(sae_results, descriptions)
        
        # Step 8: Print summary
        print(f"\n📊 Cluster-Filtered SAE Analysis Results:")
        print(f"Pattern: {analyzer.pattern_name}")
        
        for state_name, results in sae_results.items():
            print(f"\n{state_name.upper()} STATE:")
            print(f"   Filtered Activations: {results['num_filtered_activations']}")
            print(f"   L0 Sparsity: {results['avg_l0_norm']:.2f}")
            print(f"   Reconstruction MSE: {results['reconstruction_mse']:.6f}")
            print(f"   Top feature: {results['top_features']['indices'][0]} (activation: {results['top_features']['values'][0]:.4f})")
        
        print(f"\n✅ Cluster-filtered SAE analysis completed!")
        print(f"📄 Report: {report_file}")
        
    except Exception as e:
        print(f"\n💥 Error during cluster-filtered SAE analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()