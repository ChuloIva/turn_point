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
import time
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    
    def get_bulk_explanations(self, model_id: str, sae_id: str) -> pd.DataFrame:
        """Fetch all explanations for an SAE as DataFrame."""
        url = f"{self.base_url}/api/explanation/export?modelId={model_id}&saeId={sae_id}"
        try:
            response = self.fetch_with_retry(url)
            if isinstance(response, list):
                return pd.DataFrame(response)
            elif isinstance(response, dict) and 'explanations' in response:
                return pd.DataFrame(response['explanations'])
            else:
                logger.warning(f"Unexpected bulk response format: {type(response)}")
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to fetch bulk explanations: {e}")
            return pd.DataFrame()
    
    def batch_get_features(self, model_id: str, layer: str, feature_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch multiple features with rate limiting and error handling."""
        results = {}
        for feature_idx in feature_indices:
            try:
                result = self.get_feature_explanation(model_id, layer, feature_idx)
                results[feature_idx] = result
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch feature {feature_idx}: {e}")
                results[feature_idx] = {}
        return results
    
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
    
    def load_existing_activations(self) -> Dict[str, torch.Tensor]:
        """Load negative, positive, and transition activations for one cognitive state."""
        print(f"\n📊 Loading activations for all three states...")
        
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
                    # Only load layer 17 activations
                    if 'layer_17' in key:
                        new_key = f"{state_name}_{key.split('_', 1)[1]}"
                        all_activations[new_key] = tensor
        
        print(f"✅ Loaded activations for {len(activation_files)} states")
        print(f"   Available patterns: {len(self.enriched_metadata)}")
        print(f"   Tensor keys: {list(all_activations.keys())}")
        
        # Set pattern data from first pattern
        if self.enriched_metadata:
            self.pattern_data = self.enriched_metadata[0]
            pattern_name = self.pattern_data.get('cognitive_pattern_name', 'Unknown Pattern')
            print(f"   Using pattern: {pattern_name}")
        
        return all_activations
    
    def prepare_activations_for_sae(self, activations: Dict[str, torch.Tensor], pattern_index: int = 0) -> Dict[str, torch.Tensor]:
        """Prepare existing activations for SAE analysis by selecting a specific pattern."""
        print(f"\n🧠 Preparing activations for SAE analysis (pattern {pattern_index})...")
        
        if pattern_index >= len(self.enriched_metadata):
            raise IndexError(f"Pattern index {pattern_index} out of range (max: {len(self.enriched_metadata) - 1})")
        
        # Update pattern data to selected pattern
        self.pattern_data = self.enriched_metadata[pattern_index]
        pattern_name = self.pattern_data.get('cognitive_pattern_name', 'Unknown Pattern')
        print(f"   Selected pattern: {pattern_name}")
        
        # Extract activations for the selected pattern (index in the batch dimension)
        prepared_activations = {}
        for key, tensor in activations.items():
            if len(tensor.shape) >= 2:  # Should be [batch, seq, hidden] or similar
                # Select the pattern at the given index from the batch
                selected_activation = tensor[pattern_index:pattern_index+1]  # Keep batch dim = 1
                prepared_activations[key] = selected_activation
                print(f"   {key}: {selected_activation.shape}")
            else:
                prepared_activations[key] = tensor
        
        self.activations = prepared_activations
        print(f"✅ Prepared activations for pattern analysis")
        
        return prepared_activations
    
    def prepare_activations_for_pattern_name(self, activations: Dict[str, torch.Tensor], 
                                           pattern_name: str = None, 
                                           use_all_samples: bool = True) -> Dict[str, torch.Tensor]:
        """Prepare activations for SAE analysis using all samples of a specific cognitive pattern."""
        print(f"\n🧠 Preparing activations for cognitive pattern: {pattern_name or 'first pattern'}...")
        
        # Find all indices for the specified pattern
        if pattern_name is None:
            # Use first pattern if none specified
            pattern_name = self.enriched_metadata[0].get('cognitive_pattern_name', 'Unknown Pattern')
        
        # Find all samples with this pattern name
        pattern_indices = []
        for i, entry in enumerate(self.enriched_metadata):
            entry_pattern_name = entry.get('cognitive_pattern_name')
            if entry_pattern_name == pattern_name:
                pattern_indices.append(i)
        
        if not pattern_indices:
            raise ValueError(f"No samples found for pattern: {pattern_name}")
        
        print(f"   Found {len(pattern_indices)} samples for pattern: {pattern_name}")
        print(f"   Sample indices: {pattern_indices}")
        
        # Update pattern data to first sample (for metadata)
        self.pattern_data = self.enriched_metadata[pattern_indices[0]]
        
        prepared_activations = {}
        
        if use_all_samples:
            # Use ALL samples and ALL tokens for this pattern
            for key, tensor in activations.items():
                if len(tensor.shape) >= 2:  # Should be [batch, seq, hidden] or similar
                    # Select all samples for this pattern
                    selected_samples = tensor[pattern_indices]  # Shape: [n_samples, seq_len, hidden_dim]
                    
                    # Flatten to combine all samples and all tokens: [n_samples * seq_len, hidden_dim]
                    flattened_activation = selected_samples.reshape(-1, selected_samples.shape[-1])
                    prepared_activations[key] = flattened_activation
                    print(f"   {key}: {len(pattern_indices)} samples × {selected_samples.shape[1]} tokens = {flattened_activation.shape}")
                else:
                    prepared_activations[key] = tensor
        else:
            # Use only first sample (original behavior)
            for key, tensor in activations.items():
                if len(tensor.shape) >= 2:
                    selected_activation = tensor[pattern_indices[0]:pattern_indices[0]+1]
                    prepared_activations[key] = selected_activation
                    print(f"   {key}: {selected_activation.shape}")
                else:
                    prepared_activations[key] = tensor
        
        self.activations = prepared_activations
        print(f"✅ Prepared activations for pattern analysis")
        
        return prepared_activations
    
    def run_pca_on_activations(self, activations: Dict[str, torch.Tensor], n_components: int = 5) -> Dict[str, Dict]:
        """Run PCA on raw activations to find principal component directions, then analyze with SAE."""
        print(f"\n🧮 Running PCA analysis on raw activations...")
        
        pca_results = {}
        
        for state_key, activation_tensor in activations.items():
            if 'layer_17' not in state_key:
                continue
                
            state_name = state_key.split('_layer_')[0]
            print(f"   Analyzing {state_name} activations: {activation_tensor.shape}")
            
            # Flatten activations to [n_tokens, hidden_dim]
            if len(activation_tensor.shape) == 3:
                flat_acts = activation_tensor.squeeze(0)  # Remove batch dim: [seq_len, hidden_dim]
            else:
                flat_acts = activation_tensor
            
            # Convert to numpy for sklearn
            acts_np = flat_acts.detach().cpu().numpy()
            print(f"     Flattened shape: {acts_np.shape}")
            
            # Center the data (subtract mean)
            scaler = StandardScaler(with_std=False)  # Only center, don't scale
            acts_centered = scaler.fit_transform(acts_np)
            
            # Run PCA
            pca = PCA(n_components=n_components)
            pca_transformed = pca.fit_transform(acts_centered)
            
            print(f"     PCA explained variance ratios: {pca.explained_variance_ratio_}")
            print(f"     Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
            
            # Get the principal component directions (in original feature space)
            pc_directions = pca.components_  # Shape: [n_components, hidden_dim]
            
            # Convert PC directions to torch tensors and pass through SAE
            pc_sae_results = {}
            for i, pc_direction in enumerate(pc_directions):
                pc_tensor = torch.from_numpy(pc_direction).float().to(self.sae.device)
                pc_tensor = pc_tensor.unsqueeze(0)  # Add batch dimension: [1, hidden_dim]
                
                # Pass through SAE
                feature_acts = self.sae.encode(pc_tensor)
                
                # Get top features for this PC
                feature_acts_flat = feature_acts.squeeze(0)  # Remove batch dim
                top_values, top_indices = torch.topk(torch.abs(feature_acts_flat), k=20)
                
                # Store results
                pc_sae_results[f'PC{i+1}'] = {
                    'explained_variance_ratio': float(pca.explained_variance_ratio_[i]),
                    'top_features': {
                        'indices': top_indices.detach().cpu().numpy().tolist(),
                        'values': feature_acts_flat[top_indices].detach().cpu().numpy().tolist(),
                        'abs_values': top_values.detach().cpu().numpy().tolist()
                    }
                }
                
                print(f"     PC{i+1} (var={pca.explained_variance_ratio_[i]:.3f}): Top feature {top_indices[0].item()} = {feature_acts_flat[top_indices[0]].item():.4f}")
            
            pca_results[state_name] = {
                'pca_object': pca,
                'scaler': scaler,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_variance_explained': float(pca.explained_variance_ratio_.sum()),
                'pc_sae_results': pc_sae_results
            }
        
        return pca_results
    
    def run_pca_on_all_tokens(self, activations: Dict[str, torch.Tensor], n_components: int = 5) -> Dict[str, Dict]:
        """Run PCA on ALL tokens from ALL samples of the cognitive pattern."""
        print(f"\n🧮 Running PCA analysis on all tokens from all samples...")
        
        pca_results = {}
        
        for state_key, activation_tensor in activations.items():
            if 'layer_17' not in state_key:
                continue
                
            state_name = state_key.split('_layer_')[0]
            print(f"   Analyzing {state_name} activations: {activation_tensor.shape}")
            
            # The tensor is already flattened to [n_samples * seq_len, hidden_dim]
            # from prepare_activations_for_pattern_name
            flat_acts = activation_tensor
            
            # Convert to numpy for sklearn
            acts_np = flat_acts.detach().cpu().numpy()
            print(f"     Total tokens for PCA: {acts_np.shape[0]}")
            print(f"     Hidden dimensions: {acts_np.shape[1]}")
            
            # Center the data (subtract mean across all tokens)
            scaler = StandardScaler(with_std=False)  # Only center, don't scale
            acts_centered = scaler.fit_transform(acts_np)
            
            # Run PCA on all tokens
            pca = PCA(n_components=n_components)
            pca_transformed = pca.fit_transform(acts_centered)
            
            print(f"     PCA explained variance ratios: {pca.explained_variance_ratio_}")
            print(f"     Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
            
            # Get the principal component directions (in original feature space)
            pc_directions = pca.components_  # Shape: [n_components, hidden_dim]
            
            # Convert PC directions to torch tensors and pass through SAE
            pc_sae_results = {}
            for i, pc_direction in enumerate(pc_directions):
                pc_tensor = torch.from_numpy(pc_direction).float().to(self.sae.device)
                pc_tensor = pc_tensor.unsqueeze(0)  # Add batch dimension: [1, hidden_dim]
                
                # Pass through SAE
                feature_acts = self.sae.encode(pc_tensor)
                
                # Get top features for this PC
                feature_acts_flat = feature_acts.squeeze(0)  # Remove batch dim
                top_values, top_indices = torch.topk(torch.abs(feature_acts_flat), k=20)
                
                # Store results
                pc_sae_results[f'PC{i+1}'] = {
                    'explained_variance_ratio': float(pca.explained_variance_ratio_[i]),
                    'top_features': {
                        'indices': top_indices.detach().cpu().numpy().tolist(),
                        'values': feature_acts_flat[top_indices].detach().cpu().numpy().tolist(),
                        'abs_values': top_values.detach().cpu().numpy().tolist()
                    }
                }
                
                print(f"     PC{i+1} (var={pca.explained_variance_ratio_[i]:.3f}): Top feature {top_indices[0].item()} = {feature_acts_flat[top_indices[0]].item():.4f}")
            
            pca_results[state_name] = {
                'pca_object': pca,
                'scaler': scaler,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_variance_explained': float(pca.explained_variance_ratio_.sum()),
                'pc_sae_results': pc_sae_results,
                'total_tokens_analyzed': acts_np.shape[0]  # Add this for tracking
            }
        
        return pca_results
    
    def analyze_pca_features_with_neuronpedia(self, pca_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Collect PCA features and get Neuronpedia interpretations."""
        print(f"\n🔍 Collecting PCA-derived features for Neuronpedia analysis...")
        
        # Collect all unique features from PCA analysis
        all_pca_features = set()
        for state_name, state_results in pca_results.items():
            for pc_name, pc_data in state_results['pc_sae_results'].items():
                for feature_idx in pc_data['top_features']['indices']:
                    all_pca_features.add(feature_idx)
        
        print(f"   Found {len(all_pca_features)} unique PCA-derived features")
        
        # Fetch Neuronpedia descriptions
        descriptions = self.fetch_feature_descriptions(list(all_pca_features))
        
        # Create structured dataset
        pca_analysis_dataset = {
            'analysis_metadata': {
                'pattern_name': self.pattern_data.get('cognitive_pattern_name', 'Unknown'),
                'pattern_type': self.pattern_data.get('cognitive_pattern_type', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'total_unique_features': len(all_pca_features),
                'sae_config': {
                    'release': 'gemma-scope-2b-pt-res',
                    'sae_id': 'layer_17/width_65k/average_l0_125',
                    'layer': 17
                }
            },
            'states': {}
        }
        
        # Process each state's PCA results
        for state_name, state_results in pca_results.items():
            state_data = {
                'total_variance_explained': state_results['total_variance_explained'],
                'explained_variance_ratios': state_results['explained_variance_ratio'],
                'principal_components': {}
            }
            
            for pc_name, pc_data in state_results['pc_sae_results'].items():
                pc_features = []
                
                for i, feature_idx in enumerate(pc_data['top_features']['indices']):
                    feature_info = {
                        'rank': i + 1,
                        'feature_index': feature_idx,
                        'activation_value': pc_data['top_features']['values'][i],
                        'abs_activation': pc_data['top_features']['abs_values'][i],
                        'description': descriptions.get(feature_idx, {}).get('description', 'No description'),
                        'autointerp_explanation': descriptions.get(feature_idx, {}).get('autointerp_explanation', ''),
                        'autointerp_score': descriptions.get(feature_idx, {}).get('autointerp_score', 0.0),
                        'neuronpedia_url': descriptions.get(feature_idx, {}).get('neuronpedia_url', ''),
                        'pos_tokens': descriptions.get(feature_idx, {}).get('pos_tokens', []),
                        'neg_tokens': descriptions.get(feature_idx, {}).get('neg_tokens', [])
                    }
                    pc_features.append(feature_info)
                
                state_data['principal_components'][pc_name] = {
                    'explained_variance_ratio': pc_data['explained_variance_ratio'],
                    'top_features': pc_features
                }
            
            pca_analysis_dataset['states'][state_name] = state_data
        
        return pca_analysis_dataset
    
    def save_pca_analysis_json(self, pca_dataset: Dict[str, Any]) -> str:
        """Save PCA analysis results as JSON dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern_name = pca_dataset['analysis_metadata']['pattern_name']
        safe_pattern_name = "".join(c for c in pattern_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_pattern_name = safe_pattern_name.replace(' ', '_')
        
        output_file = self.output_dir / f"pca_analysis_{safe_pattern_name}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(pca_dataset, f, indent=2)
        
        print(f"💾 PCA analysis JSON saved: {output_file}")
        
        # Also create a summary
        summary_file = self.output_dir / f"pca_summary_{safe_pattern_name}_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"PCA Analysis Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Pattern: {pca_dataset['analysis_metadata']['pattern_name']}\n")
            f.write(f"Total Unique Features: {pca_dataset['analysis_metadata']['total_unique_features']}\n\n")
            
            for state_name, state_data in pca_dataset['states'].items():
                f.write(f"{state_name.upper()} STATE:\n")
                f.write(f"Total Variance Explained: {state_data['total_variance_explained']:.3f}\n")
                
                for pc_name, pc_data in state_data['principal_components'].items():
                    f.write(f"\n  {pc_name} (variance: {pc_data['explained_variance_ratio']:.3f}):\n")
                    for feature in pc_data['top_features'][:5]:  # Top 5 per PC
                        f.write(f"    {feature['rank']}. Feature {feature['feature_index']}: {feature['activation_value']:.4f}\n")
                        f.write(f"       \"{feature['description'][:80]}...\"\n")
                f.write(f"\n")
        
        print(f"📄 PCA summary saved: {summary_file}")
        return str(output_file)
    
    def get_neuronpedia_identifiers(self) -> Tuple[str, str]:
        """Convert SAE config to Neuronpedia identifiers with fallbacks."""
        if not self.sae:
            raise ValueError("SAE not loaded")
        
        # Get model name from metadata
        model_name = None
        if hasattr(self.sae.cfg, 'metadata') and hasattr(self.sae.cfg.metadata, 'model_name'):
            model_name = self.sae.cfg.metadata.model_name
        
        # Try different layer format combinations
        layer_formats = [
            "17-gemmascope-res-65k",  # Standard format for layer 17
            "layer_17/width_65k/average_l0_125",  # SAE ID format
            "17-res-65k",
            "layer_17",
            "17"
        ]
        
        model_formats = [
            model_name if model_name else "gemma-2-2b",  # Use metadata model name
            "gemma-2b",  # Alternative format
            "gemma-2-2b-it",  # Another possible format
        ]
        
        print(f"   Trying model formats: {model_formats}")
        print(f"   Trying layer formats: {layer_formats}")
        
        # Return the first combination to try
        return model_formats[0], layer_formats[0]
    
    def _extract_from_neuronpedia_id(self) -> Tuple[str, str]:
        """Extract from SAE config neuronpedia_id if available."""
        if hasattr(self.sae.cfg, 'neuronpedia_id') and self.sae.cfg.neuronpedia_id:
            parts = self.sae.cfg.neuronpedia_id.split('/')
            if len(parts) >= 2:
                return parts[0], parts[1]
        raise ValueError("No neuronpedia_id available")
    
    def _extract_from_model_name(self) -> Tuple[str, str]:
        """Extract from model name in SAE config."""
        if hasattr(self.sae.cfg, 'model_name') and self.sae.cfg.model_name:
            model_name = self.sae.cfg.model_name.lower()
            if 'gemma' in model_name and '2b' in model_name:
                return "gemma-2b", "21-res-65k"
        raise ValueError("Could not extract from model name")
    
    def _extract_from_release_name(self) -> Tuple[str, str]:
        """Extract from release name."""
        # For gemma-scope-2b-pt-res, extract model and layer info
        hook_name = None
        if hasattr(self.sae.cfg, 'hook_name'):
            hook_name = self.sae.cfg.hook_name
        elif hasattr(self.sae.cfg, 'metadata') and hasattr(self.sae.cfg.metadata, 'hook_name'):
            hook_name = self.sae.cfg.metadata.hook_name
            
        if hook_name and ('blocks.17' in hook_name or 'layer_17' in hook_name):
            return "gemma-2-2b", "17-gemmascope-res-65k"
        raise ValueError("Could not extract from release name")
    
    def _manual_mapping(self) -> Tuple[str, str]:
        """Manual mapping for known SAE configurations."""
        # Default mapping for the current SAE being used
        return "gemma-2-2b", "17-gemmascope-res-65k"
    
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
    
    def load_neuronpedia_sae(self) -> bool:
        """Load the specific SAE from HuggingFace: google/gemma-scope-2b-pt-res/layer_17/width_65k/average_l0_125"""
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
            
            # Try different ways to access hook name for compatibility
            hook_name = None
            if hasattr(self.sae.cfg, 'hook_name'):
                hook_name = self.sae.cfg.hook_name
                print(f"   Found hook_name directly: {hook_name}")
            elif hasattr(self.sae.cfg, 'metadata'):
                print(f"   Found metadata: {type(self.sae.cfg.metadata)}")
                if hasattr(self.sae.cfg.metadata, 'hook_name'):
                    hook_name = self.sae.cfg.metadata.hook_name
                    print(f"   Found hook_name in metadata: {hook_name}")
                
                # Check for neuronpedia_id in metadata
                if hasattr(self.sae.cfg.metadata, 'neuronpedia_id'):
                    neuronpedia_id = self.sae.cfg.metadata.neuronpedia_id
                    print(f"   Found neuronpedia_id: {neuronpedia_id}")
                
                # Print other useful metadata
                if hasattr(self.sae.cfg.metadata, 'model_name'):
                    model_name = self.sae.cfg.metadata.model_name
                    print(f"   Found model_name: {model_name}")
            
            print(f"   Hook point: {hook_name}")
            print(f"   Input dims: {self.sae.cfg.d_in}")
            print(f"   SAE dims: {self.sae.cfg.d_sae}")
            print(f"   Device: {self.sae.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading SAE: {e}")
            print(f"   Trying alternative SAE identifiers...")
            
            # Try alternative identifiers that might work
            alternatives = [
                ("gemma-scope-2b-pt-res-canonical", "layer_17/width_65k/average_l0_125"),
                ("gemma-scope-2b-pt-res", "layer_17/width_65k"),
                ("google/gemma-scope-2b-pt-res", "layer_17/width_65k/average_l0_125"),
            ]
            
            for alt_release, alt_sae_id in alternatives:
                try:
                    print(f"   Trying {alt_release}/{alt_sae_id}...")
                    self.sae = SAE.from_pretrained(
                        release=alt_release,
                        sae_id=alt_sae_id,
                        device=str(self.device)
                    )
                    print(f"✅ Alternative SAE loaded: {alt_release}/{alt_sae_id}")
                    return True
                except Exception as alt_e:
                    print(f"   Failed: {alt_e}")
                    continue
            
            print("❌ All SAE loading attempts failed")
            return False
    
    def run_sae_analysis(self, layer: int = 17) -> Dict[str, Any]:
        """Run SAE analysis on manually loaded activations using proper SAE workflow."""
        if not self.sae:
            raise ValueError("SAE not loaded. Call load_neuronpedia_sae() first.")
        
        if not self.activations:
            raise ValueError("No activations available. Call prepare_activations_for_sae() first.")
        
        print(f"\n🔬 Running SAE analysis on layer {layer}...")
        
        # Find activations for the specific layer
        available_keys = [k for k in self.activations.keys() if f'layer_{layer}' in k]
        if not available_keys:
            raise ValueError(f"No activations found for layer {layer}. Available: {list(self.activations.keys())}")
        
        results = {}
        
        # Set SAE to eval mode (from tutorial)
        self.sae.eval()
        
        with torch.no_grad():  # From tutorial - prevents error if expecting dead neuron mask
            # Process each activation type (negative, positive, transition)
            for key in available_keys:
                act_type = key.split('_layer_')[0]  # Extract 'negative', 'positive', 'transition'
                activations = self.activations[key]  # Keep using manually loaded activations
                
                print(f"   Processing {act_type} activations: {activations.shape}")
                
                # Move to SAE device
                activations = activations.to(self.sae.device)
                
                # Use SAE encode/decode directly on activations (like tutorial - no flattening needed)
                feature_acts = self.sae.encode(activations)
                sae_out = self.sae.decode(feature_acts)
                
                # Calculate metrics (like tutorial)
                l0_norm = (feature_acts > 0).sum(dim=-1).float()
                reconstruction_mse = torch.nn.functional.mse_loss(activations, sae_out)
                
                # Top 20 features by average activation
                avg_feature_acts = feature_acts.mean(dim=0)
                top_values, top_indices = torch.topk(avg_feature_acts, k=20)
                
                results[act_type] = {
                    'feature_activations': feature_acts.cpu(),
                    'avg_l0_norm': l0_norm.mean().item(),
                    'reconstruction_mse': reconstruction_mse.item(),
                    'top_features': {
                        'values': top_values.detach().cpu().numpy().tolist(),
                        'indices': top_indices.detach().cpu().numpy().tolist()
                    }
                }
                
                print(f"     L0 sparsity: {l0_norm.mean().item():.2f}")
                print(f"     Reconstruction MSE: {reconstruction_mse.item():.6f}")
                print(f"     Top feature: {top_indices[0].item()} (activation: {top_values[0].item():.4f})")
        
        return results
    
    def analyze_feature_differences(self, sae_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze differences between cognitive states in feature space."""
        print(f"\n🔍 Analyzing feature differences...")
        
        # Get available activation types
        available_types = list(sae_results.keys())
        print(f"   Available activation types: {available_types}")
        
        # For feature difference analysis, we need to average activations across positions
        # since our current shape might be [positions, features]
        averaged_features = {}
        for act_type, results in sae_results.items():
            feature_acts = results['feature_activations']
            if len(feature_acts.shape) == 2:  # [positions, features]
                averaged_features[act_type] = feature_acts.mean(dim=0)  # Average across positions
            else:
                averaged_features[act_type] = feature_acts.flatten()
            
            print(f"   {act_type} features shape: {averaged_features[act_type].shape}")
        
        analysis = {}
        
        # Generate all pairwise comparisons between available types
        for i, type1 in enumerate(available_types):
            for j, type2 in enumerate(available_types):
                if i < j:  # Avoid duplicate comparisons
                    comparison_name = f"{type1}_to_{type2}"
                    diff_tensor = averaged_features[type2] - averaged_features[type1]
                    
                    def get_top_diff_features(diff_tensor, name, k=10):
                        abs_diff = torch.abs(diff_tensor)
                        values, indices = torch.topk(abs_diff, k=k)
                        directions = torch.sign(diff_tensor[indices])
                        
                        print(f"   Top {k} differential features ({name}):")
                        feature_info = []
                        for idx, (val, feature_idx, direction) in enumerate(zip(values, indices, directions)):
                            direction_str = "↑" if direction > 0 else "↓"
                            print(f"     {idx+1}. Feature {feature_idx.item()}: {direction_str} {val.item():.4f}")
                            feature_info.append({
                                'rank': idx+1,
                                'feature_idx': feature_idx.item(),
                                'abs_diff': val.item(),
                                'direction': direction_str,
                                'raw_diff': diff_tensor[feature_idx].item()
                            })
                        
                        return feature_info
                    
                    analysis[comparison_name] = get_top_diff_features(
                        diff_tensor, f"{type1.title()} → {type2.title()}"
                    )
        
        return analysis
    
    def fetch_feature_descriptions(self, feature_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch descriptions for a list of feature indices with multiple identifier attempts."""
        print(f"\n📖 Fetching descriptions for {len(feature_indices)} features...")
        
        # Get potential identifier combinations
        model_name = None
        if hasattr(self.sae.cfg, 'metadata') and hasattr(self.sae.cfg.metadata, 'model_name'):
            model_name = self.sae.cfg.metadata.model_name
        
        # Use the exact working API URL format:
        # https://www.neuronpedia.org/api/feature/gemma-2-2b/17-gemmascope-res-65k/878
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
                    
                    # Check if we got actual data (not just empty response)
                    if feature_data:
                        print(f"     Raw API response keys: {list(feature_data.keys()) if isinstance(feature_data, dict) else 'Not a dict'}")
                        # Extract explanation from explanations array if available
                        explanation = ""
                        explanation_score = 0.0
                        
                        if 'explanations' in feature_data and feature_data['explanations']:
                            # Get the first explanation
                            first_explanation = feature_data['explanations'][0]
                            explanation = first_explanation.get('description', '')
                            # Try different score fields that might exist
                            scores_array = first_explanation.get('scores', [])
                            explanation_score = scores_array[0] if scores_array else 0.0
                        
                        # Also check for direct fields (backward compatibility)
                        if not explanation:
                            explanation = feature_data.get('explanation', feature_data.get('description', ''))
                            explanation_score = feature_data.get('explanationScore', 0.0)
                        
                        # Extract activating tokens from the activations data
                        pos_tokens = []
                        neg_tokens = []
                        
                        # Try different field names for token data
                        pos_tokens = feature_data.get('pos_str', [])
                        neg_tokens = feature_data.get('neg_str', [])
                        
                        # If pos_str/neg_str not found, try extracting from activations
                        if not pos_tokens and 'activations' in feature_data:
                            # Extract tokens from activation entries
                            for activation in feature_data['activations'][:10]:  # Top 10 activations
                                if 'tokens' in activation:
                                    tokens = activation['tokens']
                                    if 'maxValueTokenIndex' in activation and activation['maxValueTokenIndex'] < len(tokens):
                                        max_token = tokens[activation['maxValueTokenIndex']]
                                        if max_token not in pos_tokens:
                                            pos_tokens.append(max_token)
                        
                        # Always create an entry if we got any response from the API
                        description = explanation if explanation else f"Activates on: {', '.join(pos_tokens[:5])}" if pos_tokens else "No description available"
                        
                        descriptions[feature_idx] = {
                            'description': description,
                            'autointerp_explanation': explanation,
                            'autointerp_score': explanation_score,
                            'neuronpedia_url': f"https://neuronpedia.org/{model_id}/{sae_id}/{feature_idx}",
                            'pos_tokens': pos_tokens[:10],  # Top 10 positive tokens
                            'neg_tokens': neg_tokens[:10]   # Top 10 negative tokens
                        }
                        print(f"     ✅ Success with {model_id}/{sae_id}")
                        if explanation:
                            print(f"     Explanation: {explanation[:50]}...")
                        elif pos_tokens:
                            print(f"     Activates on: {', '.join(pos_tokens[:3])}...")
                        else:
                            print(f"     Basic data retrieved (no explanation)")
                        feature_fetched = True
                        break
                    
                    print(f"     No useful data returned for {model_id}/{sae_id}")
                    
                except Exception as e:
                    print(f"     Failed {model_id}/{sae_id}: {e}")
                    import traceback
                    traceback.print_exc()
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
    
    def analyze_feature_differences_with_descriptions(self, sae_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced feature analysis with Neuronpedia descriptions."""
        print(f"\n🔍 Analyzing feature differences with descriptions...")
        
        # Run the existing differential analysis
        analysis = self.analyze_feature_differences(sae_results)
        
        # Collect all unique feature indices from top features
        all_feature_indices = set()
        for comparison_name, features in analysis.items():
            for feature in features[:10]:  # Top 10 per comparison
                all_feature_indices.add(feature['feature_idx'])
        
        print(f"   Found {len(all_feature_indices)} unique features across all comparisons")
        
        # Fetch descriptions for all unique features
        descriptions = self.fetch_feature_descriptions(list(all_feature_indices))
        
        # Enrich analysis with descriptions
        for comparison_name, features in analysis.items():
            for feature in features:
                feature_idx = feature['feature_idx']
                feature['description'] = descriptions.get(feature_idx, {})
        
        print(f"\n🔍 Top Features with Descriptions:")
        for comparison_name, features in analysis.items():
            print(f"   {comparison_name}:")
            for i, feature in enumerate(features[:3]):  # Show top 3 with descriptions
                feature_idx = feature['feature_idx']
                direction = feature['direction']
                abs_diff = feature['abs_diff']
                desc = feature['description'].get('description', 'No description')[:60]
                print(f"     {i+1}. Feature {feature_idx}: {direction} {abs_diff:.4f} - \"{desc}...\"")
        
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
    
    def create_feature_summary_csv(self, differential_analysis: Dict[str, Any], 
                                  descriptions: Dict[int, Dict[str, Any]], timestamp: str) -> str:
        """Create CSV summary of top differential features with descriptions."""
        print(f"\n📊 Creating feature summary CSV...")
        
        summary_data = []
        for transition, features in differential_analysis.items():
            for feature in features[:10]:  # Top 10 per transition
                feature_idx = feature['feature_idx']
                desc_data = descriptions.get(feature_idx, {})
                
                summary_data.append({
                    'transition_type': transition,
                    'feature_idx': feature_idx,
                    'rank': feature['rank'],
                    'direction': feature['direction'],
                    'abs_diff': feature['abs_diff'],
                    'raw_diff': feature['raw_diff'],
                    'description': desc_data.get('description', ''),
                    'autointerp_explanation': desc_data.get('autointerp_explanation', ''),
                    'autointerp_score': desc_data.get('autointerp_score', 0.0),
                    'neuronpedia_url': desc_data.get('neuronpedia_url', '')
                })
        
        # Save as JSON
        df = pd.DataFrame(summary_data)
        json_file = self.output_dir / f"feature_summary_{timestamp}.json"
        df.to_json(json_file, orient='records', indent=2)
        
        print(f"   📊 JSON saved: {json_file}")
        print(f"   📊 Records: {len(summary_data)}")
        
        return str(json_file)
    
    def extract_descriptions_from_analysis(self, differential_analysis: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Extract descriptions from enriched differential analysis for separate handling."""
        descriptions = {}
        for comparison_name, features in differential_analysis.items():
            for feature in features:
                if 'description' in feature:
                    descriptions[feature['feature_idx']] = feature['description']
        return descriptions
    
    def save_results(self, sae_results: Dict[str, Any], differential_analysis: Dict[str, Any],
                    neuronpedia_urls: Dict[str, List[str]], descriptions: Dict[int, Dict[str, Any]], 
                    release: str, sae_id: str) -> str:
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
            'feature_descriptions': descriptions,
            'differential_analysis_with_descriptions': differential_analysis,
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
            
            # Top Feature Descriptions Section
            if 'differential_analysis_with_descriptions' in results_data:
                f.write("## Top Feature Descriptions\n")
                for transition, features in results_data['differential_analysis_with_descriptions'].items():
                    f.write(f"### {transition.replace('_', ' ').title()}\n")
                    for feature in features[:5]:
                        feature_idx = feature['feature_idx']
                        desc_data = feature.get('description', {})
                        
                        f.write(f"#### Feature {feature_idx} ({feature['direction']} {feature['abs_diff']:.4f})\n")
                        f.write(f"**Description:** {desc_data.get('description', 'No description')}\n\n")
                        
                        if desc_data.get('autointerp_explanation'):
                            f.write(f"**AutoInterp:** {desc_data['autointerp_explanation']}\n")
                            f.write(f"**Score:** {desc_data.get('autointerp_score', 0.0):.2f}\n\n")
                        
                        f.write(f"**Dashboard:** [{desc_data.get('neuronpedia_url', 'N/A')}]({desc_data.get('neuronpedia_url', '#')})\n\n")
            
            # Neuronpedia links
            f.write("## Neuronpedia Dashboard Links\n")
            for transition, url_data in results_data['neuronpedia_urls'].items():
                f.write(f"### {transition.replace('_', ' ').title()}\n")
                for data in url_data:
                    f.write(f"- [Feature {data['feature_idx']} ({data['direction']} {data['diff']:.4f})]({data['url']})\n")
                f.write("\n")
            
            f.write(f"---\n*Generated on {results_data['timestamp']}*\n")

    def create_simple_features_csv(self, sae_results: Dict[str, Any], 
                                  descriptions: Dict[int, Dict[str, Any]], timestamp: str) -> str:
        """Create a simple CSV with top features and descriptions for each state."""
        print(f"\n📊 Creating simple features CSV...")
        
        csv_data = []
        for state, results in sae_results.items():
            for i, (feature_idx, activation) in enumerate(zip(
                results['top_features']['indices'][:20],  # Top 20 features per state
                results['top_features']['values'][:20]
            )):
                desc_data = descriptions.get(feature_idx, {})
                
                csv_data.append({
                    'state': state,
                    'rank': i + 1,
                    'feature_idx': feature_idx,
                    'activation_value': activation,
                    'description': desc_data.get('description', ''),
                    'autointerp_explanation': desc_data.get('autointerp_explanation', ''),
                    'autointerp_score': desc_data.get('autointerp_score', 0.0),
                    'neuronpedia_url': desc_data.get('neuronpedia_url', ''),
                    'l0_sparsity': results['avg_l0_norm'],
                    'reconstruction_mse': results['reconstruction_mse']
                })
        
        # Save as JSON
        df = pd.DataFrame(csv_data)
        json_file = self.output_dir / f"feature_summary_{timestamp}.json"
        df.to_json(json_file, orient='records', indent=2)
        
        print(f"   📊 JSON saved: {json_file}")
        print(f"   📊 Records: {len(csv_data)}")
        
        return str(json_file)


def main():
    """Main function to run SAE analysis with optional PCA analysis and Neuronpedia interpretations."""
    
    # Configuration parameters
    RUN_PCA = False  # Set to True to enable PCA analysis, False to disable (default)
    
    if RUN_PCA:
        print("🚀 Starting PCA-SAE Analysis - All Samples & Tokens for One Cognitive Pattern")
    else:
        print("🚀 Starting SAE Analysis - All Samples & Tokens for One Cognitive Pattern")
    print("="*70)
    
    # Initialize test
    test = SinglePatternSAETest()
    
    try:
        # Step 1: Load all three activation types (negative, positive, transition)
        activations = test.load_existing_activations()
        
        # Step 2: Prepare activations for a specific pattern using ALL samples and tokens
        target_pattern = None  # Will use first pattern, or specify like "Executive Fatigue / Avolition"
        test.prepare_activations_for_pattern_name(
            activations, 
            pattern_name=target_pattern, 
            use_all_samples=True
        )
        
        # Step 3: Load SAE
        success = test.load_neuronpedia_sae()
        if not success:
            print("❌ Failed to load SAE. Exiting.")
            return
        
        if RUN_PCA:
            # PCA Analysis Path
            print("\n🧮 Running PCA analysis...")
            
            # Step 4: Run PCA on ALL tokens from ALL samples
            pca_results = test.run_pca_on_all_tokens(test.activations, n_components=5)
            
            # Step 5: Analyze PCA features with Neuronpedia
            pca_dataset = test.analyze_pca_features_with_neuronpedia(pca_results)
            
            # Step 6: Save JSON dataset
            json_file = test.save_pca_analysis_json(pca_dataset)
            
            # Step 7: Print summary results
            print(f"\n📊 PCA-SAE Analysis Results:")
            print(f"Pattern: {pca_dataset['analysis_metadata']['pattern_name']}")
            print(f"Total Unique Features: {pca_dataset['analysis_metadata']['total_unique_features']}")
            
            for state_name, state_data in pca_dataset['states'].items():
                print(f"\n{state_name.upper()} STATE:")
                if state_name in pca_results and 'total_tokens_analyzed' in pca_results[state_name]:
                    print(f"   Total Tokens Analyzed: {pca_results[state_name]['total_tokens_analyzed']}")
                print(f"   Total Variance Explained by 5 PCs: {state_data['total_variance_explained']:.3f}")
                
                for pc_name, pc_data in state_data['principal_components'].items():
                    print(f"\n   {pc_name} (explains {pc_data['explained_variance_ratio']:.3f} variance):")
                    for feature in pc_data['top_features'][:3]:  # Top 3 features per PC
                        print(f"     {feature['rank']}. Feature {feature['feature_index']}: {feature['activation_value']:.4f}")
                        print(f"        \"{feature['description'][:80]}...\"")
                        if feature['neuronpedia_url']:
                            print(f"        🔗 {feature['neuronpedia_url']}")
            
            print(f"\n✅ PCA-SAE analysis with all samples and tokens completed!")
            print(f"📄 JSON dataset: {json_file}")
            
        else:
            # Standard SAE Analysis Path (no PCA)
            print("\n🔬 Running standard SAE analysis...")
            
            # Step 4: Run SAE analysis
            sae_results = test.run_sae_analysis()
            
            # Step 5: Analyze feature differences
            differential_analysis = test.analyze_feature_differences_with_descriptions(sae_results)
            
            # Step 6: Save feature CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            descriptions = test.extract_descriptions_from_analysis(differential_analysis)
            csv_file = test.create_simple_features_csv(sae_results, descriptions, timestamp)
            
            print(f"\n📊 Standard SAE Analysis Results:")
            print(f"Pattern: {test.pattern_data.get('cognitive_pattern_name', 'Unknown')}")
            
            for state_name, results in sae_results.items():
                print(f"\n{state_name.upper()} STATE:")
                print(f"   L0 Sparsity: {results['avg_l0_norm']:.2f}")
                print(f"   Reconstruction MSE: {results['reconstruction_mse']:.6f}")
                print(f"   Top feature: {results['top_features']['indices'][0]} (activation: {results['top_features']['values'][0]:.4f})")
            
            print(f"\n✅ Standard SAE analysis completed!")
            print(f"📄 Feature data: {csv_file}")
        
    except Exception as e:
        print(f"\n💥 Error during SAE analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
