"""
SAE Analysis Workflow: From Pre-computed Activations to Neuronpedia Analysis

This module provides a comprehensive pipeline for analyzing pre-computed neural network 
activations using Sparse Autoencoders (SAEs) and visualizing results through Neuronpedia.

Key Features:
- Load and process pre-computed activations
- Discover and load appropriate SAEs
- Generate feature activations and reconstructions
- Integrate with Neuronpedia for interpretability
- Advanced analysis tools (steering, ablation, attribution)
"""

import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from dataclasses import dataclass
from IPython.display import IFrame, display
import warnings

try:
    from sae_lens import SAE
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    SAE_AVAILABLE = True
except ImportError:
    print("Warning: sae-lens not available. Install with: pip install sae-lens")
    SAE_AVAILABLE = False

try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    print("Warning: transformer-lens not available. Install with: pip install transformer-lens")
    TRANSFORMER_LENS_AVAILABLE = False


@dataclass
class SAEAnalysisConfig:
    """Configuration for SAE analysis workflow"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sae_release: str = "gpt2-small-res-jb"
    sae_id: str = "blocks.7.hook_resid_pre"
    top_k_features: int = 10
    neuronpedia_base_url: str = "https://neuronpedia.org"
    analysis_output_dir: str = "sae_analysis_outputs"
    
    def __post_init__(self):
        Path(self.analysis_output_dir).mkdir(exist_ok=True)


class SAEActivationAnalyzer:
    """Main class for SAE-based activation analysis"""
    
    def __init__(self, config: SAEAnalysisConfig = None):
        self.config = config or SAEAnalysisConfig()
        self.sae = None
        self.available_saes = None
        self.feature_explanations = None
        
    def discover_available_saes(self) -> pd.DataFrame:
        """Discover all available pre-trained SAEs"""
        if not SAE_AVAILABLE:
            raise ImportError("sae-lens required for SAE discovery")
            
        self.available_saes = get_pretrained_saes_directory()
        
        print(f"Found {len(self.available_saes)} available SAEs")
        print("\nSample SAEs by model:")
        for model in self.available_saes['model_name'].unique()[:5]:
            model_saes = self.available_saes[self.available_saes['model_name'] == model]
            print(f"  {model}: {len(model_saes)} SAEs")
            
        return self.available_saes
    
    def find_matching_saes(self, model_name: str, hook_point: str = None) -> pd.DataFrame:
        """Find SAEs matching specific model and hook point criteria"""
        if self.available_saes is None:
            self.discover_available_saes()
            
        matches = self.available_saes[self.available_saes['model_name'].str.contains(model_name, case=False)]
        
        if hook_point:
            matches = matches[matches['hook_name'].str.contains(hook_point, case=False)]
            
        print(f"Found {len(matches)} matching SAEs for model '{model_name}'")
        if hook_point:
            print(f"  with hook point containing '{hook_point}'")
            
        return matches[['model_name', 'hook_name', 'release', 'sae_id', 'd_in']].head(10)
    
    def load_sae(self, release: str = None, sae_id: str = None) -> SAE:
        """Load a specific SAE from the registry"""
        if not SAE_AVAILABLE:
            raise ImportError("sae-lens required for SAE loading")
            
        release = release or self.config.sae_release
        sae_id = sae_id or self.config.sae_id
        
        print(f"Loading SAE: {release}/{sae_id}")
        self.sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=self.config.device
        )
        
        print(f"SAE loaded successfully:")
        print(f"  Hook point: {self.sae.cfg.hook_name}")
        print(f"  Input dimensions: {self.sae.cfg.d_in}")
        print(f"  Hidden dimensions: {self.sae.cfg.d_sae}")
        print(f"  Device: {self.sae.device}")
        
        return self.sae
    
    def load_activations(self, activation_path: str) -> torch.Tensor:
        """Load pre-computed activations from file"""
        activation_path = Path(activation_path)
        
        if not activation_path.exists():
            raise FileNotFoundError(f"Activation file not found: {activation_path}")
            
        print(f"Loading activations from: {activation_path}")
        
        if activation_path.suffix == '.pt':
            activations = torch.load(activation_path, map_location=self.config.device)
        elif activation_path.suffix == '.npy':
            activations = torch.from_numpy(np.load(activation_path)).to(self.config.device)
        else:
            raise ValueError(f"Unsupported activation file format: {activation_path.suffix}")
            
        print(f"Loaded activations with shape: {activations.shape}")
        return activations
    
    def process_activations(self, activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process activations through SAE to get features and reconstructions"""
        if self.sae is None:
            raise ValueError("SAE not loaded. Call load_sae() first.")
            
        # Ensure activations are on the same device as SAE
        activations = activations.to(self.sae.device)
        
        print(f"Processing activations of shape {activations.shape} through SAE...")
        
        # Get feature activations
        feature_acts = self.sae.encode(activations)
        
        # Get reconstructions
        reconstructions = self.sae.decode(feature_acts)
        
        # Calculate reconstruction error
        reconstruction_error = torch.nn.functional.mse_loss(activations, reconstructions, reduction='none')
        
        # Calculate sparsity (L0 norm)
        l0_norm = (feature_acts > 0).sum(dim=-1).float()
        
        results = {
            'original_activations': activations,
            'feature_activations': feature_acts,
            'reconstructions': reconstructions,
            'reconstruction_error': reconstruction_error,
            'l0_norm': l0_norm
        }
        
        print(f"SAE processing complete:")
        print(f"  Feature activations shape: {feature_acts.shape}")
        print(f"  Average L0 (sparsity): {l0_norm.mean().item():.2f}")
        print(f"  Average reconstruction MSE: {reconstruction_error.mean().item():.6f}")
        
        return results
    
    def find_top_features(self, feature_acts: torch.Tensor, 
                         position: int = -1, top_k: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find top-k activating features at a specific position"""
        top_k = top_k or self.config.top_k_features
        
        if len(feature_acts.shape) == 3:  # [batch, seq, features]
            # Use the specified position (default: last token)
            feature_vector = feature_acts[0, position, :]
        elif len(feature_acts.shape) == 2:  # [batch, features]
            feature_vector = feature_acts[0, :]
        else:
            raise ValueError(f"Unexpected feature_acts shape: {feature_acts.shape}")
            
        values, indices = torch.topk(feature_vector, k=top_k)
        
        print(f"Top {top_k} features at position {position}:")
        for i, (val, idx) in enumerate(zip(values, indices)):
            print(f"  {i+1}. Feature {idx}: {val:.4f}")
            
        return values, indices
    
    def get_neuronpedia_dashboard_url(self, feature_idx: int, 
                                     sae_release: str = None, sae_id: str = None) -> str:
        """Generate Neuronpedia dashboard URL for a specific feature"""
        sae_release = sae_release or self.config.sae_release.replace('-res-jb', '')
        sae_id = sae_id or self.config.sae_id.replace('blocks.', '').replace('.hook_resid_pre', '-res-jb')
        
        url = f"{self.config.neuronpedia_base_url}/{sae_release}/{sae_id}/{feature_idx}"
        url += "?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
        
        return url
    
    def display_feature_dashboard(self, feature_idx: int, width: int = 1200, height: int = 400):
        """Display Neuronpedia dashboard for a specific feature"""
        url = self.get_neuronpedia_dashboard_url(feature_idx)
        print(f"Feature {feature_idx} dashboard: {url}")
        
        try:
            return IFrame(url, width=width, height=height)
        except Exception as e:
            print(f"Could not display dashboard: {e}")
            return None
    
    def download_feature_explanations(self, model_id: str = "gpt2-small", 
                                    sae_id: str = "7-res-jb") -> pd.DataFrame:
        """Download feature explanations from Neuronpedia API"""
        url = f"{self.config.neuronpedia_base_url}/api/explanation/export"
        params = {"modelId": model_id, "saeId": sae_id}
        
        print(f"Downloading feature explanations for {model_id}/{sae_id}...")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            self.feature_explanations = pd.DataFrame(response.json())
            print(f"Downloaded {len(self.feature_explanations)} feature explanations")
            
            # Save to disk
            output_path = Path(self.config.analysis_output_dir) / f"feature_explanations_{model_id}_{sae_id}.csv"
            self.feature_explanations.to_csv(output_path, index=False)
            print(f"Saved explanations to: {output_path}")
            
            return self.feature_explanations
            
        except Exception as e:
            print(f"Failed to download feature explanations: {e}")
            return None
    
    def search_features_by_description(self, query: str, case_sensitive: bool = False) -> pd.DataFrame:
        """Search features by description content"""
        if self.feature_explanations is None:
            print("No feature explanations loaded. Call download_feature_explanations() first.")
            return None
            
        if case_sensitive:
            mask = self.feature_explanations['description'].str.contains(query, na=False)
        else:
            mask = self.feature_explanations['description'].str.contains(query, case=False, na=False)
            
        results = self.feature_explanations[mask]
        print(f"Found {len(results)} features matching '{query}'")
        
        return results[['feature', 'description']].head(20)
    
    def visualize_feature_activations(self, feature_acts: torch.Tensor, 
                                    position: int = -1, title: str = None):
        """Create visualization of feature activation patterns"""
        if len(feature_acts.shape) == 3:
            activations = feature_acts[0, position, :].cpu().numpy()
        else:
            activations = feature_acts[0, :].cpu().numpy()
            
        # Create histogram of activation values
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Activation Distribution', 'Top 50 Features', 'Sparsity Pattern', 'Activation Heatmap']
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=activations[activations > 0], nbinsx=50, name="Active Features"),
            row=1, col=1
        )
        
        # Top features bar chart
        top_vals, top_inds = torch.topk(torch.tensor(activations), k=50)
        fig.add_trace(
            go.Bar(x=list(range(50)), y=top_vals.numpy(), name="Top 50 Features"),
            row=1, col=2
        )
        
        # Sparsity pattern
        active_mask = activations > 0
        fig.add_trace(
            go.Scatter(y=activations, mode='markers', 
                      marker=dict(color=active_mask, colorscale='Viridis'),
                      name="All Features"),
            row=2, col=1
        )
        
        # Heatmap of top features
        top_100_vals, top_100_inds = torch.topk(torch.tensor(activations), k=100)
        reshaped = top_100_vals.reshape(10, 10).numpy()
        fig.add_trace(
            go.Heatmap(z=reshaped, colorscale='Viridis', name="Top 100 Heatmap"),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title or f"Feature Activation Analysis (L0: {active_mask.sum():.0f})",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def compare_activations(self, acts1: torch.Tensor, acts2: torch.Tensor, 
                          labels: Tuple[str, str] = ("Condition 1", "Condition 2")):
        """Compare feature activations between two conditions"""
        if len(acts1.shape) == 3:
            acts1 = acts1[0, -1, :]  # Last token
        if len(acts2.shape) == 3:
            acts2 = acts2[0, -1, :]
            
        diff = acts1 - acts2
        abs_diff = torch.abs(diff)
        
        top_diff_vals, top_diff_inds = torch.topk(abs_diff, k=20)
        
        print(f"Top 20 differential features between {labels[0]} and {labels[1]}:")
        for i, (val, idx) in enumerate(zip(top_diff_vals, top_diff_inds)):
            direction = "↑" if diff[idx] > 0 else "↓"
            print(f"  {i+1}. Feature {idx}: {direction} {val:.4f}")
        
        # Create comparison visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=acts1.cpu().numpy(), name=labels[0], mode='markers', opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            y=acts2.cpu().numpy(), name=labels[1], mode='markers', opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            y=diff.cpu().numpy(), name="Difference", mode='markers', opacity=0.8
        ))
        
        fig.update_layout(
            title="Feature Activation Comparison",
            xaxis_title="Feature Index",
            yaxis_title="Activation Value"
        )
        
        return fig, top_diff_inds
    
    def perform_feature_ablation(self, feature_acts: torch.Tensor, 
                                target_features: List[int]) -> torch.Tensor:
        """Ablate (zero out) specific features and return modified reconstructions"""
        if self.sae is None:
            raise ValueError("SAE not loaded.")
            
        ablated_features = feature_acts.clone()
        ablated_features[:, :, target_features] = 0
        
        ablated_reconstructions = self.sae.decode(ablated_features)
        
        print(f"Ablated {len(target_features)} features")
        return ablated_reconstructions
    
    def create_steering_vector(self, feature_idx: int, strength: float = 1.0) -> torch.Tensor:
        """Create a steering vector for a specific feature"""
        if self.sae is None:
            raise ValueError("SAE not loaded.")
            
        steering_vector = self.sae.W_dec[feature_idx] * strength
        print(f"Created steering vector for feature {feature_idx} with strength {strength}")
        
        return steering_vector
    
    def apply_steering(self, activations: torch.Tensor, 
                      steering_vectors: List[Tuple[int, float]]) -> torch.Tensor:
        """Apply multiple steering vectors to activations"""
        if self.sae is None:
            raise ValueError("SAE not loaded.")
            
        modified_activations = activations.clone()
        
        for feature_idx, strength in steering_vectors:
            steering_vector = self.create_steering_vector(feature_idx, strength)
            modified_activations += steering_vector.unsqueeze(0).unsqueeze(0)
            
        print(f"Applied {len(steering_vectors)} steering vectors")
        return modified_activations
    
    def save_analysis_results(self, results: Dict, filename: str = "sae_analysis_results.pt"):
        """Save analysis results to disk"""
        output_path = Path(self.config.analysis_output_dir) / filename
        
        # Convert to CPU and save
        cpu_results = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                      for k, v in results.items()}
        
        torch.save(cpu_results, output_path)
        print(f"Results saved to: {output_path}")
        
        return output_path
    
    def generate_analysis_report(self, results: Dict, top_features: Tuple[torch.Tensor, torch.Tensor],
                               output_file: str = "analysis_report.md"):
        """Generate a comprehensive analysis report"""
        output_path = Path(self.config.analysis_output_dir) / output_file
        
        with open(output_path, 'w') as f:
            f.write("# SAE Activation Analysis Report\n\n")
            
            f.write(f"## Configuration\n")
            f.write(f"- SAE Release: {self.config.sae_release}\n")
            f.write(f"- SAE ID: {self.config.sae_id}\n")
            f.write(f"- Device: {self.config.device}\n")
            f.write(f"- Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Activation Statistics\n")
            f.write(f"- Original Shape: {results['original_activations'].shape}\n")
            f.write(f"- Feature Activations Shape: {results['feature_activations'].shape}\n")
            f.write(f"- Average Sparsity (L0): {results['l0_norm'].mean().item():.2f}\n")
            f.write(f"- Average Reconstruction MSE: {results['reconstruction_error'].mean().item():.6f}\n\n")
            
            f.write(f"## Top {len(top_features[0])} Features\n")
            values, indices = top_features
            for i, (val, idx) in enumerate(zip(values, indices)):
                f.write(f"{i+1}. Feature {idx}: {val:.4f}\n")
                
                # Add Neuronpedia link if available
                url = self.get_neuronpedia_dashboard_url(int(idx))
                f.write(f"   - [View in Neuronpedia]({url})\n")
            
            f.write(f"\n## Analysis Files\n")
            f.write(f"- Full results: sae_analysis_results.pt\n")
            f.write(f"- Visualizations: Generated in Jupyter/Python environment\n")
        
        print(f"Analysis report saved to: {output_path}")
        return output_path


def create_example_workflow():
    """Create an example workflow script demonstrating the full pipeline"""
    example_code = '''
# Example SAE Analysis Workflow
from sae_analysis import SAEActivationAnalyzer, SAEAnalysisConfig

# Initialize analyzer
config = SAEAnalysisConfig(
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    top_k_features=15
)
analyzer = SAEActivationAnalyzer(config)

# Discover and load SAE
analyzer.discover_available_saes()
matching_saes = analyzer.find_matching_saes("gpt2", "resid")
sae = analyzer.load_sae()

# Load and process your pre-computed activations
activations = analyzer.load_activations("path/to/your/activations.pt")
results = analyzer.process_activations(activations)

# Find top features
values, indices = analyzer.find_top_features(results['feature_activations'])

# Download feature explanations and search
explanations = analyzer.download_feature_explanations()
bible_features = analyzer.search_features_by_description("bible")

# Visualize results
fig = analyzer.visualize_feature_activations(results['feature_activations'])
fig.show()

# Display Neuronpedia dashboards for top features
for i, idx in enumerate(indices[:5]):
    print(f"\\nFeature {idx}:")
    dashboard = analyzer.display_feature_dashboard(int(idx))
    if dashboard:
        display(dashboard)

# Advanced analysis: Feature ablation
target_features = [int(idx) for idx in indices[:3]]
ablated_reconstructions = analyzer.perform_feature_ablation(
    results['feature_activations'], target_features
)

# Advanced analysis: Steering
steering_vectors = [(int(indices[0]), 2.0), (int(indices[1]), -1.5)]
steered_activations = analyzer.apply_steering(activations, steering_vectors)

# Save results and generate report
analyzer.save_analysis_results(results)
analyzer.generate_analysis_report(results, (values, indices))

print("Analysis complete! Check the sae_analysis_outputs/ directory for results.")
'''
    
    with open("/Users/ivanculo/Desktop/Projects/turn_point/example_sae_workflow.py", 'w') as f:
        f.write(example_code)
        
    return "/Users/ivanculo/Desktop/Projects/turn_point/example_sae_workflow.py"


if __name__ == "__main__":
    # Example usage demonstration
    print("SAE Analysis Module - Usage Example:")
    print("1. Import the analyzer: from sae_analysis import SAEActivationAnalyzer")
    print("2. Create configuration: config = SAEAnalysisConfig()")
    print("3. Initialize analyzer: analyzer = SAEActivationAnalyzer(config)")
    print("4. Follow the workflow in example_sae_workflow.py")
    
    # Create example workflow file
    example_path = create_example_workflow()
    print(f"\\nExample workflow created at: {example_path}")