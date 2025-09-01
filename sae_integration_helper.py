#!/usr/bin/env python3
"""
Helper functions for integrating PCA-based cognitive transformation analysis with SAE analysis.
This module provides easy loading and preprocessing of the PCA analysis results for SAE exploration.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CognitiveSAEIntegration:
    def __init__(self, analysis_data_path: str):
        """
        Initialize with path to saved PCA analysis data
        
        Args:
            analysis_data_path: Path to the .pt file saved by test_single_pattern_pca.py
        """
        self.data_path = Path(analysis_data_path)
        self.data = torch.load(self.data_path, map_location='cpu')
        
    def get_activations_for_sae(self, state_type: str, format_for_sae: bool = True) -> torch.Tensor:
        """
        Get activations formatted for SAE analysis
        
        Args:
            state_type: 'negative', 'positive', 'transition', or 'reconstructed_*'
            format_for_sae: If True, reshape to [batch_size * seq_len, hidden_dim]
        """
        if state_type in self.data['original_activations']:
            activations = self.data['original_activations'][state_type]
        elif state_type in self.data['reconstructed_states']:
            # For reconstructed states, we need to expand to proper shape
            canonical = self.data['reconstructed_states'][state_type]
            # Expand to match original activation shape
            original_shape = self.data['original_activations']['negative'].shape
            activations = canonical.unsqueeze(0).unsqueeze(0).expand(
                original_shape[0], original_shape[1], -1
            )
        else:
            raise ValueError(f"Unknown state type: {state_type}")
        
        if format_for_sae:
            # Flatten to [batch_size * seq_len, hidden_dim] for SAE processing
            return activations.reshape(-1, activations.shape[-1])
        else:
            return activations
    
    def get_direction_vectors(self) -> Dict[str, torch.Tensor]:
        """Get all computed direction vectors"""
        return self.data['direction_vectors']
    
    def get_canonical_representations(self) -> Dict[str, torch.Tensor]:
        """Get PCA-based canonical representations"""
        return self.data['canonical_representations']
    
    def create_sae_experiment_batch(self, include_reconstructed: bool = True) -> Dict[str, torch.Tensor]:
        """
        Create a batch of activations for comprehensive SAE analysis
        
        Returns:
            Dictionary with keys as labels and values as flattened activations ready for SAE
        """
        experiment_batch = {}
        
        # Original states
        for state in ['negative', 'positive', 'transition']:
            experiment_batch[state] = self.get_activations_for_sae(state)
        
        if include_reconstructed:
            # Reconstructed states - expand canonical representations
            for recon_key, recon_tensor in self.data['reconstructed_states'].items():
                # Create a batch by repeating the canonical representation
                batch_size = 10  # Create small batch for SAE analysis
                seq_len = 20     # Reasonable sequence length
                hidden_dim = recon_tensor.shape[0]
                
                # Expand to [batch_size, seq_len, hidden_dim]
                expanded = recon_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, hidden_dim)
                # Flatten for SAE
                experiment_batch[recon_key] = expanded.reshape(-1, hidden_dim)
        
        return experiment_batch
    
    def create_direction_intervention_data(self, base_state: str = 'negative', 
                                         direction: str = 'recovery_direction',
                                         strengths: List[float] = [0.5, 1.0, 1.5, 2.0]) -> Dict[str, torch.Tensor]:
        """
        Create data for testing direction vector interventions in SAE space
        
        Args:
            base_state: Starting state ('negative', 'positive', 'transition')
            direction: Direction vector to apply
            strengths: List of multiplication factors for the direction vector
            
        Returns:
            Dictionary with intervention labels and corresponding activations
        """
        base_activations = self.get_canonical_representations()[base_state]
        direction_vector = self.get_direction_vectors()[direction]
        
        intervention_data = {}
        intervention_data[f'{base_state}_original'] = base_activations.unsqueeze(0)  # Add batch dim
        
        for strength in strengths:
            intervened = base_activations + strength * direction_vector
            intervention_data[f'{base_state}_{direction}_strength_{strength}'] = intervened.unsqueeze(0)
        
        return intervention_data
    
    def get_analysis_summary(self) -> Dict:
        """Get the analysis summary and metrics"""
        return self.data['analysis_metrics']
    
    def get_metadata(self) -> Dict:
        """Get analysis metadata"""
        return self.data['metadata']
    
    def create_sae_analysis_report(self, sae_results: Dict) -> str:
        """
        Create a formatted report comparing SAE analysis with PCA results
        
        Args:
            sae_results: Results from SAE analysis (features, activations, etc.)
            
        Returns:
            Formatted report string
        """
        metadata = self.get_metadata()
        metrics = self.get_analysis_summary()
        
        report = f"""
# Cognitive Transformation SAE Analysis Report

## Analysis Overview
- **Cognitive Pattern**: {metadata['cognitive_pattern']}
- **Layer**: {metadata['layer']}
- **Samples**: {metadata['num_samples']}
- **Timestamp**: {metadata['timestamp']}

## PCA-Based Direction Vectors
- **Recovery Direction**: negative → positive (similarity: {metrics['reconstruction_similarities']['negative_to_positive']:.4f})
- **Therapeutic Direction**: negative → transition (similarity: {metrics['reconstruction_similarities']['negative_to_transition']:.4f})
- **Transition Progression**: temporal development (similarity: {metrics['reconstruction_similarities']['temporal_progression']:.4f})

## Direction Vector Relationships
- **Recovery ↔ Therapeutic Alignment**: {metrics['direction_similarities']['recovery_therapeutic']:.4f}
- **Recovery ↔ Temporal Alignment**: {metrics['direction_similarities']['recovery_temporal']:.4f}

## SAE Feature Analysis
(Add your SAE results here)

## Key Insights
1. **Best Reconstruction Path**: {metrics['reconstruction_similarities']}
2. **Temporal Progression**: {"Works well" if metrics['reconstruction_similarities']['temporal_progression'] > 0.8 else "Needs investigation"}
3. **Direction Coherence**: {"High" if metrics['direction_similarities']['recovery_therapeutic'] > 0.5 else "Moderate"}

## Recommendations for Further Analysis
1. Focus on features that activate differently between negative/positive states
2. Investigate features that show progression in transition sequences
3. Test direction vector steering in SAE feature space
4. Analyze feature ablation effects on reconstruction quality
"""
        return report

def load_latest_analysis(output_dir: str = "/Users/ivanculo/Desktop/Projects/turn_point/pca_analysis_outputs") -> CognitiveSAEIntegration:
    """
    Load the most recent PCA analysis data
    
    Args:
        output_dir: Directory containing PCA analysis outputs
        
    Returns:
        CognitiveSAEIntegration instance with loaded data
    """
    output_path = Path(output_dir)
    analysis_files = list(output_path.glob("sae_analysis_data_*.pt"))
    
    if not analysis_files:
        raise FileNotFoundError(f"No analysis files found in {output_dir}")
    
    # Get most recent file
    latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading latest analysis: {latest_file}")
    
    return CognitiveSAEIntegration(str(latest_file))

# Example usage functions for Jupyter notebook
def quick_sae_setup():
    """Quick setup function for Jupyter notebook"""
    try:
        integration = load_latest_analysis()
        experiment_batch = integration.create_sae_experiment_batch()
        
        print("✅ SAE integration ready!")
        print(f"🔬 Analysis: {integration.get_metadata()['cognitive_pattern']}")
        print(f"📊 Experiment batch contains {len(experiment_batch)} conditions:")
        for condition, data in experiment_batch.items():
            print(f"   - {condition}: {data.shape}")
        
        return integration, experiment_batch
    
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None, None

def print_direction_vectors(integration: CognitiveSAEIntegration):
    """Print summary of direction vectors for easy reference"""
    directions = integration.get_direction_vectors()
    
    print("📐 Direction Vectors Summary:")
    print("-" * 40)
    for name, vector in directions.items():
        norm = torch.norm(vector)
        print(f"{name:30s}: norm={norm:.4f}, shape={vector.shape}")