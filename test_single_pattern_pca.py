#!/usr/bin/env python3
"""
Improved test script using PCA-based direction vectors for cognitive transformations.

This version:
1. Uses PCA to find principal components instead of simple means
2. Analyzes beginning vs later tokens for transitions
3. Implements comprehensive reconstruction experiments
4. Prepares outputs for SAE analysis
"""

import torch
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
base_path = Path("/Users/ivanculo/Desktop/Projects/turn_point")
activations_dir = base_path / "activations"
output_dir = base_path / "pca_analysis_outputs"
output_dir.mkdir(exist_ok=True)

negative_activations = torch.load(
    activations_dir / "activations_8ff00d963316212d.pt", 
    map_location=device
)
positive_activations = torch.load(
    activations_dir / "activations_e5ad16e9b3c33c9b.pt", 
    map_location=device
)
transition_activations = torch.load(
    activations_dir / "activations_332f24de2a3f82ff.pt", 
    map_location=device
)

# Load metadata
with open(base_path / "data" / "final" / "enriched_metadata.json", 'r') as f:
    metadata = json.load(f)

# Create pattern indices
pattern_indices = {}
for i, entry in enumerate(metadata):
    pattern_name = entry['bad_good_narratives_match']['cognitive_pattern_name_from_bad_good']
    if pattern_name not in pattern_indices:
        pattern_indices[pattern_name] = []
    pattern_indices[pattern_name].append(i)

# Test with first pattern and layer 17 only
first_pattern = list(pattern_indices.keys())[0]
indices = pattern_indices[first_pattern]
layer = 17

print(f"Testing pattern: {first_pattern}")
print(f"Pattern has {len(indices)} samples")

# Get activations for this pattern
neg_data = negative_activations[f'negative_layer_{layer}'][indices]
pos_data = positive_activations[f'positive_layer_{layer}'][indices]
trans_data = transition_activations[f'transition_layer_{layer}'][indices]

print(f"Full shapes - Neg: {neg_data.shape}, Pos: {pos_data.shape}, Trans: {trans_data.shape}")

def extract_token_segments(data, segment_type="all"):
    """Extract different token segments for analysis"""
    seq_len = data.shape[1]
    if segment_type == "beginning":
        # First 25% of tokens
        return data[:, :seq_len//4, :]
    elif segment_type == "later":
        # Last 25% of tokens  
        return data[:, 3*seq_len//4:, :]
    elif segment_type == "middle":
        # Middle 50% of tokens
        return data[:, seq_len//4:3*seq_len//4, :]
    else:  # "all"
        # Every 4th token to reduce memory
        return data[:, ::4, :]

def compute_pca_representation(data, n_components=10, return_pca=False):
    """Compute PCA representation of activation data"""
    # Flatten tokens
    flat_data = data.reshape(-1, data.shape[-1]).cpu().numpy()
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(flat_data)
    
    # PCA
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(data_scaled)
    
    # Get principal component (first component)
    principal_component = torch.tensor(pca.components_[0], dtype=torch.float32)
    
    # Project original data onto principal component
    projected_data = torch.tensor(data_scaled @ pca.components_[0].T, dtype=torch.float32)
    canonical_representation = projected_data.mean() * principal_component
    
    if return_pca:
        return canonical_representation, principal_component, pca, scaler, projected_data
    else:
        return canonical_representation, principal_component

print("="*60)
print("COMPUTING PCA-BASED DIRECTION VECTORS")
print("="*60)

# 1. Compute PCA representations for each cognitive state
print("\n1. Computing canonical representations using PCA...")

# Full sequence analysis
neg_canonical, neg_pc = compute_pca_representation(extract_token_segments(neg_data, "all"))
pos_canonical, pos_pc = compute_pca_representation(extract_token_segments(pos_data, "all"))
trans_canonical, trans_pc = compute_pca_representation(extract_token_segments(trans_data, "all"))

print(f"Canonical representation stats:")
print(f"  Negative: {neg_canonical.shape}, norm={torch.norm(neg_canonical):.4f}")
print(f"  Positive: {pos_canonical.shape}, norm={torch.norm(pos_canonical):.4f}")
print(f"  Transition: {trans_canonical.shape}, norm={torch.norm(trans_canonical):.4f}")

# 2. Transition temporal analysis (beginning vs later tokens)
print("\n2. Computing transition temporal analysis...")

trans_begin_canonical, trans_begin_pc = compute_pca_representation(
    extract_token_segments(trans_data, "beginning")
)
trans_later_canonical, trans_later_pc = compute_pca_representation(
    extract_token_segments(trans_data, "later")
)

print(f"Transition temporal analysis:")
print(f"  Beginning: norm={torch.norm(trans_begin_canonical):.4f}")
print(f"  Later: norm={torch.norm(trans_later_canonical):.4f}")

# 3. Compute direction vectors
print("\n3. Computing direction vectors...")

# Core direction vectors as requested
recovery_direction = pos_canonical - neg_canonical
therapeutic_direction = trans_canonical - neg_canonical  
change_process_direction = trans_canonical - neg_canonical  # Same as therapeutic for now

# Additional temporal directions
temporal_transition_direction = trans_later_canonical - trans_begin_canonical
beginning_therapeutic_direction = trans_begin_canonical - neg_canonical
later_therapeutic_direction = trans_later_canonical - neg_canonical

print(f"Direction vector norms:")
print(f"  Recovery (pos - neg): {torch.norm(recovery_direction):.4f}")
print(f"  Therapeutic (trans - neg): {torch.norm(therapeutic_direction):.4f}")
print(f"  Temporal transition (later - begin): {torch.norm(temporal_transition_direction):.4f}")

# 4. Direction vector analysis
print(f"\nDirection vector similarities:")
cos_sim_recovery_therapeutic = cosine_similarity(
    recovery_direction.unsqueeze(0).numpy(),
    therapeutic_direction.unsqueeze(0).numpy()
)[0, 0]
print(f"  Recovery ↔ Therapeutic: {cos_sim_recovery_therapeutic:.4f}")

cos_sim_recovery_temporal = cosine_similarity(
    recovery_direction.unsqueeze(0).numpy(), 
    temporal_transition_direction.unsqueeze(0).numpy()
)[0, 0]
print(f"  Recovery ↔ Temporal: {cos_sim_recovery_temporal:.4f}")

print("="*60)
print("RECONSTRUCTION EXPERIMENTS")
print("="*60)

# Test reconstruction experiments as requested
print("\n1. Testing: depressive + recovery_direction ≈ positive")
reconstructed_positive = neg_canonical + recovery_direction
pos_reconstruction_similarity = cosine_similarity(
    reconstructed_positive.unsqueeze(0).numpy(),
    pos_canonical.unsqueeze(0).numpy()
)[0, 0]
print(f"   Similarity: {pos_reconstruction_similarity:.6f}")

print("\n2. Testing: depressive + therapeutic_direction ≈ transition")
reconstructed_transition = neg_canonical + therapeutic_direction
trans_reconstruction_similarity = cosine_similarity(
    reconstructed_transition.unsqueeze(0).numpy(),
    trans_canonical.unsqueeze(0).numpy()
)[0, 0]
print(f"   Similarity: {trans_reconstruction_similarity:.6f}")

print("\n3. Testing: depressive + transition_vector lands closer to transition")
# Use the actual difference vector from our data
actual_transition_vector = trans_canonical - neg_canonical
reconstructed_via_transition = neg_canonical + actual_transition_vector
transition_closeness = cosine_similarity(
    reconstructed_via_transition.unsqueeze(0).numpy(),
    trans_canonical.unsqueeze(0).numpy()
)[0, 0]
print(f"   Similarity to transition: {transition_closeness:.6f}")

# Additional test: Can we get from transition to positive?
print("\n4. Testing: transition + (positive - transition) ≈ positive")
transition_to_positive_direction = pos_canonical - trans_canonical
reconstructed_pos_from_trans = trans_canonical + transition_to_positive_direction
pos_from_trans_similarity = cosine_similarity(
    reconstructed_pos_from_trans.unsqueeze(0).numpy(),
    pos_canonical.unsqueeze(0).numpy()
)[0, 0]
print(f"   Similarity: {pos_from_trans_similarity:.6f}")

print("="*60)
print("TEMPORAL TRANSITION ANALYSIS")
print("="*60)

# Test temporal progression in transitions
print("\n1. Beginning transition reconstruction:")
reconstructed_begin_trans = neg_canonical + beginning_therapeutic_direction
begin_trans_similarity = cosine_similarity(
    reconstructed_begin_trans.unsqueeze(0).numpy(),
    trans_begin_canonical.unsqueeze(0).numpy()
)[0, 0]
print(f"   Neg + begin_therapeutic ≈ begin_transition: {begin_trans_similarity:.6f}")

print("\n2. Later transition reconstruction:")
reconstructed_later_trans = neg_canonical + later_therapeutic_direction  
later_trans_similarity = cosine_similarity(
    reconstructed_later_trans.unsqueeze(0).numpy(),
    trans_later_canonical.unsqueeze(0).numpy()
)[0, 0]
print(f"   Neg + later_therapeutic ≈ later_transition: {later_trans_similarity:.6f}")

print("\n3. Temporal progression test:")
# Can we go from beginning transition to later transition?
begin_to_later_direction = trans_later_canonical - trans_begin_canonical
reconstructed_later_from_begin = trans_begin_canonical + begin_to_later_direction
temporal_progression_similarity = cosine_similarity(
    reconstructed_later_from_begin.unsqueeze(0).numpy(),
    trans_later_canonical.unsqueeze(0).numpy()
)[0, 0]
print(f"   Begin_trans + temporal_direction ≈ later_trans: {temporal_progression_similarity:.6f}")

print("="*60)
print("PREPARING SAE ANALYSIS OUTPUTS")
print("="*60)

# Prepare comprehensive data for SAE analysis
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Original activations (for baseline SAE analysis)
sae_analysis_data = {
    'original_activations': {
        'negative': neg_data.cpu(),
        'positive': pos_data.cpu(), 
        'transition': trans_data.cpu()
    },
    
    # 2. Canonical representations from PCA
    'canonical_representations': {
        'negative': neg_canonical.cpu(),
        'positive': pos_canonical.cpu(),
        'transition': trans_canonical.cpu(),
        'transition_beginning': trans_begin_canonical.cpu(),
        'transition_later': trans_later_canonical.cpu()
    },
    
    # 3. Direction vectors
    'direction_vectors': {
        'recovery_direction': recovery_direction.cpu(),
        'therapeutic_direction': therapeutic_direction.cpu(),
        'change_process_direction': change_process_direction.cpu(),
        'temporal_transition_direction': temporal_transition_direction.cpu(),
        'beginning_therapeutic_direction': beginning_therapeutic_direction.cpu(),
        'later_therapeutic_direction': later_therapeutic_direction.cpu(),
        'transition_to_positive_direction': transition_to_positive_direction.cpu()
    },
    
    # 4. Reconstructed states (for testing SAE feature differences)
    'reconstructed_states': {
        'reconstructed_positive': reconstructed_positive.cpu(),
        'reconstructed_transition': reconstructed_transition.cpu(),
        'reconstructed_positive_from_transition': reconstructed_pos_from_trans.cpu(),
        'reconstructed_later_transition': reconstructed_later_trans.cpu()
    },
    
    # 5. Analysis metrics
    'analysis_metrics': {
        'reconstruction_similarities': {
            'negative_to_positive': float(pos_reconstruction_similarity),
            'negative_to_transition': float(trans_reconstruction_similarity),
            'transition_to_positive': float(pos_from_trans_similarity),
            'temporal_progression': float(temporal_progression_similarity)
        },
        'direction_similarities': {
            'recovery_therapeutic': float(cos_sim_recovery_therapeutic),
            'recovery_temporal': float(cos_sim_recovery_temporal)
        }
    },
    
    # 6. Metadata
    'metadata': {
        'cognitive_pattern': first_pattern,
        'layer': layer,
        'num_samples': len(indices),
        'timestamp': timestamp,
        'device_used': str(device)
    }
}

# Save data for SAE analysis
sae_output_path = output_dir / f"sae_analysis_data_{first_pattern}_{timestamp}.pt"
torch.save(sae_analysis_data, sae_output_path)

# Create analysis summary for Jupyter notebook
analysis_summary = {
    'best_reconstruction_experiment': 'negative_to_transition' if trans_reconstruction_similarity > pos_reconstruction_similarity else 'negative_to_positive',
    'best_similarity_score': max(pos_reconstruction_similarity, trans_reconstruction_similarity),
    'temporal_progression_works': temporal_progression_similarity > 0.8,
    'direction_vector_alignment': cos_sim_recovery_therapeutic,
    'recommended_sae_analysis': {
        'compare_features': ['negative', 'positive', 'transition'],
        'test_directions': ['recovery_direction', 'therapeutic_direction'],
        'reconstruction_tests': ['reconstructed_positive', 'reconstructed_transition']
    }
}

# Save summary
summary_path = output_dir / f"analysis_summary_{first_pattern}_{timestamp}.json"
with open(summary_path, 'w') as f:
    # Convert tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json.dump(convert_for_json(analysis_summary), f, indent=2)

print(f"\n✅ Analysis completed successfully!")
print(f"📊 SAE analysis data saved to: {sae_output_path}")
print(f"📋 Summary saved to: {summary_path}")

print(f"\n🎯 Key Results:")
print(f"   Best reconstruction: {analysis_summary['best_reconstruction_experiment']} (similarity: {analysis_summary['best_similarity_score']:.4f})")
print(f"   Temporal progression works: {analysis_summary['temporal_progression_works']}")
print(f"   Direction alignment: {analysis_summary['direction_vector_alignment']:.4f}")

print(f"\n🔬 Ready for SAE Analysis!")
print(f"   Load this data in SAE_Analysis_Interactive.ipynb:")
print(f"   data = torch.load('{sae_output_path}')")

print(f"\n📋 Recommended SAE experiments:")
print(f"   1. Compare features between negative/positive/transition states")
print(f"   2. Test if direction vectors activate different SAE features")
print(f"   3. Analyze feature differences in reconstructed vs original states")
print(f"   4. Study temporal progression features (beginning vs later transitions)")
