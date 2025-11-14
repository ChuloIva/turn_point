#!/usr/bin/env python3
"""
Extract Universal Directional Vectors

Computes universal directional vectors by aggregating activations across ALL
cognitive patterns. This captures the common semantic axis of therapeutic
reframing that generalizes across different cognitive patterns.

Uses the same pca_center method but with activations pooled from all patterns.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import get_layer_indices

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
ACTIVATIONS_DIR = PROJECT_ROOT / "activations/therapeutic_reframing/by_pattern"
OUTPUT_DIR = PROJECT_ROOT / "data/therapeutic_reframing/universal_vectors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_pattern_activations(patterns: List[str], layer_indices: List[int]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Load and aggregate activations from all patterns.

    Returns:
        Dict with keys 'negative', 'transformed', 'positive'
        Each maps to dict of {layer_idx: activations_array} with all patterns aggregated
    """
    aggregated = {
        'negative': {layer_idx: [] for layer_idx in layer_indices},
        'transformed': {layer_idx: [] for layer_idx in layer_indices},
        'positive': {layer_idx: [] for layer_idx in layer_indices}
    }

    print("\n📥 Loading activations from all patterns...")
    for pattern_name in tqdm(patterns, desc="  Patterns"):
        pattern_slug = pattern_name.lower().replace(' ', '_').replace('&', 'and')
        pattern_dir = ACTIVATIONS_DIR / pattern_slug

        for text_type in ['negative', 'transformed', 'positive']:
            file_path = pattern_dir / f"{text_type}_examples.npz"

            if not file_path.exists():
                print(f"\n    ⚠️  Missing: {file_path}")
                continue

            data = np.load(file_path)

            for layer_idx in layer_indices:
                layer_acts = data[f'layer_{layer_idx}']
                aggregated[text_type][layer_idx].append(layer_acts)

    # Concatenate all patterns
    print("\n  Concatenating activations across patterns...")
    for text_type in aggregated.keys():
        for layer_idx in layer_indices:
            if aggregated[text_type][layer_idx]:
                aggregated[text_type][layer_idx] = np.concatenate(
                    aggregated[text_type][layer_idx], axis=0
                )
            else:
                raise ValueError(f"No activations found for {text_type} at layer {layer_idx}")

    # Print stats
    sample_layer = layer_indices[0]
    for text_type in ['negative', 'transformed', 'positive']:
        n_examples = aggregated[text_type][sample_layer].shape[0]
        print(f"    {text_type}: {n_examples} examples")

    return aggregated


def compute_pca_center_vector(positive_examples: np.ndarray,
                               negative_examples: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
    """
    Compute directional vector using pca_center method.

    Args:
        positive_examples: Shape (n_examples, hidden_dim)
        negative_examples: Shape (n_examples, hidden_dim)

    Returns:
        direction_vector: Shape (hidden_dim,)
        explained_variance: Fraction of variance explained by PC1
        stats: Dict with projection statistics
    """
    assert positive_examples.shape == negative_examples.shape, \
        "Positive and negative examples must have same shape"

    n_examples, hidden_dim = positive_examples.shape

    # Compute pairwise centers
    centers = (positive_examples + negative_examples) / 2

    # Center both sets around midpoint
    centered_positive = positive_examples - centers
    centered_negative = negative_examples - centers

    # Combine centered activations
    centered_data = np.concatenate([centered_positive, centered_negative], axis=0)

    # Apply PCA
    pca = PCA(n_components=1, whiten=False)
    pca.fit(centered_data)

    direction = pca.components_[0].astype(np.float32)
    explained_variance = pca.explained_variance_ratio_[0]

    # Sign correction: ensure positive examples project higher
    pos_projections = positive_examples @ direction
    neg_projections = negative_examples @ direction

    mean_pos_proj = pos_projections.mean()
    mean_neg_proj = neg_projections.mean()

    if mean_pos_proj < mean_neg_proj:
        # Flip direction
        direction = -direction
        mean_pos_proj, mean_neg_proj = -mean_neg_proj, -mean_pos_proj
        pos_projections = -pos_projections
        neg_projections = -neg_projections

    # Compute separation statistics
    stats = {
        'mean_positive_projection': float(mean_pos_proj),
        'mean_negative_projection': float(mean_neg_proj),
        'projection_difference': float(mean_pos_proj - mean_neg_proj),
        'std_positive_projection': float(pos_projections.std()),
        'std_negative_projection': float(neg_projections.std()),
        'explained_variance_ratio': float(explained_variance),
        'n_examples': n_examples
    }

    return direction, explained_variance, stats


def extract_universal_vectors(activations: Dict, layer_indices: List[int]) -> Dict:
    """
    Extract universal directional vectors across all patterns.

    Returns:
        Dict containing vectors and metadata
    """
    print("\n🔄 Computing universal directional vectors...")

    results = {
        'n_patterns': 'all',
        'layer_indices': layer_indices,
        'vectors': {},
        'statistics': {}
    }

    # Define the three vector types
    vector_types = [
        ('neg_to_trans', 'negative', 'transformed'),
        ('trans_to_pos', 'transformed', 'positive'),
        ('neg_to_pos', 'negative', 'positive')
    ]

    for vector_name, negative_type, positive_type in vector_types:
        print(f"\n  {vector_name}: {negative_type} → {positive_type}")

        results['vectors'][vector_name] = {}
        results['statistics'][vector_name] = {}

        for layer_idx in tqdm(layer_indices, desc="    Layers"):
            negative_acts = activations[negative_type][layer_idx]
            positive_acts = activations[positive_type][layer_idx]

            # Compute directional vector
            direction, explained_var, stats = compute_pca_center_vector(
                positive_acts, negative_acts
            )

            # Store results
            results['vectors'][vector_name][f'layer_{layer_idx}'] = direction
            results['statistics'][vector_name][layer_idx] = stats

    return results


def save_universal_vectors(results: Dict):
    """Save universal vectors and statistics."""
    print("\n💾 Saving universal vectors...")

    # Save vectors as .npz
    vectors_to_save = {}
    for vector_type, layer_vectors in results['vectors'].items():
        for layer_key, vector in layer_vectors.items():
            key = f"{vector_type}_{layer_key}"
            vectors_to_save[key] = vector

    np.savez_compressed(
        OUTPUT_DIR / "universal_vectors.npz",
        **vectors_to_save
    )

    # Save statistics as JSON
    stats_json = {
        'n_patterns': results['n_patterns'],
        'layer_indices': results['layer_indices'],
        'statistics': {
            vector_type: {
                str(layer_idx): layer_stats
                for layer_idx, layer_stats in type_stats.items()
            }
            for vector_type, type_stats in results['statistics'].items()
        }
    }

    with open(OUTPUT_DIR / "statistics.json", 'w') as f:
        json.dump(stats_json, f, indent=2)

    print(f"  ✅ Saved to {OUTPUT_DIR}/")


def main():
    print("=" * 80)
    print("UNIVERSAL DIRECTIONAL VECTOR EXTRACTION")
    print("=" * 80)

    # Load dataset metadata
    print("\n📥 Loading dataset metadata...")
    df = pd.read_csv(DATA_DIR / "pattern_metadata.csv")
    patterns = sorted(df['pattern_name'].unique())
    print(f"  Found {len(patterns)} patterns")

    # Get layer indices
    layer_indices = get_layer_indices()
    print(f"\n📊 Processing layers: {layer_indices}")

    # Load all activations
    activations = load_all_pattern_activations(patterns, layer_indices)

    # Extract universal vectors
    vector_results = extract_universal_vectors(activations, layer_indices)
    save_universal_vectors(vector_results)

    # Print statistics
    print("\n" + "=" * 80)
    print("UNIVERSAL VECTOR STATISTICS")
    print("=" * 80)

    for vector_type in ['neg_to_trans', 'trans_to_pos', 'neg_to_pos']:
        print(f"\n{vector_type.upper().replace('_', ' ')}:")
        print("-" * 40)
        for layer_idx in layer_indices:
            stats = vector_results['statistics'][vector_type][layer_idx]
            print(f"  Layer {layer_idx:2d}: "
                  f"Explained var = {stats['explained_variance_ratio']:.3f}, "
                  f"Proj diff = {stats['projection_difference']:.3f}, "
                  f"N = {stats['n_examples']}")

    print("\n" + "=" * 80)
    print("✅ UNIVERSAL VECTOR EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Vectors saved to: {OUTPUT_DIR}")
    print(f"\nNext step: Run 09_interpret_universal_vector.py to interpret at high intensities")


if __name__ == "__main__":
    main()