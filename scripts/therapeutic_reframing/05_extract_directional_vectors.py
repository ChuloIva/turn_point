#!/usr/bin/env python3
"""
Extract Directional Vectors using PCA Center Method

Computes directional vectors for therapeutic reframing patterns using the
pca_center method from repeng. Creates three types of directional vectors:
1. negative → transformed (early therapeutic intervention)
2. transformed → positive (completion of therapeutic trajectory)
3. negative → positive (full direct trajectory)

Each vector represents the semantic axis connecting two states in activation space.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, Tuple
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import get_layer_indices

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
ACTIVATIONS_DIR = PROJECT_ROOT / "activations/therapeutic_reframing/by_pattern"
OUTPUT_DIR = PROJECT_ROOT / "data/therapeutic_reframing/directional_vectors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pattern_activations(pattern_slug: str, layer_indices: list) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Load activations for a pattern across all text types.

    Returns:
        Dict with keys 'negative', 'transformed', 'positive'
        Each maps to dict of {layer_idx: activations_array}
    """
    pattern_dir = ACTIVATIONS_DIR / pattern_slug

    activations = {}
    for text_type in ['negative', 'transformed', 'positive']:
        file_path = pattern_dir / f"{text_type}_examples.npz"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing activations: {file_path}")

        data = np.load(file_path)
        activations[text_type] = {
            layer_idx: data[f'layer_{layer_idx}']
            for layer_idx in layer_indices
        }

    return activations


def compute_pca_center_vector(positive_examples: np.ndarray,
                               negative_examples: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
    """
    Compute directional vector using pca_center method.

    Based on repeng's implementation:
    1. Compute pairwise centers: (positive + negative) / 2
    2. Center both sets around midpoint
    3. Apply PCA to find principal component
    4. Sign-correct to ensure vector points toward positive pole

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
        'explained_variance_ratio': float(explained_variance)
    }

    return direction, explained_variance, stats


def extract_directional_vectors_for_pattern(pattern_slug: str,
                                            layer_indices: list) -> Dict:
    """
    Extract all directional vectors for a single pattern.

    Returns:
        Dict containing vectors and metadata
    """
    print(f"  Loading activations for {pattern_slug}...")
    activations = load_pattern_activations(pattern_slug, layer_indices)

    # Get number of examples (should be same across all types and layers)
    n_examples = activations['negative'][layer_indices[0]].shape[0]
    print(f"    {n_examples} examples loaded")

    results = {
        'pattern_slug': pattern_slug,
        'n_examples': n_examples,
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

    print(f"  Computing directional vectors...")
    for vector_name, negative_type, positive_type in vector_types:
        print(f"    {vector_name}: {negative_type} → {positive_type}")

        results['vectors'][vector_name] = {}
        results['statistics'][vector_name] = {}

        for layer_idx in layer_indices:
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


def save_pattern_results(results: Dict, output_dir: Path):
    """Save directional vectors and statistics for a pattern."""
    pattern_slug = results['pattern_slug']
    pattern_dir = output_dir / pattern_slug
    pattern_dir.mkdir(parents=True, exist_ok=True)

    # Save vectors as .npz (efficient binary format)
    vectors_to_save = {}
    for vector_type, layer_vectors in results['vectors'].items():
        for layer_key, vector in layer_vectors.items():
            key = f"{vector_type}_{layer_key}"
            vectors_to_save[key] = vector

    np.savez_compressed(
        pattern_dir / "directional_vectors.npz",
        **vectors_to_save
    )

    # Save statistics as JSON
    stats_json = {
        'pattern_slug': results['pattern_slug'],
        'n_examples': results['n_examples'],
        'layer_indices': results['layer_indices'],
        'statistics': {
            vector_type: {
                str(layer_idx): layer_stats
                for layer_idx, layer_stats in type_stats.items()
            }
            for vector_type, type_stats in results['statistics'].items()
        }
    }

    with open(pattern_dir / "statistics.json", 'w') as f:
        json.dump(stats_json, f, indent=2)

    print(f"    ✅ Saved to {pattern_dir}/")


def print_summary_statistics(all_results: list, layer_indices: list):
    """Print summary statistics across all patterns."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    vector_types = ['neg_to_trans', 'trans_to_pos', 'neg_to_pos']

    for vector_type in vector_types:
        print(f"\n{vector_type.upper().replace('_', ' ')}:")
        print("-" * 40)

        for layer_idx in layer_indices:
            # Collect stats across patterns
            explained_vars = []
            proj_diffs = []

            for results in all_results:
                stats = results['statistics'][vector_type][layer_idx]
                explained_vars.append(stats['explained_variance_ratio'])
                proj_diffs.append(stats['projection_difference'])

            avg_exp_var = np.mean(explained_vars)
            avg_proj_diff = np.mean(proj_diffs)

            print(f"  Layer {layer_idx:2d}: "
                  f"Explained variance = {avg_exp_var:.3f}, "
                  f"Projection diff = {avg_proj_diff:.3f}")


def main():
    print("=" * 80)
    print("DIRECTIONAL VECTOR EXTRACTION (PCA Center Method)")
    print("=" * 80)

    # Load dataset metadata
    print("\n📥 Loading dataset metadata...")
    df = pd.read_csv(DATA_DIR / "pattern_metadata.csv")
    patterns = sorted(df['pattern_name'].unique())
    print(f"  Found {len(patterns)} patterns")

    # Get layer indices
    layer_indices = get_layer_indices()
    print(f"\n📊 Processing layers: {layer_indices}")

    # Process each pattern
    print(f"\n🔄 Extracting directional vectors...\n")

    all_results = []

    for pattern_name in tqdm(patterns, desc="Processing patterns"):
        pattern_slug = pattern_name.lower().replace(' ', '_').replace('&', 'and')

        try:
            results = extract_directional_vectors_for_pattern(
                pattern_slug, layer_indices
            )
            all_results.append(results)
            save_pattern_results(results, OUTPUT_DIR)

        except Exception as e:
            print(f"\n❌ Error processing {pattern_name}: {e}")
            continue

    # Print summary statistics
    if all_results:
        print_summary_statistics(all_results, layer_indices)

    # Save global metadata
    print(f"\n💾 Saving global metadata...")
    global_metadata = {
        'n_patterns': len(all_results),
        'layer_indices': layer_indices,
        'vector_types': ['neg_to_trans', 'trans_to_pos', 'neg_to_pos'],
        'patterns_processed': [r['pattern_slug'] for r in all_results]
    }

    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(global_metadata, f, indent=2)

    print(f"  ✅ Saved global metadata")

    print("\n" + "=" * 80)
    print("✅ DIRECTIONAL VECTOR EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    print(f"  - Per-pattern directories with vectors.npz and statistics.json")
    print(f"  - Global metadata.json")
    print(f"\nProcessed {len(all_results)}/{len(patterns)} patterns successfully")


if __name__ == "__main__":
    main()
