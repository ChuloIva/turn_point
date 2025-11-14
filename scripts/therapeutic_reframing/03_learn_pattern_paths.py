#!/usr/bin/env python3
"""
Phase 2.1: Learn Multi-Layer Paths for Pattern-Specific Reframing

Creates 3-landmark paths (negative → transformed → positive) using:
- LandmarkPath (piecewise slerp)
- ParametricCurvePath (Bezier)
- TangentVectorFieldPath (tangent field)

For all 8 strategic layers across all cognitive patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from nnsight_selfie.semantic_path_learning import (
    LandmarkPath,
    ParametricCurvePath,
    TangentVectorFieldPath,
    MultiLayerLandmarkPath,
    MultiLayerParametricCurvePath,
    MultiLayerTangentVectorFieldPath
)

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import (
    load_activation_index,
    load_pattern_activations,
    get_layer_indices
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
PATHS_DIR = PROJECT_ROOT / "learned_paths/therapeutic_reframing/pattern_specific"

# Configuration
ALPHAS_3_LANDMARK = [0.0, 0.5, 1.0]  # negative, transformed, positive

# Single-layer vs Multi-layer mode
USE_SINGLE_LAYER = True  # Set to False for multi-layer paths
SINGLE_LAYER_INDEX = 11  # Which layer to use for single-layer paths (only used if USE_SINGLE_LAYER=True) 
#   - Layer 1 - Very early layer
#   - Layer 5 - Early layer
#   - Layer 7 - Early-middle layer
#   - Layer 11 - Middle layer
#   - Layer 15 - Middle-late layer
#   - Layer 22 - Late layer
#   - Layer 27 - Very late layer
#   - Layer 29 - Final layer (out of 30 total)


def create_landmark_path(landmarks: Dict[int, List], layer_indices: List[int],
                        metadata: Dict, single_layer: bool = False,
                        single_layer_idx: int = None):
    """
    Create landmark path from 3 landmarks per layer.

    Args:
        landmarks: Dict mapping layer_idx -> [neg, trans, pos] vectors
        layer_indices: List of layer indices
        metadata: Metadata dict
        single_layer: If True, create single-layer path; else multi-layer
        single_layer_idx: Which layer to use for single-layer mode

    Returns:
        LandmarkPath if single_layer=True, else MultiLayerLandmarkPath
    """
    if single_layer:
        return LandmarkPath(
            landmarks=landmarks[single_layer_idx],
            alphas=ALPHAS_3_LANDMARK,
            metadata=metadata.copy()
        )
    else:
        layer_paths = {}
        for layer_idx in layer_indices:
            layer_paths[layer_idx] = LandmarkPath(
                landmarks=landmarks[layer_idx],
                alphas=ALPHAS_3_LANDMARK,
                metadata=metadata.copy()
            )
        return MultiLayerLandmarkPath(
            layer_paths=layer_paths,
            layer_indices=layer_indices,
            metadata=metadata
        )


def create_parametric_path(landmarks: Dict[int, List], layer_indices: List[int],
                          metadata: Dict, single_layer: bool = False,
                          single_layer_idx: int = None):
    """
    Create parametric curve (Bezier) from 3 landmarks.

    Args:
        landmarks: Dict mapping layer_idx -> [neg, trans, pos] vectors
        layer_indices: List of layer indices
        metadata: Metadata dict
        single_layer: If True, create single-layer path; else multi-layer
        single_layer_idx: Which layer to use for single-layer mode

    Returns:
        ParametricCurvePath if single_layer=True, else MultiLayerParametricCurvePath
    """
    if single_layer:
        path = ParametricCurvePath.fit_from_landmarks(
            landmarks=landmarks[single_layer_idx],
            alphas=ALPHAS_3_LANDMARK,
            curve_type="bezier"
        )
        path.metadata.update(metadata)
        return path
    else:
        layer_paths = {}
        for layer_idx in layer_indices:
            layer_paths[layer_idx] = ParametricCurvePath.fit_from_landmarks(
                landmarks=landmarks[layer_idx],
                alphas=ALPHAS_3_LANDMARK,
                curve_type="bezier"
            )
            layer_paths[layer_idx].metadata.update(metadata)
        return MultiLayerParametricCurvePath(
            layer_paths=layer_paths,
            layer_indices=layer_indices,
            metadata=metadata
        )


def create_tangent_path(landmarks: Dict[int, List], layer_indices: List[int],
                       metadata: Dict, single_layer: bool = False,
                       single_layer_idx: int = None):
    """
    Create tangent vector field path from 3 landmarks.

    Args:
        landmarks: Dict mapping layer_idx -> [neg, trans, pos] vectors
        layer_indices: List of layer indices
        metadata: Metadata dict
        single_layer: If True, create single-layer path; else multi-layer
        single_layer_idx: Which layer to use for single-layer mode

    Returns:
        TangentVectorFieldPath if single_layer=True, else MultiLayerTangentVectorFieldPath
    """
    if single_layer:
        path = TangentVectorFieldPath.fit_from_landmarks(
            landmarks=landmarks[single_layer_idx],
            alphas=ALPHAS_3_LANDMARK
        )
        path.metadata.update(metadata)
        return path
    else:
        layer_paths = {}
        for layer_idx in layer_indices:
            layer_paths[layer_idx] = TangentVectorFieldPath.fit_from_landmarks(
                landmarks=landmarks[layer_idx],
                alphas=ALPHAS_3_LANDMARK
            )
            layer_paths[layer_idx].metadata.update(metadata)
        return MultiLayerTangentVectorFieldPath(
            layer_paths=layer_paths,
            layer_indices=layer_indices,
            metadata=metadata
        )


def process_pattern(pattern_name: str, pattern_slug: str, df_pattern: pd.DataFrame,
                   layer_indices: List[int], single_layer: bool, single_layer_idx: int = None) -> Dict:
    """
    Learn paths for all examples in a cognitive pattern.

    Args:
        pattern_name: Name of the cognitive pattern
        pattern_slug: Slugified pattern name for file paths
        df_pattern: DataFrame with examples for this pattern
        layer_indices: List of layer indices
        single_layer: If True, create single-layer paths
        single_layer_idx: Which layer to use for single-layer mode

    Returns:
        Stats dict with counts
    """
    # Load activations
    print(f"\n  Loading activations for {pattern_name}...")
    activations = load_pattern_activations(pattern_slug, device='cpu')

    # Determine which layer to use for n_examples check
    check_layer = single_layer_idx if single_layer else layer_indices[0]
    n_examples = activations['negative'][check_layer].shape[0]
    print(f"  Processing {n_examples} examples...")

    # Create output directory
    pattern_dir = PATHS_DIR / pattern_slug
    pattern_dir.mkdir(parents=True, exist_ok=True)

    stats = {'landmark': 0, 'parametric': 0, 'tangent': 0}

    for ex_idx in tqdm(range(n_examples), desc=f"  {pattern_name}"):
        # Get example ID
        example_id = df_pattern.iloc[ex_idx]['example_id']

        # Build landmarks dict: layer_idx -> [neg, trans, pos]
        # Always build for all layers, but only the needed layer will be used in single-layer mode
        landmarks = {}
        layers_to_load = [single_layer_idx] if single_layer else layer_indices
        for layer_idx in layers_to_load:
            neg_vec = activations['negative'][layer_idx][ex_idx]
            trans_vec = activations['transformed'][layer_idx][ex_idx]
            pos_vec = activations['positive'][layer_idx][ex_idx]

            landmarks[layer_idx] = [neg_vec, trans_vec, pos_vec]

        # Metadata
        metadata = {
            'example_id': example_id,
            'pattern_name': pattern_name,
            'pattern_type': df_pattern.iloc[ex_idx]['pattern_type']
        }
        if single_layer:
            metadata['layer_index'] = single_layer_idx

        # Create 3 path types
        landmark_path = create_landmark_path(landmarks, layer_indices, metadata,
                                            single_layer, single_layer_idx)
        parametric_path = create_parametric_path(landmarks, layer_indices, metadata,
                                                single_layer, single_layer_idx)
        tangent_path = create_tangent_path(landmarks, layer_indices, metadata,
                                          single_layer, single_layer_idx)

        # Save paths
        with open(pattern_dir / f"{example_id}_landmark.pkl", 'wb') as f:
            pickle.dump(landmark_path, f)
        stats['landmark'] += 1

        with open(pattern_dir / f"{example_id}_parametric.pkl", 'wb') as f:
            pickle.dump(parametric_path, f)
        stats['parametric'] += 1

        with open(pattern_dir / f"{example_id}_tangent.pkl", 'wb') as f:
            pickle.dump(tangent_path, f)
        stats['tangent'] += 1

    print(f"  ✅ Saved {stats['landmark'] + stats['parametric'] + stats['tangent']} paths")
    return stats


def main():
    print("=" * 80)
    print("PHASE 2.1: Pattern-Specific Path Learning")
    print("=" * 80)

    # Load dataset
    print(f"\n📥 Loading dataset...")
    df = pd.read_csv(DATA_DIR / "pattern_metadata.csv")
    print(f"✅ Loaded {len(df)} examples from {df['pattern_name'].nunique()} patterns")

    # Load train split
    with open(DATA_DIR / "train_test_split.json", 'r') as f:
        splits = json.load(f)

    # Get layer indices
    layer_indices = get_layer_indices()
    print(f"\n📊 Strategic layers: {layer_indices}")

    # Print configuration
    print(f"\n⚙️  Configuration:")
    print(f"  Mode: {'Single-layer' if USE_SINGLE_LAYER else 'Multi-layer'}")
    if USE_SINGLE_LAYER:
        print(f"  Target layer: {SINGLE_LAYER_INDEX}")
        if SINGLE_LAYER_INDEX not in layer_indices:
            print(f"\n⚠️  WARNING: SINGLE_LAYER_INDEX {SINGLE_LAYER_INDEX} not in layer_indices!")
            print(f"  Available layers: {layer_indices}")
            return

    # Process each pattern
    print(f"\n🔄 Learning paths for all patterns...")
    print(f"  3 landmarks per example: negative → transformed → positive")
    print(f"  3 path representations: Landmark, Parametric, Tangent")
    if USE_SINGLE_LAYER:
        print(f"  Single layer: {SINGLE_LAYER_INDEX}")
    else:
        print(f"  {len(layer_indices)} layers per path")

    total_stats = {'landmark': 0, 'parametric': 0, 'tangent': 0}
    pattern_stats = {}

    for pattern_name in sorted(df['pattern_name'].unique()):
        # Get training examples only
        train_ids = set(splits['train'][pattern_name])
        df_pattern = df[df['example_id'].isin(train_ids)]

        pattern_slug = pattern_name.lower().replace(' ', '_').replace('&', 'and')

        # Process pattern
        stats = process_pattern(pattern_name, pattern_slug, df_pattern, layer_indices,
                              USE_SINGLE_LAYER, SINGLE_LAYER_INDEX)

        pattern_stats[pattern_name] = stats
        for key in stats:
            total_stats[key] += stats[key]

    # Save summary
    summary = {
        'mode': 'single_layer' if USE_SINGLE_LAYER else 'multi_layer',
        'layer_indices': layer_indices if not USE_SINGLE_LAYER else [SINGLE_LAYER_INDEX],
        'single_layer_index': SINGLE_LAYER_INDEX if USE_SINGLE_LAYER else None,
        'n_landmarks': len(ALPHAS_3_LANDMARK),
        'path_types': ['landmark', 'parametric', 'tangent'],
        'total_paths': sum(total_stats.values()),
        'paths_per_type': total_stats,
        'patterns': pattern_stats
    }

    with open(PATHS_DIR.parent / "metadata" / "pattern_specific_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n" + "=" * 80)
    print(f"✅ Phase 2.1 Complete!")
    print(f"=" * 80)
    print(f"\n📊 Summary:")
    print(f"  Patterns processed: {len(pattern_stats)}")
    print(f"  Total paths created: {sum(total_stats.values())}")
    print(f"    Landmark paths: {total_stats['landmark']}")
    print(f"    Parametric paths: {total_stats['parametric']}")
    print(f"    Tangent paths: {total_stats['tangent']}")
    print(f"\n📁 Paths saved to: {PATHS_DIR}")


if __name__ == "__main__":
    main()
