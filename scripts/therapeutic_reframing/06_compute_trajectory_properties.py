#!/usr/bin/env python3
"""
Phase 4: Compute Semantic Trajectory Analysis

Analyzes geometric properties of all learned paths:
- Curvature profiles (mean, max, variance)
- Distance metrics (semantic distance, path length, geodesic efficiency)
- Reframing difficulty scores
- Layer importance
- Landmark accuracy

Generates comprehensive CSV datasets for visualization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import (
    get_layer_indices,
    load_pattern_activations
)
from path_analysis import (
    compute_comprehensive_path_analysis,
    compute_reframing_difficulty
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
PATHS_DIR = PROJECT_ROOT / "learned_paths/therapeutic_reframing/pattern_specific"
OUTPUT_DIR = PROJECT_ROOT / "analysis/therapeutic_reframing/geometric_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
FINE_ALPHAS = np.linspace(0.0, 1.0, 50).tolist()  # 50 points for fine-grained analysis


def analyze_pattern_paths(pattern_name: str, pattern_slug: str,
                         df_pattern: pd.DataFrame, layer_indices: List[int],
                         path_type: str = 'landmark') -> List[Dict]:
    """
    Analyze all paths for one cognitive pattern.

    Returns:
        List of analysis results (one per example)
    """
    pattern_dir = PATHS_DIR / pattern_slug

    # Load activations for landmark accuracy
    activations = load_pattern_activations(pattern_slug, device='cpu')

    results = []

    for idx, row in df_pattern.iterrows():
        example_id = row['example_id']

        # Load path
        path_file = pattern_dir / f"{example_id}_{path_type}.pkl"
        if not path_file.exists():
            continue

        with open(path_file, 'rb') as f:
            path = pickle.load(f)

        # Get transformed landmark vectors for accuracy check
        ex_idx = df_pattern[df_pattern['example_id'] == example_id].index[0] - df_pattern.index[0]
        landmark_vecs = {
            layer_idx: activations['transformed'][layer_idx][ex_idx]
            for layer_idx in layer_indices
        }

        # Comprehensive analysis
        analysis = compute_comprehensive_path_analysis(
            path, layer_indices, FINE_ALPHAS, landmark_vecs
        )

        # Add metadata
        result = {
            'example_id': example_id,
            'pattern_name': pattern_name,
            'pattern_type': row['pattern_type'],
            'path_type': path_type,
            **analysis
        }

        results.append(result)

    return results


def flatten_analysis_for_csv(analyses: List[Dict]) -> pd.DataFrame:
    """Convert nested analysis dicts to flat DataFrame for CSV export."""
    flat_rows = []

    for analysis in analyses:
        # Per-layer metrics
        for layer_idx in analysis['per_layer'].keys():
            layer_data = analysis['per_layer'][layer_idx]

            row = {
                'example_id': analysis['example_id'],
                'pattern_name': analysis['pattern_name'],
                'pattern_type': analysis['pattern_type'],
                'path_type': analysis['path_type'],
                'layer_idx': layer_idx,

                # Curvature stats
                'mean_curvature': layer_data['curvature_stats']['mean_curvature'],
                'max_curvature': layer_data['curvature_stats']['max_curvature'],
                'variance_curvature': layer_data['curvature_stats']['variance_curvature'],
                'total_curvature': layer_data['curvature_stats']['total_curvature'],

                # Distance metrics
                'semantic_distance': layer_data['distance_metrics']['semantic_distance'],
                'path_length': layer_data['distance_metrics']['path_length'],
                'geodesic_efficiency': layer_data['distance_metrics']['geodesic_efficiency'],

                # Landmark accuracy
                'landmark_accuracy': layer_data['landmark_accuracy']
            }

            flat_rows.append(row)

    return pd.DataFrame(flat_rows)


def compute_difficulty_scores(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized difficulty scores.

    Returns:
        DataFrame with difficulty scores per example and layer
    """
    # Normalize metrics per layer
    difficulty_rows = []

    for layer_idx in df_analysis['layer_idx'].unique():
        df_layer = df_analysis[df_analysis['layer_idx'] == layer_idx].copy()

        # Normalize to [0, 1]
        df_layer['norm_semantic_distance'] = (
            (df_layer['semantic_distance'] - df_layer['semantic_distance'].min()) /
            (df_layer['semantic_distance'].max() - df_layer['semantic_distance'].min() + 1e-10)
        )

        df_layer['norm_mean_curvature'] = (
            (df_layer['mean_curvature'] - df_layer['mean_curvature'].min()) /
            (df_layer['mean_curvature'].max() - df_layer['mean_curvature'].min() + 1e-10)
        )

        df_layer['norm_max_curvature'] = (
            (df_layer['max_curvature'] - df_layer['max_curvature'].min()) /
            (df_layer['max_curvature'].max() - df_layer['max_curvature'].min() + 1e-10)
        )

        # Compute difficulty
        df_layer['difficulty_score'] = df_layer.apply(
            lambda row: compute_reframing_difficulty(
                row['norm_semantic_distance'],
                row['norm_mean_curvature'],
                row['norm_max_curvature'],
                row['geodesic_efficiency']
            ),
            axis=1
        )

        difficulty_rows.append(df_layer)

    return pd.concat(difficulty_rows, ignore_index=True)


def compute_layer_importance(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Compute which layers show the most transformation.

    Returns:
        DataFrame with importance metrics per pattern and layer
    """
    importance = df_analysis.groupby(['pattern_name', 'layer_idx']).agg({
        'semantic_distance': 'mean',
        'mean_curvature': 'mean',
        'path_length': 'mean',
        'geodesic_efficiency': 'mean',
        'landmark_accuracy': 'mean'
    }).reset_index()

    importance.columns = [
        'pattern_name', 'layer_idx',
        'avg_semantic_distance', 'avg_mean_curvature', 'avg_path_length',
        'avg_geodesic_efficiency', 'avg_landmark_accuracy'
    ]

    return importance


def main():
    print("=" * 80)
    print("PHASE 4: Semantic Trajectory Analysis")
    print("=" * 80)

    # Load dataset
    print(f"\n📥 Loading dataset...")
    df = pd.read_csv(DATA_DIR / "pattern_metadata.csv")

    # Load train split
    with open(DATA_DIR / "train_test_split.json", 'r') as f:
        splits = json.load(f)

    # Get layer indices
    layer_indices = get_layer_indices()
    print(f"\n📊 Configuration:")
    print(f"  Strategic layers: {layer_indices}")
    print(f"  Fine-grained alphas: {len(FINE_ALPHAS)} points")

    # Analyze all patterns
    print(f"\n🔄 Analyzing geometric properties for all patterns...")

    all_analyses = []

    for pattern_name in sorted(df['pattern_name'].unique()):
        # Get training examples
        train_ids = set(splits['train'][pattern_name])
        df_pattern = df[df['example_id'].isin(train_ids)]
        pattern_slug = pattern_name.lower().replace(' ', '_').replace('&', 'and')

        print(f"\n  Analyzing {pattern_name} ({len(df_pattern)} examples)...")

        # Analyze landmark paths
        analyses = analyze_pattern_paths(
            pattern_name, pattern_slug, df_pattern, layer_indices, 'landmark'
        )

        all_analyses.extend(analyses)
        print(f"    ✅ Analyzed {len(analyses)} paths")

    print(f"\n✅ Total analyses: {len(all_analyses)}")

    # Convert to DataFrames
    print(f"\n🔄 Processing analysis results...")

    df_analysis = flatten_analysis_for_csv(all_analyses)
    print(f"  ✅ Flattened to {len(df_analysis)} rows")

    # Compute difficulty scores
    print(f"\n🔄 Computing difficulty scores...")
    df_difficulty = compute_difficulty_scores(df_analysis)
    print(f"  ✅ Difficulty scores computed")

    # Compute layer importance
    print(f"\n🔄 Computing layer importance...")
    df_importance = compute_layer_importance(df_analysis)
    print(f"  ✅ Layer importance computed")

    # Save results
    print(f"\n💾 Saving analysis results...")

    df_analysis.to_csv(OUTPUT_DIR / "distance_metrics.csv", index=False)
    print(f"  ✅ Saved: distance_metrics.csv")

    df_difficulty.to_csv(OUTPUT_DIR / "difficulty_scores.csv", index=False)
    print(f"  ✅ Saved: difficulty_scores.csv")

    df_importance.to_csv(OUTPUT_DIR / "layer_importance.csv", index=False)
    print(f"  ✅ Saved: layer_importance.csv")

    # Save detailed curvature profiles (sampled)
    print(f"\n💾 Saving curvature profiles (first 10 examples per pattern)...")
    curvature_data = []

    for analysis in all_analyses[:100]:  # Limit to first 100 for file size
        for layer_idx, layer_data in analysis['per_layer'].items():
            for i, (alpha, curv) in enumerate(zip(FINE_ALPHAS, layer_data['curvatures'])):
                curvature_data.append({
                    'example_id': analysis['example_id'],
                    'pattern_name': analysis['pattern_name'],
                    'layer_idx': layer_idx,
                    'alpha': alpha,
                    'curvature': curv
                })

    df_curvatures = pd.DataFrame(curvature_data)
    df_curvatures.to_csv(OUTPUT_DIR / "curvature_profiles.csv", index=False)
    print(f"  ✅ Saved: curvature_profiles.csv")

    # Compute aggregate statistics
    print(f"\n📊 Computing aggregate statistics...")

    # Pattern-level summary
    pattern_summary = df_difficulty.groupby('pattern_name').agg({
        'difficulty_score': ['mean', 'std'],
        'semantic_distance': 'mean',
        'mean_curvature': 'mean',
        'geodesic_efficiency': 'mean',
        'landmark_accuracy': 'mean'
    }).reset_index()

    pattern_summary.columns = [
        'pattern_name',
        'avg_difficulty', 'std_difficulty',
        'avg_semantic_distance', 'avg_mean_curvature',
        'avg_geodesic_efficiency', 'avg_landmark_accuracy'
    ]

    pattern_summary = pattern_summary.sort_values('avg_difficulty', ascending=False)
    pattern_summary.to_csv(OUTPUT_DIR / "pattern_summary.csv", index=False)
    print(f"  ✅ Saved: pattern_summary.csv")

    # Print summary
    print(f"\n" + "=" * 80)
    print(f"✅ Phase 4 Complete!")
    print(f"=" * 80)
    print(f"\n📊 Analysis Summary:")
    print(f"  Patterns analyzed: {df_analysis['pattern_name'].nunique()}")
    print(f"  Total examples: {df_analysis['example_id'].nunique()}")
    print(f"  Layers analyzed: {len(layer_indices)}")
    print(f"\n  Top 5 most difficult patterns:")
    for idx, row in pattern_summary.head().iterrows():
        print(f"    {row['pattern_name']}: {row['avg_difficulty']:.3f}")
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
