#!/usr/bin/env python3
"""
Phase 5: Create Visualizations

Generates publication-ready visualizations:
1. UMAP/t-SNE embeddings showing pattern clusters
2. Pattern difficulty heatmaps
3. Layer importance heatmaps
4. Path trajectory curvature plots
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import (
    load_pattern_activations,
    get_layer_indices
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
ANALYSIS_DIR = PROJECT_ROOT / "analysis/therapeutic_reframing/geometric_analysis"
OUTPUT_DIR = PROJECT_ROOT / "analysis/therapeutic_reframing/visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_LARGE = (14, 10)
FIGSIZE_MEDIUM = (12, 8)


def create_umap_embeddings(layer_idx: int = 15):
    """Create UMAP visualization for middle layer."""
    print(f"\n📊 Creating UMAP embeddings for layer {layer_idx}...")

    # Load pattern metadata
    df = pd.read_csv(DATA_DIR / "pattern_metadata.csv")

    # Collect activations
    all_negative = []
    all_transformed = []
    all_positive = []
    labels = []
    pattern_names = []

    for pattern_name in sorted(df['pattern_name'].unique()):
        pattern_slug = pattern_name.lower().replace(' ', '_').replace('&', 'and')

        activations = load_pattern_activations(pattern_slug, device='cpu')

        neg = activations['negative'][layer_idx].numpy()
        trans = activations['transformed'][layer_idx].numpy()
        pos = activations['positive'][layer_idx].numpy()

        all_negative.append(neg)
        all_transformed.append(trans)
        all_positive.append(pos)

        n_examples = neg.shape[0]
        labels.extend(['negative'] * n_examples)
        labels.extend(['transformed'] * n_examples)
        labels.extend(['positive'] * n_examples)
        pattern_names.extend([pattern_name] * (n_examples * 3))

    # Concatenate
    X = np.vstack(all_negative + all_transformed + all_positive)
    print(f"  Total vectors: {X.shape[0]}")

    # Reduce dimensionality
    print(f"  Fitting {'UMAP' if HAS_UMAP else 't-SNE'}...")
    if HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    X_embedded = reducer.fit_transform(X)

    # Create DataFrame
    df_embed = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'type': labels,
        'pattern': pattern_names
    })

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    # Plot by type
    for text_type, marker, color in [
        ('negative', 'o', '#e74c3c'),
        ('transformed', 's', '#f39c12'),
        ('positive', '^', '#27ae60')
    ]:
        subset = df_embed[df_embed['type'] == text_type]
        ax.scatter(subset['x'], subset['y'], alpha=0.6, s=50,
                  marker=marker, c=color, label=text_type.capitalize(),
                  edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(f'{"UMAP" if HAS_UMAP else "t-SNE"} Embedding of Activation Space (Layer {layer_idx})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"umap_layer_{layer_idx}.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: umap_layer_{layer_idx}.png")
    plt.close()

    # Plot by pattern (separate figure)
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    unique_patterns = df_embed['pattern'].unique()
    colors = sns.color_palette("husl", len(unique_patterns))

    for pattern, color in zip(unique_patterns, colors):
        subset = df_embed[df_embed['pattern'] == pattern]
        ax.scatter(subset['x'], subset['y'], alpha=0.5, s=30,
                  c=[color], label=pattern[:30], edgecolors='none')

    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(f'{"UMAP" if HAS_UMAP else "t-SNE"} Colored by Cognitive Pattern (Layer {layer_idx})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"umap_by_pattern_layer_{layer_idx}.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: umap_by_pattern_layer_{layer_idx}.png")
    plt.close()


def create_difficulty_heatmap():
    """Create heatmap of difficulty scores by pattern."""
    print(f"\n📊 Creating difficulty heatmaps...")

    # Load difficulty scores
    df_difficulty = pd.read_csv(ANALYSIS_DIR / "difficulty_scores.csv")

    # Pattern × Layer heatmap
    pivot = df_difficulty.pivot_table(
        index='pattern_name',
        columns='layer_idx',
        values='difficulty_score',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Difficulty Score'},
                linewidths=0.5, ax=ax)
    ax.set_title('Reframing Difficulty by Pattern and Layer',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Cognitive Pattern', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pattern_difficulty_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: pattern_difficulty_heatmap.png")
    plt.close()

    # Overall pattern difficulty (averaged across layers)
    pattern_avg = df_difficulty.groupby('pattern_name')['difficulty_score'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    pattern_avg.plot(kind='barh', color='coral', edgecolor='black', ax=ax)
    ax.set_xlabel('Average Difficulty Score', fontsize=12)
    ax.set_ylabel('Cognitive Pattern', fontsize=12)
    ax.set_title('Reframing Difficulty by Pattern (Averaged Across Layers)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pattern_difficulty_ranking.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: pattern_difficulty_ranking.png")
    plt.close()


def create_layer_importance_heatmap():
    """Create heatmap showing which layers are most important."""
    print(f"\n📊 Creating layer importance heatmaps...")

    # Load layer importance
    df_importance = pd.read_csv(ANALYSIS_DIR / "layer_importance.csv")

    # Semantic distance by pattern × layer
    pivot_dist = df_importance.pivot_table(
        index='pattern_name',
        columns='layer_idx',
        values='avg_semantic_distance',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot_dist, annot=True, fmt='.1f', cmap='viridis',
                cbar_kws={'label': 'Avg Semantic Distance'},
                linewidths=0.5, ax=ax)
    ax.set_title('Semantic Distance by Pattern and Layer',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Cognitive Pattern', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layer_semantic_distance_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: layer_semantic_distance_heatmap.png")
    plt.close()

    # Curvature by pattern × layer
    pivot_curv = df_importance.pivot_table(
        index='pattern_name',
        columns='layer_idx',
        values='avg_mean_curvature',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot_curv, annot=True, fmt='.3f', cmap='plasma',
                cbar_kws={'label': 'Avg Mean Curvature (rad)'},
                linewidths=0.5, ax=ax)
    ax.set_title('Path Curvature by Pattern and Layer',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Cognitive Pattern', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layer_curvature_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: layer_curvature_heatmap.png")
    plt.close()


def create_trajectory_plots():
    """Create curvature trajectory plots."""
    print(f"\n📊 Creating trajectory plots...")

    # Load curvature profiles
    df_curvatures = pd.read_csv(ANALYSIS_DIR / "curvature_profiles.csv")

    # Plot: Average curvature trajectory by pattern
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    patterns = sorted(df_curvatures['pattern_name'].unique())

    for idx, pattern in enumerate(patterns):
        if idx >= len(axes):
            break

        ax = axes[idx]
        df_pattern = df_curvatures[df_curvatures['pattern_name'] == pattern]

        # Average across examples and layers
        avg_curv = df_pattern.groupby('alpha')['curvature'].mean()

        ax.plot(avg_curv.index, avg_curv.values, linewidth=2, color='#3498db')
        ax.fill_between(avg_curv.index, 0, avg_curv.values, alpha=0.3, color='#3498db')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Landmark')
        ax.set_title(pattern[:30], fontsize=10, fontweight='bold')
        ax.set_xlabel('Alpha', fontsize=9)
        ax.set_ylabel('Curvature (rad)', fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(patterns), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Average Curvature Trajectories by Pattern', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curvature_trajectories_by_pattern.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: curvature_trajectories_by_pattern.png")
    plt.close()

    # Plot: Curvature by layer (one pattern as example)
    example_pattern = patterns[0]
    df_example = df_curvatures[df_curvatures['pattern_name'] == example_pattern]

    layer_indices = sorted(df_example['layer_idx'].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    colors = sns.color_palette("viridis", len(layer_indices))

    for layer_idx, color in zip(layer_indices, colors):
        df_layer = df_example[df_example['layer_idx'] == layer_idx]
        avg_curv = df_layer.groupby('alpha')['curvature'].mean()
        ax.plot(avg_curv.index, avg_curv.values, label=f'Layer {layer_idx}',
               linewidth=2, color=color)

    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Landmark')
    ax.set_xlabel('Alpha (Position Along Path)', fontsize=12)
    ax.set_ylabel('Curvature (radians)', fontsize=12)
    ax.set_title(f'Curvature Profiles Across Layers: {example_pattern}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curvature_by_layer_example.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: curvature_by_layer_example.png")
    plt.close()


def create_summary_plots():
    """Create additional summary visualizations."""
    print(f"\n📊 Creating summary plots...")

    # Load data
    df_difficulty = pd.read_csv(ANALYSIS_DIR / "difficulty_scores.csv")

    # Box plot: Difficulty distribution by layer
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    layer_indices = sorted(df_difficulty['layer_idx'].unique())
    data_by_layer = [df_difficulty[df_difficulty['layer_idx'] == l]['difficulty_score'].values
                     for l in layer_indices]

    bp = ax.boxplot(data_by_layer, labels=layer_indices, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Difficulty Score', fontsize=12)
    ax.set_title('Distribution of Reframing Difficulty by Layer',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "difficulty_distribution_by_layer.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: difficulty_distribution_by_layer.png")
    plt.close()


def main():
    print("=" * 80)
    print("PHASE 5: Visualization Generation")
    print("=" * 80)

    # Create visualizations
    create_umap_embeddings(layer_idx=15)  # Middle layer
    create_difficulty_heatmap()
    create_layer_importance_heatmap()
    create_trajectory_plots()
    create_summary_plots()

    # Summary
    print(f"\n" + "=" * 80)
    print(f"✅ Phase 5 Complete!")
    print(f"=" * 80)
    print(f"\n📊 Visualizations created:")
    print(f"  - UMAP embeddings (2 plots)")
    print(f"  - Difficulty heatmaps (2 plots)")
    print(f"  - Layer importance heatmaps (2 plots)")
    print(f"  - Trajectory plots (2 plots)")
    print(f"  - Summary plots (1 plot)")
    print(f"\n📁 Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
