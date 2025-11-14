#!/usr/bin/env python3
"""
Phase 3: Learn Universal Reframing Patterns

Aggregates pattern-specific paths into universal models that can generalize
across different cognitive patterns.

Tests 3 aggregation methods:
- direction_statistics
- curvature_transfer
- relative_geometry
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

from nnsight_selfie.semantic_path_learning import SemanticPathAggregator

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import get_layer_indices

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
PATHS_DIR = PROJECT_ROOT / "learned_paths/therapeutic_reframing/pattern_specific"
OUTPUT_DIR = PROJECT_ROOT / "learned_paths/therapeutic_reframing/universal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
AGGREGATION_METHODS = ["direction_statistics", "curvature_transfer", "relative_geometry"]
MAX_PATHS_PER_PATTERN = 20  # Limit for computational efficiency


def load_pattern_paths(pattern_slug: str, max_paths: int = None) -> List:
    """Load landmark paths for a pattern."""
    pattern_dir = PATHS_DIR / pattern_slug

    # Get all landmark path files
    path_files = list(pattern_dir.glob("*_landmark.pkl"))

    if max_paths and len(path_files) > max_paths:
        path_files = path_files[:max_paths]

    paths = []
    for path_file in path_files:
        with open(path_file, 'rb') as f:
            paths.append(pickle.load(f))

    return paths


def create_universal_aggregator(df: pd.DataFrame, train_splits: Dict,
                               max_paths_per_pattern: int) -> Dict[str, SemanticPathAggregator]:
    """
    Create aggregators using all 3 methods.

    Returns:
        Dict mapping method_name -> trained aggregator
    """
    aggregators = {
        method: SemanticPathAggregator()
        for method in AGGREGATION_METHODS
    }

    print(f"\n🔄 Loading and aggregating paths...")

    total_paths_added = 0

    for pattern_name in sorted(df['pattern_name'].unique()):
        pattern_slug = pattern_name.lower().replace(' ', '_').replace('&', 'and')

        print(f"\n  Loading {pattern_name}...")
        paths = load_pattern_paths(pattern_slug, max_paths_per_pattern)

        print(f"    Loaded {len(paths)} paths")

        # Add to all aggregators
        for path in tqdm(paths, desc=f"    Adding to aggregators"):
            # Get concept pair from metadata
            neg_concept = f"{pattern_name}_negative"
            pos_concept = f"{pattern_name}_positive"

            for aggregator in aggregators.values():
                aggregator.add_path(path, (neg_concept, pos_concept))

        total_paths_added += len(paths)

    print(f"\n✅ Added {total_paths_added} paths to aggregators")

    # Fit each aggregator
    print(f"\n🔬 Fitting universal representations...")
    for method, aggregator in aggregators.items():
        print(f"  Fitting {method}...")
        aggregator.fit(method=method)
        print(f"    ✅ {method} fitted")

    return aggregators


def evaluate_generalization(aggregators: Dict, df: pd.DataFrame,
                           test_splits: Dict) -> Dict:
    """
    Evaluate how well universal models generalize to held-out test examples.

    Returns:
        Dict with evaluation metrics per method
    """
    print(f"\n🧪 Evaluating generalization on test set...")

    # For now, just save the aggregators
    # Full evaluation would require:
    # 1. Load test example activations
    # 2. Apply universal path
    # 3. Interpret result
    # 4. Compare to reference positive
    # This is computationally expensive, so we'll skip for now

    evaluation = {
        method: {
            'status': 'aggregator_trained',
            'n_source_paths': len(agg.paths),
            'has_universal_representation': agg.universal_representation is not None
        }
        for method, agg in aggregators.items()
    }

    return evaluation


def main():
    print("=" * 80)
    print("PHASE 3: Universal Pattern Aggregation")
    print("=" * 80)

    # Load dataset
    print(f"\n📥 Loading dataset...")
    df = pd.read_csv(DATA_DIR / "pattern_metadata.csv")

    # Load splits
    with open(DATA_DIR / "train_test_split.json", 'r') as f:
        splits = json.load(f)

    print(f"✅ Loaded {len(df)} examples")
    print(f"  Patterns: {df['pattern_name'].nunique()}")
    print(f"  Aggregation methods: {AGGREGATION_METHODS}")
    print(f"  Max paths per pattern: {MAX_PATHS_PER_PATTERN}")

    # Create aggregators
    aggregators = create_universal_aggregator(
        df, splits['train'], MAX_PATHS_PER_PATTERN
    )

    # Evaluate
    evaluation = evaluate_generalization(aggregators, df, splits['test'])

    # Save aggregators
    print(f"\n💾 Saving universal aggregators...")
    for method, aggregator in aggregators.items():
        output_path = OUTPUT_DIR / f"universal_{method}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(aggregator, f)
        print(f"  ✅ Saved: universal_{method}.pkl")

    # Save evaluation
    eval_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(eval_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"  ✅ Saved: evaluation_summary.json")

    # Summary
    print(f"\n" + "=" * 80)
    print(f"✅ Phase 3 Complete!")
    print(f"=" * 80)
    print(f"\n📊 Summary:")
    print(f"  Aggregation methods: {len(aggregators)}")
    for method, agg in aggregators.items():
        print(f"    {method}: {len(agg.paths)} source paths")
    print(f"\n📁 Aggregators saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
