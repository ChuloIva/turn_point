#!/usr/bin/env python3
"""
Phase 1.1: Parse and Organize Therapeutic Reframing Dataset

Extracts 3 text types from each example:
- Negative: reference_negative_example
- Transformed: reference_transformed_example (intermediate landmark)
- Positive: positive_thought_pattern

Creates train/test splits per cognitive pattern type.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data/therapeutic_reframing/raw/patterns.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
STRATEGIC_LAYERS = [1, 5, 7, 11, 15, 22, 27, 29]
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_patterns() -> List[Dict]:
    """Load all patterns from JSONL file."""
    patterns = []
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                patterns.append(json.loads(line))
    return patterns


def extract_text_fields(pattern: Dict) -> Dict:
    """Extract the 3 text types we need for path learning."""
    return {
        'example_id': f"{pattern['cognitive_pattern_name'].lower().replace(' ', '_').replace('&', 'and')}_{len(extracted_data.get(pattern['cognitive_pattern_name'], []))}",
        'pattern_name': pattern['cognitive_pattern_name'],
        'pattern_type': pattern['cognitive_pattern_type'],
        'pattern_description': pattern['pattern_description'],
        'negative_text': pattern['reference_negative_example'],
        'transformed_text': pattern['reference_transformed_example'],
        'positive_text': pattern['positive_thought_pattern'],
        'source_question': pattern.get('source_question', ''),
        'model': pattern.get('model', ''),
        'timestamp': pattern.get('timestamp', '')
    }


# Track extracted data by pattern name
extracted_data = defaultdict(list)


def create_train_test_splits(data_by_pattern: Dict[str, List[Dict]]) -> Tuple[Dict, Dict]:
    """Create stratified train/test splits per pattern type."""
    train_splits = {}
    test_splits = {}

    for pattern_name, examples in data_by_pattern.items():
        n_examples = len(examples)
        n_test = max(1, int(n_examples * TEST_SPLIT_RATIO))

        # Shuffle indices
        indices = np.random.permutation(n_examples)
        test_indices = set(indices[:n_test].tolist())

        train_examples = [ex for i, ex in enumerate(examples) if i not in test_indices]
        test_examples = [ex for i, ex in enumerate(examples) if i in test_indices]

        train_splits[pattern_name] = train_examples
        test_splits[pattern_name] = test_examples

        print(f"  {pattern_name}: {len(train_examples)} train, {len(test_examples)} test")

    return train_splits, test_splits


def main():
    print("=" * 80)
    print("PHASE 1.1: Dataset Preparation")
    print("=" * 80)

    # Load patterns
    print(f"\n📥 Loading patterns from: {RAW_DATA_PATH}")
    patterns = load_patterns()
    print(f"✅ Loaded {len(patterns)} examples")

    # Extract text fields
    print("\n🔄 Extracting text fields...")
    data_by_pattern = defaultdict(list)
    all_data = []

    for pattern in patterns:
        extracted = extract_text_fields(pattern)
        pattern_name = extracted['pattern_name']
        data_by_pattern[pattern_name].append(extracted)
        all_data.append(extracted)

    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total examples: {len(all_data)}")
    print(f"  Unique patterns: {len(data_by_pattern)}")
    print(f"\n  Pattern distribution:")
    for pattern_name, examples in sorted(data_by_pattern.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"    {pattern_name}: {len(examples)} examples")

    # Create train/test splits
    print(f"\n🔀 Creating train/test splits (test={TEST_SPLIT_RATIO*100:.0f}%)...")
    train_splits, test_splits = create_train_test_splits(data_by_pattern)

    # Save full dataset
    print(f"\n💾 Saving processed data...")
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_DIR / "pattern_metadata.csv", index=False)
    print(f"  ✅ Saved: pattern_metadata.csv ({len(df)} rows)")

    # Save splits
    splits_data = {
        'train': {pattern: [ex['example_id'] for ex in examples]
                  for pattern, examples in train_splits.items()},
        'test': {pattern: [ex['example_id'] for ex in examples]
                 for pattern, examples in test_splits.items()},
        'random_seed': RANDOM_SEED
    }

    with open(OUTPUT_DIR / "train_test_split.json", 'w') as f:
        json.dump(splits_data, f, indent=2)
    print(f"  ✅ Saved: train_test_split.json")

    # Save example pairs for reference
    example_pairs = [
        {
            'example_id': ex['example_id'],
            'pattern_name': ex['pattern_name'],
            'negative': ex['negative_text'][:200] + '...' if len(ex['negative_text']) > 200 else ex['negative_text'],
            'transformed': ex['transformed_text'][:200] + '...' if len(ex['transformed_text']) > 200 else ex['transformed_text'],
            'positive': ex['positive_text'][:200] + '...' if len(ex['positive_text']) > 200 else ex['positive_text']
        }
        for ex in all_data[:10]  # Just first 10 for reference
    ]

    with open(OUTPUT_DIR / "example_pairs_preview.json", 'w') as f:
        json.dump(example_pairs, f, indent=2)
    print(f"  ✅ Saved: example_pairs_preview.json")

    # Save strategic layers config
    config = {
        'strategic_layers': STRATEGIC_LAYERS,
        'layer_rationale': {
            'early': [1, 5, 7],
            'middle': [11, 15],
            'late': [22, 27, 29]
        },
        'total_layers': len(STRATEGIC_LAYERS)
    }

    with open(OUTPUT_DIR / "strategic_layers.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✅ Saved: strategic_layers.json")

    print(f"\n✅ Phase 1.1 Complete!")
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"  - pattern_metadata.csv: Full dataset")
    print(f"  - train_test_split.json: Train/test splits")
    print(f"  - example_pairs_preview.json: Sample examples")
    print(f"  - strategic_layers.json: Layer configuration")


if __name__ == "__main__":
    main()
