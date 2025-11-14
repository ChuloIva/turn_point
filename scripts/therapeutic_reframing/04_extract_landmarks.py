#!/usr/bin/env python3
"""
Phase 2.2: Interpret and Validate Semantic Landmarks

Interprets paths at multiple alpha values and validates that:
1. alpha=0.0 matches negative example
2. alpha=0.5 matches transformed example (landmark)
3. alpha=1.0 matches positive example

Also extracts fine-grained intermediate steps for analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from nnsight_selfie import ModelAgnosticSelfie, get_optimal_device, InterpretationPrompt
from nnsight_selfie.semantic_path_learning import interpret_multilayer_path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import get_layer_indices

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
PATHS_DIR = PROJECT_ROOT / "learned_paths/therapeutic_reframing/pattern_specific"
OUTPUT_DIR = PROJECT_ROOT / "analysis/therapeutic_reframing/interpretation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
INTERPRETATION_ALPHAS = [0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0]
MAX_INTERPRETATION_TOKENS = 60
SAMPLE_SIZE_PER_PATTERN = 1  # Interpret 5 examples per pattern for validation


def create_interpretation_prompt(tokenizer) -> InterpretationPrompt:
    """Create prompt for interpreting activations."""
    return InterpretationPrompt(
        tokenizer,
        [
            """You are analyzing neural network activations extracted from other texts and injected at positions marked with '_' below. All '_' marks contain the same semantic representation.

Your task is to decode what concept or meaning is encoded at the '_' positions.

Here is the text:""",

            None,  # Placeholder for generated text

            """ . That is the text with activations injected at '_'.

Interpret what concept or meaning is represented at the '_' marks. Focus on:
- The most salient semantic content
- Key conceptual associations
- Dominant thematic elements

Describe the concept clearly and directly in 1-2 sentences."""
        ]
    )


def interpret_single_layer_path(selfie, path, layer_idx: int, alpha: float,
                                prompt: InterpretationPrompt) -> str:
    """
    Interpret path at a single layer only.

    Args:
        selfie: ModelAgnosticSelfie instance
        path: Path object (single-layer or multi-layer)
        layer_idx: Single layer index to inject at
        alpha: Position along path (0.0 to 1.0)
        prompt: InterpretationPrompt object

    Returns:
        Generated interpretation text
    """
    from nnsight_selfie.utils import get_layer_by_path

    # Get vector at alpha for this specific layer
    # Handle both multi-layer and single-layer paths
    if hasattr(path, 'layer_paths'):
        # Multi-layer path
        layer_vector = path.layer_paths[layer_idx].interpolate(alpha)
    else:
        # Single-layer path - just interpolate directly
        layer_vector = path.interpolate(alpha)

    # Get prompt text
    formatted_prompt = prompt.get_prompt()

    # Ensure activation is on correct device
    device = selfie.device
    layer_vector = layer_vector.to(device)

    # Generate with single-layer injection
    injection_position = -1  # Last token
    injection_strength = 1.0

    with selfie.model.generate(formatted_prompt, max_new_tokens=MAX_INTERPRETATION_TOKENS) as tracer:
        layer = get_layer_by_path(selfie.model, selfie.layer_paths[layer_idx])

        # Get original activations
        original_output = layer.output[0]

        # Get shape
        batch_size, seq_len, hidden_size = original_output.shape

        # Expand activation for injection
        try:
            activation_expanded = layer_vector.expand(batch_size, 1, hidden_size)
        except Exception:
            activation_expanded = layer_vector.repeat(batch_size, 1, 1)

        # Inject at position
        original_output[:, injection_position, :] = (
            original_output[:, injection_position, :] +
            injection_strength * activation_expanded[:, 0, :]
        )

        output_ids = selfie.model.generator.output.save()

    # Decode output
    generated_text = selfie.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the new generated text (remove prompt)
    prompt_text = selfie.model.tokenizer.decode(
        selfie.model.tokenizer.encode(formatted_prompt, add_special_tokens=False),
        skip_special_tokens=True
    )

    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text):].strip()

    return generated_text


def interpret_path_at_alphas(selfie, path, alphas: List[float],
                            prompt: InterpretationPrompt) -> Dict[int, List[Dict]]:
    """
    Interpret path at multiple alpha values, testing each layer individually.

    Returns:
        Dict mapping layer_idx -> List of dicts with 'alpha' and 'interpretation'
    """
    # Handle both single-layer and multi-layer paths
    if hasattr(path, 'layer_indices'):
        # Multi-layer path
        layer_indices = path.layer_indices
    else:
        # Single-layer path - get layer from metadata
        layer_idx = path.metadata.get('layer_index')
        if layer_idx is None:
            raise ValueError("Single-layer path missing 'layer_index' in metadata")
        layer_indices = [layer_idx]

    results_by_layer = {}

    for layer_idx in layer_indices:
        print(f"      Testing layer {layer_idx}...")
        layer_interpretations = []

        for alpha in alphas:
            interp_text = interpret_single_layer_path(
                selfie,
                path,
                layer_idx,
                alpha,
                prompt
            )

            layer_interpretations.append({
                'alpha': alpha,
                'interpretation': interp_text.strip()
            })

        results_by_layer[layer_idx] = layer_interpretations

    return results_by_layer


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple word overlap similarity between two texts.
    (For better similarity, could use sentence embeddings)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def process_pattern_sample(selfie, prompt: InterpretationPrompt, pattern_name: str,
                          pattern_slug: str, df_pattern: pd.DataFrame,
                          sample_size: int = 5) -> List[Dict]:
    """
    Interpret and validate paths for a sample of examples from one pattern.

    Returns:
        List of validation results (one per example, with layer-specific results)
    """
    pattern_dir = PATHS_DIR / pattern_slug

    # Sample random examples
    sample_indices = np.random.choice(len(df_pattern), min(sample_size, len(df_pattern)), replace=False)

    results = []

    for idx in tqdm(sample_indices, desc=f"  {pattern_name}"):
        row = df_pattern.iloc[idx]
        example_id = row['example_id']

        # Load landmark path
        path_file = pattern_dir / f"{example_id}_landmark.pkl"
        if not path_file.exists():
            continue

        with open(path_file, 'rb') as f:
            path = pickle.load(f)

        # Interpret at alphas for each layer
        interpretations_by_layer = interpret_path_at_alphas(selfie, path, INTERPRETATION_ALPHAS, prompt)

        # Compute validation metrics for each layer
        layer_validations = {}
        for layer_idx, layer_interps in interpretations_by_layer.items():
            # Validate key landmarks for this layer
            neg_similarity = compute_text_similarity(
                layer_interps[0]['interpretation'],  # alpha=0.0
                row['negative_text']
            )

            trans_similarity = compute_text_similarity(
                layer_interps[3]['interpretation'],  # alpha=0.5
                row['transformed_text']
            )

            pos_similarity = compute_text_similarity(
                layer_interps[6]['interpretation'],  # alpha=1.0
                row['positive_text']
            )

            layer_validations[layer_idx] = {
                'negative_similarity': neg_similarity,
                'transformed_similarity': trans_similarity,
                'positive_similarity': pos_similarity,
                'average_similarity': (neg_similarity + trans_similarity + pos_similarity) / 3
            }

        result = {
            'example_id': example_id,
            'pattern_name': pattern_name,
            'layer_interpretations': interpretations_by_layer,
            'layer_validations': layer_validations,
            'reference_texts': {
                'negative': row['negative_text'],
                'transformed': row['transformed_text'],
                'positive': row['positive_text']
            }
        }

        results.append(result)

    return results


def main():
    print("=" * 80)
    print("PHASE 2.2: Landmark Interpretation & Validation")
    print("=" * 80)

    # Load dataset
    print(f"\n📥 Loading dataset...")
    df = pd.read_csv(DATA_DIR / "pattern_metadata.csv")

    # Load train split
    with open(DATA_DIR / "train_test_split.json", 'r') as f:
        splits = json.load(f)

    # Initialize model
    print(f"\n🔧 Loading model: {MODEL_NAME}")
    device = get_optimal_device()
    print(f"  Device: {device}")

    selfie = ModelAgnosticSelfie(
        MODEL_NAME,
        dtype=torch.bfloat16,
        load_in_8bit=False
    )
    print(f"✅ Model loaded!")

    # Create interpretation prompt
    prompt = create_interpretation_prompt(selfie.model.tokenizer)
    print(f"\n✅ Interpretation prompt created")

    # Get layer indices
    layer_indices = get_layer_indices()
    print(f"\n📊 Testing layers: {layer_indices}")

    # Process patterns
    print(f"\n🔄 Interpreting paths for sample examples...")
    print(f"  Alphas: {INTERPRETATION_ALPHAS}")
    print(f"  Sample size per pattern: {SAMPLE_SIZE_PER_PATTERN}")
    print(f"  Each example will be tested with {len(layer_indices)} layers independently")

    all_results = []
    pattern_summaries = {}

    for pattern_name in sorted(df['pattern_name'].unique()):
        # Get training examples
        train_ids = set(splits['train'][pattern_name])
        df_pattern = df[df['example_id'].isin(train_ids)]
        pattern_slug = pattern_name.lower().replace(' ', '_').replace('&', 'and')

        print(f"\n  Processing {pattern_name}...")
        results = process_pattern_sample(
            selfie, prompt, pattern_name, pattern_slug,
            df_pattern, SAMPLE_SIZE_PER_PATTERN
        )

        all_results.extend(results)

        # Compute pattern-level statistics per layer
        if results:
            layer_stats = {}
            for layer_idx in layer_indices:
                layer_results = [r['layer_validations'][layer_idx] for r in results if layer_idx in r['layer_validations']]

                if layer_results:
                    avg_neg_sim = np.mean([lr['negative_similarity'] for lr in layer_results])
                    avg_trans_sim = np.mean([lr['transformed_similarity'] for lr in layer_results])
                    avg_pos_sim = np.mean([lr['positive_similarity'] for lr in layer_results])

                    layer_stats[layer_idx] = {
                        'avg_negative_similarity': float(avg_neg_sim),
                        'avg_transformed_similarity': float(avg_trans_sim),
                        'avg_positive_similarity': float(avg_pos_sim),
                        'avg_overall_similarity': float((avg_neg_sim + avg_trans_sim + avg_pos_sim) / 3)
                    }

            pattern_summaries[pattern_name] = {
                'n_examples': len(results),
                'layer_statistics': layer_stats
            }

            # Print summary for best and worst layers
            if layer_stats:
                best_layer = max(layer_stats.items(), key=lambda x: x[1]['avg_overall_similarity'])
                worst_layer = min(layer_stats.items(), key=lambda x: x[1]['avg_overall_similarity'])
                print(f"    Best layer: {best_layer[0]} (avg={best_layer[1]['avg_overall_similarity']:.3f})")
                print(f"    Worst layer: {worst_layer[0]} (avg={worst_layer[1]['avg_overall_similarity']:.3f})")

    # Save results
    print(f"\n💾 Saving interpretation results...")

    # Convert integer keys to strings for JSON compatibility
    all_results_json = []
    for result in all_results:
        result_json = result.copy()
        result_json['layer_interpretations'] = {str(k): v for k, v in result['layer_interpretations'].items()}
        result_json['layer_validations'] = {str(k): v for k, v in result['layer_validations'].items()}
        all_results_json.append(result_json)

    with open(OUTPUT_DIR / "landmark_validation.json", 'w') as f:
        json.dump(all_results_json, f, indent=2)
    print(f"  ✅ Saved: landmark_validation.json ({len(all_results)} examples)")

    # Convert integer keys for pattern summaries
    pattern_summaries_json = {}
    for pattern_name, pattern_data in pattern_summaries.items():
        pattern_data_json = pattern_data.copy()
        if 'layer_statistics' in pattern_data_json:
            pattern_data_json['layer_statistics'] = {str(k): v for k, v in pattern_data['layer_statistics'].items()}
        pattern_summaries_json[pattern_name] = pattern_data_json

    with open(OUTPUT_DIR / "pattern_validation_summary.json", 'w') as f:
        json.dump(pattern_summaries_json, f, indent=2)
    print(f"  ✅ Saved: pattern_validation_summary.json")

    # Print summary
    print(f"\n" + "=" * 80)
    print(f"✅ Phase 2.2 Complete!")
    print(f"=" * 80)
    print(f"\n📊 Validation Summary:")
    print(f"  Total examples interpreted: {len(all_results)}")
    print(f"  Patterns validated: {len(pattern_summaries)}")

    # Compute cross-pattern, cross-layer averages
    if pattern_summaries:
        print(f"\n  Average similarities by layer (across all patterns):")
        layer_summary = {}
        for layer_idx in layer_indices:
            layer_results = []
            for pattern_data in pattern_summaries.values():
                if layer_idx in pattern_data['layer_statistics']:
                    layer_results.append(pattern_data['layer_statistics'][layer_idx])

            if layer_results:
                avg_neg = np.mean([lr['avg_negative_similarity'] for lr in layer_results])
                avg_trans = np.mean([lr['avg_transformed_similarity'] for lr in layer_results])
                avg_pos = np.mean([lr['avg_positive_similarity'] for lr in layer_results])
                avg_overall = np.mean([lr['avg_overall_similarity'] for lr in layer_results])

                layer_summary[layer_idx] = avg_overall
                print(f"    Layer {layer_idx}: neg={avg_neg:.3f}, trans={avg_trans:.3f}, pos={avg_pos:.3f}, overall={avg_overall:.3f}")

        if layer_summary:
            best_layer_overall = max(layer_summary.items(), key=lambda x: x[1])
            worst_layer_overall = min(layer_summary.items(), key=lambda x: x[1])
            print(f"\n  Overall best layer: {best_layer_overall[0]} (avg={best_layer_overall[1]:.3f})")
            print(f"  Overall worst layer: {worst_layer_overall[0]} (avg={worst_layer_overall[1]:.3f})")

    print(f"\n📁 Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
