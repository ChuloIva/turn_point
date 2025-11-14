#!/usr/bin/env python3
"""
Interpret Directional Vectors

Uses activation injection to decode what semantic concepts are encoded in the
directional vectors computed by 05_extract_directional_vectors.py.

For each directional vector, we inject it at different intensities and generate
text to see what concept emerges at each point along the semantic axis.

Intensities: [-2.0, -1.0, 0.0, 1.0, 2.0]
- Negative intensities: Move toward negative pole
- Zero: Baseline (no injection)
- Positive intensities: Move toward positive pole
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, List
from tqdm import tqdm

from nnsight_selfie import ModelAgnosticSelfie, get_optimal_device, InterpretationPrompt

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import get_layer_indices

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
VECTORS_DIR = PROJECT_ROOT / "data/therapeutic_reframing/directional_vectors"
OUTPUT_DIR = PROJECT_ROOT / "analysis/therapeutic_reframing/directional_interpretations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
INTENSITIES = [-2.0, -1.0, 0.0, 1.0, 2.0]
MAX_INTERPRETATION_TOKENS = 60


def create_interpretation_prompt(tokenizer) -> InterpretationPrompt:
    """Create prompt for interpreting injected activations."""
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


def load_directional_vectors(pattern_slug: str) -> Dict:
    """
    Load directional vectors for a pattern.

    Returns:
        Dict with structure:
        {
            'neg_to_trans': {'layer_1': array, 'layer_5': array, ...},
            'trans_to_pos': {'layer_1': array, ...},
            'neg_to_pos': {'layer_1': array, ...}
        }
    """
    pattern_dir = VECTORS_DIR / pattern_slug

    if not pattern_dir.exists():
        raise FileNotFoundError(f"Pattern directory not found: {pattern_dir}")

    # Load vectors
    vectors_file = pattern_dir / "directional_vectors.npz"
    data = np.load(vectors_file)

    # Load statistics
    stats_file = pattern_dir / "statistics.json"
    with open(stats_file, 'r') as f:
        statistics = json.load(f)

    # Organize by vector type
    vectors = {
        'neg_to_trans': {},
        'trans_to_pos': {},
        'neg_to_pos': {}
    }

    for key in data.keys():
        # Key format: {vector_type}_layer_{layer_idx}
        for vector_type in vectors.keys():
            if key.startswith(vector_type):
                layer_key = key[len(vector_type) + 1:]  # Remove "{vector_type}_"
                vectors[vector_type][layer_key] = data[key]

    return {
        'vectors': vectors,
        'statistics': statistics['statistics']
    }


def interpret_directional_vector(selfie, direction_vector: np.ndarray,
                                 layer_idx: int, intensity: float,
                                 prompt: InterpretationPrompt) -> str:
    """
    Interpret a directional vector at a specific intensity.

    Args:
        selfie: ModelAgnosticSelfie instance
        direction_vector: Vector to inject (shape: hidden_dim)
        layer_idx: Layer index to inject at
        intensity: Multiplier for the vector
        prompt: InterpretationPrompt object

    Returns:
        Generated interpretation text
    """
    from nnsight_selfie.utils import get_layer_by_path

    # Get prompt text
    formatted_prompt = prompt.get_prompt()

    # Ensure vector is on correct device
    device = selfie.device
    direction_vector = torch.from_numpy(direction_vector).to(device)

    # Generate with injection
    injection_position = -1  # Last token

    with selfie.model.generate(formatted_prompt, max_new_tokens=MAX_INTERPRETATION_TOKENS) as tracer:
        layer = get_layer_by_path(selfie.model, selfie.layer_paths[layer_idx])

        # Get original activations
        original_output = layer.output[0]

        # Get shape
        batch_size, seq_len, hidden_size = original_output.shape

        # Expand direction vector for injection
        try:
            vector_expanded = direction_vector.expand(batch_size, 1, hidden_size)
        except Exception:
            vector_expanded = direction_vector.repeat(batch_size, 1, 1)

        # Inject: original + (intensity * direction)
        original_output[:, injection_position, :] = (
            original_output[:, injection_position, :] +
            intensity * vector_expanded[:, 0, :]
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


def interpret_pattern_vectors(selfie, prompt: InterpretationPrompt,
                              pattern_slug: str, layer_indices: List[int]) -> Dict:
    """
    Interpret all directional vectors for a pattern.

    Returns:
        Dict with interpretation results organized by vector type, layer, and intensity
    """
    print(f"\n  Loading vectors for {pattern_slug}...")
    data = load_directional_vectors(pattern_slug)
    vectors = data['vectors']
    statistics = data['statistics']

    results = {
        'pattern_slug': pattern_slug,
        'interpretations': {},
        'statistics': statistics
    }

    vector_types = ['neg_to_trans', 'trans_to_pos', 'neg_to_pos']

    for vector_type in vector_types:
        print(f"\n  Interpreting {vector_type}...")
        results['interpretations'][vector_type] = {}

        for layer_idx in tqdm(layer_indices, desc=f"    Layers", leave=False):
            layer_key = f'layer_{layer_idx}'
            direction_vector = vectors[vector_type][layer_key]

            layer_results = []

            for intensity in INTENSITIES:
                interpretation = interpret_directional_vector(
                    selfie,
                    direction_vector,
                    layer_idx,
                    intensity,
                    prompt
                )

                layer_results.append({
                    'intensity': intensity,
                    'interpretation': interpretation.strip()
                })

            results['interpretations'][vector_type][layer_idx] = layer_results

    return results


def save_pattern_interpretations(results: Dict):
    """Save interpretation results for a pattern."""
    pattern_slug = results['pattern_slug']
    output_file = OUTPUT_DIR / f"{pattern_slug}_interpretations.json"

    # Convert integer keys to strings for JSON compatibility
    results_json = {
        'pattern_slug': results['pattern_slug'],
        'interpretations': {
            vector_type: {
                str(layer_idx): layer_results
                for layer_idx, layer_results in type_results.items()
            }
            for vector_type, type_results in results['interpretations'].items()
        },
        'statistics': results['statistics']
    }

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"    ✅ Saved: {output_file.name}")


def create_summary_report(all_results: List[Dict], layer_indices: List[int]):
    """Create a summary report across all patterns."""
    print("\n" + "=" * 80)
    print("CREATING SUMMARY REPORT")
    print("=" * 80)

    summary = {
        'n_patterns': len(all_results),
        'layer_indices': layer_indices,
        'intensities': INTENSITIES,
        'vector_types': ['neg_to_trans', 'trans_to_pos', 'neg_to_pos'],
        'patterns': []
    }

    for results in all_results:
        pattern_summary = {
            'pattern_slug': results['pattern_slug'],
            'sample_interpretations': {}
        }

        # Include one sample interpretation per vector type (middle layer, intensity=1.0)
        middle_layer_idx = layer_indices[len(layer_indices) // 2]

        for vector_type in summary['vector_types']:
            layer_results = results['interpretations'][vector_type][middle_layer_idx]

            # Find intensity=1.0
            for item in layer_results:
                if item['intensity'] == 1.0:
                    pattern_summary['sample_interpretations'][vector_type] = {
                        'layer': middle_layer_idx,
                        'intensity': 1.0,
                        'interpretation': item['interpretation']
                    }
                    break

        summary['patterns'].append(pattern_summary)

    # Save summary
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved: {summary_file}")

    # Print some examples
    print("\n" + "=" * 80)
    print("SAMPLE INTERPRETATIONS (intensity=1.0, middle layer)")
    print("=" * 80)

    for pattern_summary in summary['patterns'][:3]:  # Show first 3 patterns
        print(f"\nPattern: {pattern_summary['pattern_slug']}")
        print("-" * 40)
        for vector_type, interp_data in pattern_summary['sample_interpretations'].items():
            print(f"\n  {vector_type}:")
            print(f"    {interp_data['interpretation'][:150]}...")


def main():
    print("=" * 80)
    print("DIRECTIONAL VECTOR INTERPRETATION")
    print("=" * 80)

    # Load metadata
    print("\n📥 Loading metadata...")
    with open(VECTORS_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)

    patterns = metadata['patterns_processed']
    layer_indices = metadata['layer_indices']

    print(f"  Found {len(patterns)} patterns")
    print(f"  Layers: {layer_indices}")
    print(f"  Intensities: {INTENSITIES}")

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

    # Process patterns
    print(f"\n🔄 Interpreting directional vectors...")

    all_results = []

    for pattern_slug in patterns:
        print(f"\nProcessing: {pattern_slug}")
        print("-" * 40)

        try:
            results = interpret_pattern_vectors(
                selfie, prompt, pattern_slug, layer_indices
            )
            all_results.append(results)
            save_pattern_interpretations(results)

        except Exception as e:
            print(f"\n❌ Error processing {pattern_slug}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create summary report
    if all_results:
        create_summary_report(all_results, layer_indices)

    print("\n" + "=" * 80)
    print("✅ DIRECTIONAL VECTOR INTERPRETATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    print(f"  - Per-pattern interpretation files")
    print(f"  - summary.json with sample interpretations")
    print(f"\nProcessed {len(all_results)}/{len(patterns)} patterns successfully")


if __name__ == "__main__":
    main()
