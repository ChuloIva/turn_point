#!/usr/bin/env python3
"""
Test High Intensity Directional Vector Injections

Tests directional vectors with much higher injection intensities to explore
the extreme poles of semantic axes. Uses 3 representative patterns and tests
across all layers with intensities ranging from -400 to +400.

This helps understand the semantic boundaries and saturation points of
directional vectors in therapeutic reframing.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm

from nnsight_selfie import ModelAgnosticSelfie, get_optimal_device, InterpretationPrompt

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from activation_cache import get_layer_indices

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VECTORS_DIR = PROJECT_ROOT / "data/therapeutic_reframing/directional_vectors"
OUTPUT_DIR = PROJECT_ROOT / "analysis/therapeutic_reframing/high_intensity_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
HIGH_INTENSITIES = [-400, -100, -10, -5, 5, 10, 100, 400]
MAX_INTERPRETATION_TOKENS = 80

# Test these 3 patterns as representatives
TEST_PATTERNS = [
    'self-critical_rumination',
    'persistent_suicidal_ideation_focus',
    'executive_fatigue_and_avolition'
]


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
- Emotional valence and intensity

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

    return vectors


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


def test_pattern_high_intensities(selfie, prompt: InterpretationPrompt,
                                   pattern_slug: str, layer_indices: List[int]) -> Dict:
    """
    Test high intensity injections for a pattern.

    Returns:
        Dict with interpretation results organized by vector type, layer, and intensity
    """
    print(f"\n  Loading vectors for {pattern_slug}...")
    vectors = load_directional_vectors(pattern_slug)

    results = {
        'pattern_slug': pattern_slug,
        'intensities_tested': HIGH_INTENSITIES,
        'interpretations': {}
    }

    vector_types = ['neg_to_trans', 'trans_to_pos', 'neg_to_pos']

    for vector_type in vector_types:
        print(f"\n  Testing {vector_type} at high intensities...")
        results['interpretations'][vector_type] = {}

        for layer_idx in tqdm(layer_indices, desc=f"    Layers", leave=False):
            layer_key = f'layer_{layer_idx}'
            direction_vector = vectors[vector_type][layer_key]

            layer_results = []

            for intensity in HIGH_INTENSITIES:
                try:
                    interpretation = interpret_directional_vector(
                        selfie,
                        direction_vector,
                        layer_idx,
                        intensity,
                        prompt
                    )

                    layer_results.append({
                        'intensity': intensity,
                        'interpretation': interpretation.strip(),
                        'success': True
                    })
                except Exception as e:
                    layer_results.append({
                        'intensity': intensity,
                        'interpretation': f"ERROR: {str(e)}",
                        'success': False
                    })

            results['interpretations'][vector_type][layer_idx] = layer_results

    return results


def save_pattern_results(results: Dict):
    """Save high intensity test results for a pattern."""
    pattern_slug = results['pattern_slug']
    output_file = OUTPUT_DIR / f"{pattern_slug}_high_intensity.json"

    # Convert integer keys to strings for JSON compatibility
    results_json = {
        'pattern_slug': results['pattern_slug'],
        'intensities_tested': results['intensities_tested'],
        'interpretations': {
            vector_type: {
                str(layer_idx): layer_results
                for layer_idx, layer_results in type_results.items()
            }
            for vector_type, type_results in results['interpretations'].items()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"    ✅ Saved: {output_file.name}")


def create_summary_report(all_results: List[Dict], layer_indices: List[int]):
    """Create a summary report comparing patterns at extreme intensities."""
    print("\n" + "=" * 80)
    print("CREATING SUMMARY REPORT")
    print("=" * 80)

    summary = {
        'n_patterns': len(all_results),
        'layer_indices': layer_indices,
        'intensities': HIGH_INTENSITIES,
        'vector_types': ['neg_to_trans', 'trans_to_pos', 'neg_to_pos'],
        'patterns': []
    }

    # Pick middle layer and extreme intensities for summary
    middle_layer_idx = layer_indices[len(layer_indices) // 2]
    extreme_negative = HIGH_INTENSITIES[0]  # -400
    extreme_positive = HIGH_INTENSITIES[-1]  # 400

    for results in all_results:
        pattern_summary = {
            'pattern_slug': results['pattern_slug'],
            'extreme_interpretations': {}
        }

        for vector_type in summary['vector_types']:
            layer_results = results['interpretations'][vector_type][middle_layer_idx]

            # Find extreme negative and positive
            neg_result = next((r for r in layer_results if r['intensity'] == extreme_negative), None)
            pos_result = next((r for r in layer_results if r['intensity'] == extreme_positive), None)

            pattern_summary['extreme_interpretations'][vector_type] = {
                'layer': middle_layer_idx,
                f'intensity_{extreme_negative}': neg_result['interpretation'] if neg_result else 'N/A',
                f'intensity_{extreme_positive}': pos_result['interpretation'] if pos_result else 'N/A'
            }

        summary['patterns'].append(pattern_summary)

    # Save summary
    summary_file = OUTPUT_DIR / "high_intensity_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved: {summary_file}")

    # Print some examples
    print("\n" + "=" * 80)
    print(f"EXTREME INTENSITY INTERPRETATIONS (Layer {middle_layer_idx})")
    print("=" * 80)

    for pattern_summary in summary['patterns']:
        print(f"\n{'='*80}")
        print(f"Pattern: {pattern_summary['pattern_slug']}")
        print(f"{'='*80}")

        for vector_type, interp_data in pattern_summary['extreme_interpretations'].items():
            print(f"\n  {vector_type.upper().replace('_', ' ')}:")
            print(f"  {'-'*76}")
            print(f"    Intensity {extreme_negative}:")
            print(f"      {interp_data[f'intensity_{extreme_negative}'][:200]}...")
            print(f"\n    Intensity {extreme_positive}:")
            print(f"      {interp_data[f'intensity_{extreme_positive}'][:200]}...")


def main():
    print("=" * 80)
    print("HIGH INTENSITY DIRECTIONAL VECTOR TESTING")
    print("=" * 80)

    print(f"\n📊 Configuration:")
    print(f"  Patterns: {TEST_PATTERNS}")
    print(f"  Intensities: {HIGH_INTENSITIES}")
    print(f"  Layers: {get_layer_indices()}")

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

    # Process patterns
    print(f"\n🔄 Testing high intensity injections...")

    all_results = []

    for pattern_slug in TEST_PATTERNS:
        print(f"\n{'='*80}")
        print(f"Processing: {pattern_slug}")
        print(f"{'='*80}")

        try:
            results = test_pattern_high_intensities(
                selfie, prompt, pattern_slug, layer_indices
            )
            all_results.append(results)
            save_pattern_results(results)

        except Exception as e:
            print(f"\n❌ Error processing {pattern_slug}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create summary report
    if all_results:
        create_summary_report(all_results, layer_indices)

    print("\n" + "=" * 80)
    print("✅ HIGH INTENSITY TESTING COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    print(f"  - Per-pattern detailed results")
    print(f"  - high_intensity_summary.json with extreme comparisons")
    print(f"\nProcessed {len(all_results)}/{len(TEST_PATTERNS)} patterns successfully")


if __name__ == "__main__":
    main()