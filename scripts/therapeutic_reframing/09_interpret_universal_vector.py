#!/usr/bin/env python3
"""
Interpret Universal Directional Vectors at High Intensities

Tests the universal directional vectors (extracted by 08_extract_universal_vector.py)
at high injection intensities to explore the semantic concepts they encode.

Intensities range from -400 to +400 to see extreme poles of the universal
therapeutic reframing axis.
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
VECTORS_DIR = PROJECT_ROOT / "data/therapeutic_reframing/universal_vectors"
OUTPUT_DIR = PROJECT_ROOT / "analysis/therapeutic_reframing/universal_interpretations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
HIGH_INTENSITIES = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]  # Fine-grained range around sweet spot
FOCUS_LAYERS = [5, 11]  # Best performing layers from analysis
MAX_INTERPRETATION_TOKENS = 80


def load_universal_vectors() -> Dict:
    """
    Load universal vectors from disk.

    Returns:
        Dict with structure:
        {
            'neg_to_trans': {'layer_1': array, 'layer_5': array, ...},
            'trans_to_pos': {'layer_1': array, ...},
            'neg_to_pos': {'layer_1': array, ...}
        }
    """
    vectors_file = VECTORS_DIR / "universal_vectors.npz"

    if not vectors_file.exists():
        raise FileNotFoundError(
            f"Universal vectors not found at {vectors_file}\n"
            "Please run 08_extract_universal_vector.py first"
        )

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

    # Load statistics
    stats_file = VECTORS_DIR / "statistics.json"
    with open(stats_file, 'r') as f:
        statistics = json.load(f)

    return {
        'vectors': vectors,
        'statistics': statistics['statistics']
    }


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


def interpret_directional_vector(selfie, direction_vector: np.ndarray,
                                 layer_idx: int, intensity: float,
                                 prompt: InterpretationPrompt) -> str:
    """Interpret a directional vector at a specific intensity."""
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


def test_universal_vectors(selfie, prompt: InterpretationPrompt,
                           vectors: Dict, layer_indices: List[int]) -> Dict:
    """Test universal vectors at high intensities."""
    print("\n🔬 Testing universal vectors at high intensities...")

    results = {
        'vector_type': 'universal',
        'intensities_tested': HIGH_INTENSITIES,
        'interpretations': {}
    }

    vector_types = ['neg_to_trans', 'trans_to_pos', 'neg_to_pos']

    for vector_type in vector_types:
        print(f"\n  Testing {vector_type}...")
        results['interpretations'][vector_type] = {}

        for layer_idx in tqdm(layer_indices, desc="    Layers", leave=False):
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


def save_interpretation_results(results: Dict):
    """Save interpretation results."""
    output_file = OUTPUT_DIR / "universal_vector_interpretations.json"

    # Convert integer keys to strings for JSON compatibility
    results_json = {
        'vector_type': results['vector_type'],
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

    print(f"\n  ✅ Saved interpretations: {output_file}")


def print_summary(results: Dict, layer_indices: List[int]):
    """Print summary of extreme interpretations."""
    print("\n" + "=" * 80)
    print("UNIVERSAL VECTOR INTERPRETATIONS - FOCUSED RANGE")
    print("=" * 80)

    middle_layer_idx = layer_indices[len(layer_indices) // 2]
    extreme_negative = HIGH_INTENSITIES[0]  # Min intensity
    extreme_positive = HIGH_INTENSITIES[-1]  # Max intensity

    vector_types = ['neg_to_trans', 'trans_to_pos', 'neg_to_pos']

    for vector_type in vector_types:
        print(f"\n{'='*80}")
        print(f"{vector_type.upper().replace('_', ' ')} (Layer {middle_layer_idx})")
        print(f"{'='*80}")

        layer_results = results['interpretations'][vector_type][middle_layer_idx]

        # Find extreme intensities
        neg_result = next((r for r in layer_results if r['intensity'] == extreme_negative), None)
        pos_result = next((r for r in layer_results if r['intensity'] == extreme_positive), None)
        zero_result = next((r for r in layer_results if r['intensity'] == 0), None)

        if neg_result:
            print(f"\n  Intensity {extreme_negative} (extreme negative pole):")
            print(f"    {neg_result['interpretation']}")

        if zero_result:
            print(f"\n  Intensity 0 (baseline, no injection):")
            print(f"    {zero_result['interpretation']}")

        if pos_result:
            print(f"\n  Intensity {extreme_positive} (extreme positive pole):")
            print(f"    {pos_result['interpretation']}")


def main():
    print("=" * 80)
    print("UNIVERSAL VECTOR INTERPRETATION AT HIGH INTENSITIES")
    print("=" * 80)

    # Load universal vectors
    print("\n📥 Loading universal vectors...")
    data = load_universal_vectors()
    vectors = data['vectors']
    statistics = data['statistics']
    print(f"  ✅ Loaded universal vectors")

    # Use focused layer indices (best performing from analysis)
    layer_indices = FOCUS_LAYERS
    print(f"\n📊 Testing layers: {layer_indices} (focused on best performers)")
    print(f"  Intensities: {HIGH_INTENSITIES}")

    # Load model
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

    # Test universal vectors
    interpretation_results = test_universal_vectors(
        selfie, prompt, vectors, layer_indices
    )

    # Save and display results
    save_interpretation_results(interpretation_results)
    print_summary(interpretation_results, layer_indices)

    print("\n" + "=" * 80)
    print("✅ UNIVERSAL VECTOR INTERPRETATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
