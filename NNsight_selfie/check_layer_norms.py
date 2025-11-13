#!/usr/bin/env python3
"""
Check typical activation magnitudes across different layers.
This helps us understand if we need layer-specific normalization.
"""

# FOR AMD GPU
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["TORCH_USE_HIP_DSA"] = "1"

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from nnsight_selfie import ModelAgnosticSelfie, get_optimal_device

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
TEST_CONCEPTS = ["car", "horse", "happiness", "sadness", "hot", "cold"]
SAMPLE_LAYERS = list(range(1, 34))  # Include all layers from 1 to 33

print("="*80)
print("Layer-Specific Activation Magnitude Analysis")
print("="*80)

# Load model
print(f"\n📥 Loading {MODEL_NAME}...")
device = get_optimal_device()
selfie = ModelAgnosticSelfie(MODEL_NAME, dtype=torch.bfloat16, load_in_8bit=False)
print(f"✅ Model loaded on {device}")
print(f"   Total layers: {len(selfie.layer_paths)}")

# Extract activations for all concepts across sample layers
print(f"\n🧮 Extracting activations for {len(TEST_CONCEPTS)} concepts...")
print(f"   Concepts: {', '.join(TEST_CONCEPTS)}")
print(f"   Layers: {SAMPLE_LAYERS}")

activations = selfie.get_concept_activations(
    concepts=TEST_CONCEPTS,
    layer_indices=SAMPLE_LAYERS,
    use_chat_template=True  # Use chat template like in the notebook
)

# Analyze norms per layer
print("\n" + "="*80)
print("ACTIVATION MAGNITUDE ANALYSIS")
print("="*80)

layer_stats = {}

for layer_idx in SAMPLE_LAYERS:
    norms = []
    for concept in TEST_CONCEPTS:
        vec = activations[concept][layer_idx]
        norm = torch.norm(vec).item()
        norms.append(norm)

    layer_stats[layer_idx] = {
        'mean': np.mean(norms),
        'std': np.std(norms),
        'min': np.min(norms),
        'max': np.max(norms)
    }

# Print results
print("\n📊 Norm Statistics by Layer:\n")
print(f"{'Layer':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Range':<10}")
print("-" * 80)

for layer_idx in SAMPLE_LAYERS:
    stats = layer_stats[layer_idx]
    range_val = stats['max'] - stats['min']
    print(f"{layer_idx:<8} {stats['mean']:<10.2f} {stats['std']:<10.2f} "
          f"{stats['min']:<10.2f} {stats['max']:<10.2f} {range_val:<10.2f}")

# Check if normalization is needed
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

mean_norms = [stats['mean'] for stats in layer_stats.values()]
mean_range = max(mean_norms) - min(mean_norms)
mean_variation = np.std(mean_norms) / np.mean(mean_norms) * 100  # CV%

print(f"\n📈 Cross-layer variation:")
print(f"   Mean norm range: {min(mean_norms):.2f} - {max(mean_norms):.2f} (Δ={mean_range:.2f})")
print(f"   Coefficient of variation: {mean_variation:.1f}%")

if mean_variation > 10:
    print("\n⚠️  HIGH VARIATION DETECTED!")
    print("   Recommendation: Normalize vectors to layer-specific norms when:")
    print("   1. Applying learned paths across different layers")
    print("   2. Using multi-layer paths")
    print("   3. Generalizing transformations to new concept pairs")
else:
    print("\n✅ LOW VARIATION")
    print("   Current approach (no layer-specific normalization) is likely fine.")

# Print per-concept details for one sample layer
print(f"\n📋 Detailed breakdown for layer {SAMPLE_LAYERS[len(SAMPLE_LAYERS)//2]}:")
sample_layer = SAMPLE_LAYERS[len(SAMPLE_LAYERS)//2]
print(f"\n{'Concept':<15} {'Norm':<10}")
print("-" * 30)
for concept in TEST_CONCEPTS:
    vec = activations[concept][sample_layer]
    norm = torch.norm(vec).item()
    print(f"{concept:<15} {norm:<10.2f}")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)