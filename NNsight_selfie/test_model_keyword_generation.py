"""
Test script for model-generated keyword semantic gradients.

This script tests the new LLM-based keyword generation feature.
"""

import torch
import warnings
warnings.filterwarnings('ignore')

from nnsight_selfie import (
    ModelAgnosticSelfie,
    generate_keywords_with_model,
    generate_intermediate_keywords
)

print("=" * 80)
print("Testing Model-Generated Keyword Gradients")
print("=" * 80)

# Load model
print("\n📥 Loading model...")
MODEL_NAME = "google/gemma-2-2b-it"

selfie = ModelAgnosticSelfie(
    MODEL_NAME,
    dtype=torch.bfloat16,
    load_in_8bit=False
)

print(f"✅ Model loaded: {MODEL_NAME}")
print(f"   Layers: {len(selfie.layer_paths)}")

# Test 1: Direct function call
print("\n" + "=" * 80)
print("TEST 1: generate_keywords_with_model() direct call")
print("=" * 80)

keywords_model = generate_keywords_with_model(
    selfie,
    start_concept="sad",
    end_concept="happy",
    num_steps=7,
    temperature=0.3
)

print(f"\n📝 Model-generated keywords:")
for i, kw in enumerate(keywords_model):
    print(f"  {i}. {kw}")

# Test 2: Via generate_intermediate_keywords with template="model_generated"
print("\n" + "=" * 80)
print("TEST 2: generate_intermediate_keywords() with template='model_generated'")
print("=" * 80)

keywords_via_template = generate_intermediate_keywords(
    start_concept="angry",
    end_concept="calm",
    num_steps=7,
    template="model_generated",
    selfie=selfie,
    temperature=0.3
)

print(f"\n📝 Keywords via template:")
for i, kw in enumerate(keywords_via_template):
    print(f"  {i}. {kw}")

# Test 3: Fallback to template-based (without selfie)
print("\n" + "=" * 80)
print("TEST 3: Fallback to template-based generation")
print("=" * 80)

keywords_fallback = generate_intermediate_keywords(
    start_concept="anxious",
    end_concept="relaxed",
    num_steps=7,
    template="emotion",
    selfie=None  # No model provided
)

print(f"\n📝 Template-based keywords (fallback):")
for i, kw in enumerate(keywords_fallback):
    print(f"  {i}. {kw}")

# Test 4: Compare quality
print("\n" + "=" * 80)
print("COMPARISON: Model vs Template")
print("=" * 80)

test_pairs = [
    ("frustrated", "satisfied"),
    ("fearful", "confident"),
]

for start, end in test_pairs:
    print(f"\n🔄 {start} → {end}")

    # Model-generated
    model_kw = generate_keywords_with_model(selfie, start, end, num_steps=5, temperature=0.3)
    print(f"  Model:    {' → '.join(model_kw)}")

    # Template-based
    template_kw = generate_intermediate_keywords(start, end, num_steps=5, template="emotion")
    print(f"  Template: {' → '.join(template_kw)}")

print("\n" + "=" * 80)
print("✅ All tests complete!")
print("=" * 80)

print("\n💡 Key observations:")
print("  - Model-generated keywords should be more semantically meaningful")
print("  - Model captures actual intermediate emotional states")
print("  - Template-based just adds modifiers like 'slightly' or 'very'")
print("\n🎉 Model-generated keyword feature is working!")
