"""
Quick test for chat-templated activation extraction.

This script tests the new get_concept_activations() method with chat templates.
"""

import sys
sys.path.insert(0, '.')

import torch
from nnsight_selfie import ModelAgnosticSelfie

def test_chat_template_extraction():
    """Test chat-templated concept extraction."""

    print("=" * 60)
    print("Testing Chat Template Activation Extraction")
    print("=" * 60)

    # Load a small model for testing
    MODEL_NAME = "google/gemma-2-2b-it"

    print(f"\n📥 Loading {MODEL_NAME}...")
    selfie = ModelAgnosticSelfie(
        MODEL_NAME,
        dtype=torch.bfloat16,
        load_in_8bit=False
    )
    print("✅ Model loaded")

    # Test concepts
    concepts = ["happiness", "sadness", "anger"]
    test_layer = 15

    print(f"\n🧪 Testing concept extraction for: {concepts}")
    print(f"   Layer: {test_layer}")

    # Test 1: Without chat template
    print("\n" + "-" * 60)
    print("Test 1: WITHOUT chat template")
    print("-" * 60)

    try:
        result_no_chat = selfie.get_concept_activations(
            concepts=concepts,
            layer_indices=[test_layer],
            use_chat_template=False
        )

        for concept in concepts:
            vec = result_no_chat[concept][test_layer]
            print(f"  ✅ {concept:12s}: shape={vec.shape}, norm={torch.norm(vec):.2f}")

        print("✅ Test 1 passed")

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False

    # Test 2: With chat template
    print("\n" + "-" * 60)
    print("Test 2: WITH chat template")
    print("-" * 60)

    try:
        result_with_chat = selfie.get_concept_activations(
            concepts=concepts,
            layer_indices=[test_layer],
            use_chat_template=True
        )

        for concept in concepts:
            vec = result_with_chat[concept][test_layer]
            print(f"  ✅ {concept:12s}: shape={vec.shape}, norm={torch.norm(vec):.2f}")

        print("✅ Test 2 passed")

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False

    # Test 3: Show chat template formatting
    print("\n" + "-" * 60)
    print("Test 3: Chat template formatting with concept in model response")
    print("-" * 60)

    try:
        test_concept = "happiness"
        user_prompt = f"think about the {test_concept}"

        # Format with assistant response containing the concept
        tokenizer = selfie.model.tokenizer

        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": test_concept}
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            formatted = f"User: {user_prompt}\nAssistant: {test_concept}"

        # Find where the concept appears
        tokens = tokenizer.encode(formatted)
        concept_tokens = tokenizer.encode(test_concept, add_special_tokens=False)

        capture_pos = None
        for i in range(len(tokens) - len(concept_tokens), -1, -1):
            if tokens[i:i+len(concept_tokens)] == concept_tokens:
                capture_pos = i
                break

        print(f"  User prompt: '{user_prompt}'")
        print(f"  Formatted chat:")
        for line in formatted.split('\n'):
            print(f"    {line}")
        print(f"  Concept '{test_concept}' found at token position: {capture_pos}")
        print("✅ Test 3 passed")

    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False

    # Test 4: Compare activations
    print("\n" + "-" * 60)
    print("Test 4: Activation differences")
    print("-" * 60)

    for concept in concepts:
        vec_no_chat = result_no_chat[concept][test_layer]
        vec_with_chat = result_with_chat[concept][test_layer]

        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            vec_no_chat.flatten(),
            vec_with_chat.flatten(),
            dim=0
        ).item()

        print(f"  {concept:12s}: cosine_similarity={cos_sim:.4f}")

    print("✅ Test 4 passed")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_chat_template_extraction()
    sys.exit(0 if success else 1)