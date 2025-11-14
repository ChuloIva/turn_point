#!/usr/bin/env python3
"""
Phase 1.2: Extract Multi-Layer Activations with Chat Template

Extracts activations from last token for 3 text types:
- Negative examples
- Transformed examples (intermediate landmarks)
- Positive examples

Uses chat template to properly format text as user messages.
Strategic layers: [1, 5, 7, 11, 15, 22, 27, 29]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from nnsight_selfie import ModelAgnosticSelfie, get_optimal_device

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/therapeutic_reframing/processed"
ACTIVATIONS_DIR = PROJECT_ROOT / "activations/therapeutic_reframing"
CACHE_DIR = ACTIVATIONS_DIR / "cache"

# Configuration
STRATEGIC_LAYERS = [1, 5, 7, 11, 15, 22, 27, 29]
MODEL_NAME = "google/gemma-3-4b-it"


def extract_with_chat_template(selfie, text: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    """
    Extract activation from last token with chat template applied.

    Args:
        selfie: ModelAgnosticSelfie instance
        text: Text to extract activation from
        layer_indices: List of layer indices to extract from

    Returns:
        Dict[layer_idx -> tensor of shape [hidden_dim]]
    """
    # Format as user message
    messages = [{"role": "user", "content": text}]
    formatted = selfie.model.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False
    )

    # Extract from last token of each layer
    activations = selfie.get_concept_activations(
        concepts=[formatted],
        layer_indices=layer_indices,
        use_chat_template=False  # Already applied manually
    )

    # Return dict mapping layer_idx -> tensor
    # activations is Dict[concept_str -> Dict[layer_idx -> tensor]]
    return {layer_idx: tensor.squeeze() for layer_idx, tensor in activations[formatted].items()}


def process_pattern_batch(selfie, df_pattern: pd.DataFrame, pattern_name: str,
                          layer_indices: List[int]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Process all examples for a single cognitive pattern.

    Returns:
        Dict with keys 'negative', 'transformed', 'positive'
        Values are Dict[layer_idx -> ndarray of shape (n_examples, hidden_dim)]
    """
    n_examples = len(df_pattern)
    # Get hidden size (handle Gemma3's nested config)
    if hasattr(selfie.model.config, 'hidden_size'):
        hidden_dim = selfie.model.config.hidden_size
    elif hasattr(selfie.model.config, 'text_config'):
        hidden_dim = selfie.model.config.text_config.hidden_size
    else:
        raise ValueError("Could not find hidden_size in model config")

    # Initialize storage
    activations = {
        'negative': {layer_idx: [] for layer_idx in layer_indices},
        'transformed': {layer_idx: [] for layer_idx in layer_indices},
        'positive': {layer_idx: [] for layer_idx in layer_indices}
    }

    print(f"\n  Processing {pattern_name} ({n_examples} examples)...")

    for idx, row in tqdm(df_pattern.iterrows(), total=n_examples, desc=f"  {pattern_name}"):
        # Extract all 3 types
        neg_acts = extract_with_chat_template(selfie, row['negative_text'], layer_indices)
        trans_acts = extract_with_chat_template(selfie, row['transformed_text'], layer_indices)
        pos_acts = extract_with_chat_template(selfie, row['positive_text'], layer_indices)

        # Store by layer (convert bfloat16 to float32 for numpy compatibility)
        for layer_idx in layer_indices:
            activations['negative'][layer_idx].append(neg_acts[layer_idx].float().cpu().numpy())
            activations['transformed'][layer_idx].append(trans_acts[layer_idx].float().cpu().numpy())
            activations['positive'][layer_idx].append(pos_acts[layer_idx].float().cpu().numpy())

    # Convert lists to arrays: shape (n_examples, hidden_dim)
    for text_type in ['negative', 'transformed', 'positive']:
        for layer_idx in layer_indices:
            activations[text_type][layer_idx] = np.stack(activations[text_type][layer_idx], axis=0)

    return activations


def save_pattern_activations(activations: Dict, pattern_name: str, pattern_dir: Path):
    """Save activations for one pattern to compressed .npz files."""
    pattern_dir.mkdir(parents=True, exist_ok=True)

    for text_type in ['negative', 'transformed', 'positive']:
        # Create dict with layer_{idx} keys for npz
        save_dict = {f"layer_{layer_idx}": arr
                     for layer_idx, arr in activations[text_type].items()}

        output_path = pattern_dir / f"{text_type}_examples.npz"
        np.savez_compressed(output_path, **save_dict)
        print(f"    ✅ Saved: {text_type}_examples.npz")


def main():
    print("=" * 80)
    print("PHASE 1.2: Multi-Layer Activation Extraction")
    print("=" * 80)

    # Load dataset
    print(f"\n📥 Loading dataset...")
    df = pd.read_csv(DATA_DIR / "pattern_metadata.csv")
    print(f"✅ Loaded {len(df)} examples from {df['pattern_name'].nunique()} patterns")

    # Load strategic layers
    with open(DATA_DIR / "strategic_layers.json", 'r') as f:
        config = json.load(f)
    layer_indices = config['strategic_layers']
    print(f"\n📊 Strategic layers: {layer_indices}")
    print(f"  Early: {config['layer_rationale']['early']}")
    print(f"  Middle: {config['layer_rationale']['middle']}")
    print(f"  Late: {config['layer_rationale']['late']}")

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

    # Get hidden size (handle Gemma3's nested config)
    if hasattr(selfie.model.config, 'hidden_size'):
        hidden_size = selfie.model.config.hidden_size
    elif hasattr(selfie.model.config, 'text_config'):
        hidden_size = selfie.model.config.text_config.hidden_size
    else:
        raise ValueError("Could not find hidden_size in model config")

    print(f"  Hidden size: {hidden_size}")
    print(f"  Vocab size: {len(selfie.model.tokenizer):,}")

    # Process each pattern
    print(f"\n🔄 Extracting activations for all patterns...")
    print(f"  Total: {len(df)} examples × 3 text types × {len(layer_indices)} layers")

    activation_index = {}

    for pattern_name in sorted(df['pattern_name'].unique()):
        df_pattern = df[df['pattern_name'] == pattern_name]
        pattern_slug = pattern_name.lower().replace(' ', '_').replace('&', 'and')
        pattern_dir = ACTIVATIONS_DIR / "by_pattern" / pattern_slug

        # Extract activations
        activations = process_pattern_batch(selfie, df_pattern, pattern_name, layer_indices)

        # Save
        print(f"\n  💾 Saving activations for {pattern_name}...")
        save_pattern_activations(activations, pattern_name, pattern_dir)

        # Update index
        activation_index[pattern_name] = {
            'pattern_slug': pattern_slug,
            'directory': str(pattern_dir.relative_to(PROJECT_ROOT)),
            'n_examples': len(df_pattern),
            'files': {
                'negative': f"{pattern_slug}/negative_examples.npz",
                'transformed': f"{pattern_slug}/transformed_examples.npz",
                'positive': f"{pattern_slug}/positive_examples.npz"
            }
        }

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save activation index
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index_metadata = {
        'model_name': MODEL_NAME,
        'strategic_layers': layer_indices,
        'hidden_dim': hidden_size,
        'extraction_method': 'chat_template_last_token',
        'patterns': activation_index
    }

    with open(CACHE_DIR / "activation_index.json", 'w') as f:
        json.dump(index_metadata, f, indent=2)
    print(f"\n✅ Saved activation index: {CACHE_DIR / 'activation_index.json'}")

    # Summary
    print(f"\n" + "=" * 80)
    print(f"✅ Phase 1.2 Complete!")
    print(f"=" * 80)
    print(f"\n📊 Summary:")
    print(f"  Patterns processed: {len(activation_index)}")
    print(f"  Total examples: {len(df)}")
    print(f"  Text types per example: 3 (negative, transformed, positive)")
    print(f"  Layers per text: {len(layer_indices)}")
    print(f"  Total activations: {len(df) * 3 * len(layer_indices):,}")
    print(f"\n📁 Activations saved to: {ACTIVATIONS_DIR}")


if __name__ == "__main__":
    main()
