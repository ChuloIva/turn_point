"""
Utilities for loading and managing cached activations.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ACTIVATIONS_DIR = PROJECT_ROOT / "activations/therapeutic_reframing"
CACHE_DIR = ACTIVATIONS_DIR / "cache"


def load_activation_index() -> Dict:
    """Load the activation index metadata."""
    with open(CACHE_DIR / "activation_index.json", 'r') as f:
        return json.load(f)


def load_pattern_activations(pattern_slug: str, device: str = 'cpu') -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Load all activations for a specific pattern.

    Args:
        pattern_slug: Pattern directory name (e.g., 'suicidal_planning_and_rationalization')
        device: Device to load tensors to

    Returns:
        Dict with keys 'negative', 'transformed', 'positive'
        Values are Dict[layer_idx -> tensor of shape (n_examples, hidden_dim)]
    """
    pattern_dir = ACTIVATIONS_DIR / "by_pattern" / pattern_slug

    activations = {}

    for text_type in ['negative', 'transformed', 'positive']:
        file_path = pattern_dir / f"{text_type}_examples.npz"
        data = np.load(file_path)

        # Convert to torch tensors
        activations[text_type] = {
            int(key.split('_')[1]): torch.from_numpy(data[key]).to(device)
            for key in data.files
        }

    return activations


def load_single_example_activations(pattern_slug: str, example_idx: int,
                                   device: str = 'cpu') -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Load activations for a single example from a pattern.

    Returns:
        Dict with keys 'negative', 'transformed', 'positive'
        Values are Dict[layer_idx -> tensor of shape (hidden_dim,)]
    """
    all_activations = load_pattern_activations(pattern_slug, device)

    return {
        text_type: {
            layer_idx: tensor[example_idx]
            for layer_idx, tensor in layer_acts.items()
        }
        for text_type, layer_acts in all_activations.items()
    }


def get_layer_indices() -> List[int]:
    """Get the strategic layer indices used for extraction."""
    index = load_activation_index()
    return index['strategic_layers']


def get_pattern_slugs() -> List[str]:
    """Get list of all pattern slugs."""
    index = load_activation_index()
    return [info['pattern_slug'] for info in index['patterns'].values()]


def get_pattern_info(pattern_name: str) -> Dict:
    """Get metadata for a specific pattern."""
    index = load_activation_index()
    return index['patterns'][pattern_name]
