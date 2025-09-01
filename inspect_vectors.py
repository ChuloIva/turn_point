#!/usr/bin/env python3

import torch
import json
import numpy as np
from pathlib import Path

# Load activations
base_path = Path("/Users/ivanculo/Desktop/Projects/turn_point")
activations_dir = base_path / "activations"

negative_activations = torch.load(
    activations_dir / "activations_8ff00d963316212d.pt", 
    map_location='cpu'
)
positive_activations = torch.load(
    activations_dir / "activations_e5ad16e9b3c33c9b.pt", 
    map_location='cpu'
)

# Load metadata
with open(base_path / "data" / "final" / "enriched_metadata.json", 'r') as f:
    metadata = json.load(f)

# Create mapping from pattern names to indices
pattern_indices = {}
for i, entry in enumerate(metadata):
    pattern_name = entry['bad_good_narratives_match']['cognitive_pattern_name_from_bad_good']
    if pattern_name not in pattern_indices:
        pattern_indices[pattern_name] = []
    pattern_indices[pattern_name].append(i)

# Pick first pattern and layer 17
first_pattern = list(pattern_indices.keys())[0]
indices = pattern_indices[first_pattern][:5]  # First 5 samples
layer = 17

print(f"Inspecting pattern: {first_pattern}")
print(f"Using indices: {indices}")

neg_data = negative_activations[f'negative_layer_{layer}'][indices]
pos_data = positive_activations[f'positive_layer_{layer}'][indices]

print(f"Negative shape: {neg_data.shape}")
print(f"Positive shape: {pos_data.shape}")

# Test different token positions
positions_to_try = [-1, -2, -3, 1, 2, 3, 4, 5]

for pos in positions_to_try:
    neg_tokens = neg_data[:, pos, :]
    pos_tokens = pos_data[:, pos, :]
    
    # Compute means
    neg_mean = neg_tokens.mean(dim=0)
    pos_mean = pos_tokens.mean(dim=0)
    
    # Compute recovery direction
    recovery_direction = pos_mean - neg_mean
    
    print(f"\nToken position {pos}:")
    print(f"  Negative mean stats: mean={neg_mean.mean().item():.6f}, std={neg_mean.std().item():.6f}")
    print(f"  Positive mean stats: mean={pos_mean.mean().item():.6f}, std={pos_mean.std().item():.6f}")
    print(f"  Recovery direction stats: mean={recovery_direction.mean().item():.6f}, std={recovery_direction.std().item():.6f}")
    print(f"  Non-zero elements in recovery: {(recovery_direction != 0).sum().item()} / {recovery_direction.numel()}")

# Try averaging multiple positions
print(f"\nAveraging positions 1-5:")
neg_avg = neg_data[:, 1:6, :].mean(dim=1).mean(dim=0)  
pos_avg = pos_data[:, 1:6, :].mean(dim=1).mean(dim=0)
recovery_avg = pos_avg - neg_avg

print(f"  Negative avg stats: mean={neg_avg.mean().item():.6f}, std={neg_avg.std().item():.6f}")
print(f"  Positive avg stats: mean={pos_avg.mean().item():.6f}, std={pos_avg.std().item():.6f}")
print(f"  Recovery direction stats: mean={recovery_avg.mean().item():.6f}, std={recovery_avg.std().item():.6f}")
print(f"  Non-zero elements: {(recovery_avg != 0).sum().item()} / {recovery_avg.numel()}")

print(f"\nFirst 10 values of recovery direction (avg 1-5): {recovery_avg[:10].tolist()}")