#!/usr/bin/env python3
"""
Script to examine the structure of activation files and metadata
"""

import torch
import json
import pandas as pd
from pathlib import Path
import numpy as np

def examine_activation_files():
    """Examine the structure of the activation files"""
    
    activation_dir = Path("/Users/ivanculo/Desktop/Projects/turn_point/activations")
    metadata_path = Path("/Users/ivanculo/Desktop/Projects/turn_point/data/enriched_metadata.json")
    
    # Load metadata
    print("Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} entries in metadata")
    
    # Examine first few entries structure
    print("\nFirst entry structure:")
    first_entry = metadata[0]
    for key in first_entry.keys():
        if isinstance(first_entry[key], str) and len(first_entry[key]) > 100:
            print(f"{key}: [TEXT TOO LONG - {len(first_entry[key])} chars]")
        else:
            print(f"{key}: {first_entry[key]}")
    
    # Check activation files
    activation_files = list(activation_dir.glob("*.pt"))
    print(f"\nFound {len(activation_files)} activation files:")
    
    for file_path in activation_files:
        print(f"\nExamining {file_path.name}...")
        try:
            data = torch.load(file_path, map_location='cpu')
            print(f"Data type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}, dtype: {value.dtype}")
                    else:
                        print(f"  {key}: {type(value)}")
            elif isinstance(data, torch.Tensor):
                print(f"Tensor shape: {data.shape}, dtype: {data.dtype}")
            elif isinstance(data, list):
                print(f"List length: {len(data)}")
                if len(data) > 0:
                    print(f"First element type: {type(data[0])}")
                    if isinstance(data[0], torch.Tensor):
                        print(f"First tensor shape: {data[0].shape}")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
    
    # Check cognitive patterns mapping
    patterns_path = Path("/Users/ivanculo/Desktop/Projects/turn_point/data/cognitive_patterns_short14.csv")
    patterns_df = pd.read_csv(patterns_path)
    print(f"\nCognitive patterns found: {len(patterns_df)}")
    print("Patterns:")
    for idx, row in patterns_df.iterrows():
        print(f"  {row['Concept Name']}: {row['Cognitive Pattern']}")
    
    # Check which cognitive patterns are represented in the metadata
    print("\nCognitive patterns in metadata:")
    pattern_counts = {}
    for entry in metadata:
        pattern_name = entry.get('cognitive_pattern_name', 'Unknown')
        pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
    
    for pattern, count in sorted(pattern_counts.items()):
        print(f"  {pattern}: {count} entries")

if __name__ == "__main__":
    examine_activation_files()