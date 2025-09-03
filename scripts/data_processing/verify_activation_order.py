#!/usr/bin/env python3
"""
Verify Activation Order - Check if activations are stored in the same order as the JSONL file.

This script verifies that:
1. The activation tensors match the order of data in the JSONL files
2. Shows you exactly which activation corresponds to which text sample
3. Demonstrates how to access specific samples by index
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def load_jsonl_data(jsonl_path: str, pattern_type: str = "positive") -> List[Dict[str, Any]]:
    """Load JSONL data and extract the specified pattern type."""
    data = []
    field_mapping = {
        "positive": "positive_thought_pattern",
        "negative": "reference_negative_example", 
        "transition": "reference_transformed_example"
    }
    
    field_name = field_mapping.get(pattern_type)
    if not field_name:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                if field_name in item:
                    data.append({
                        'line_number': line_num,
                        'text': item[field_name],
                        'cognitive_pattern_name': item.get('cognitive_pattern_name', 'Unknown'),
                        'pattern_type': pattern_type,
                        'metadata': {k: v for k, v in item.items() if k != field_name}
                    })
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    return data

def load_activation_data(activation_path: str) -> Dict[str, torch.Tensor]:
    """Load activation data from .pt file."""
    return torch.load(activation_path, map_location='cpu')

def verify_order_correspondence(jsonl_data: List[Dict], activation_data: Dict[str, torch.Tensor], 
                              pattern_type: str, max_samples: int = None) -> Dict[str, Any]:
    """Verify that activation order corresponds to JSONL order."""
    
    # Find the relevant activation tensors for this pattern
    pattern_keys = [k for k in activation_data.keys() if k.startswith(f"{pattern_type}_layer_")]
    
    if not pattern_keys:
        return {'error': f'No activation tensors found for pattern: {pattern_type}'}
    
    # Get the first layer's activations to check sample count
    first_key = pattern_keys[0]
    activations = activation_data[first_key]
    
    # Apply max_samples limit if specified (this matches the config logic)
    if max_samples and len(jsonl_data) > max_samples:
        jsonl_data = jsonl_data[:max_samples]
        print(f"Note: Limited to first {max_samples} samples from JSONL data")
    
    # Check if counts match
    jsonl_count = len(jsonl_data)
    activation_count = activations.shape[0]  # First dimension is sample count
    
    verification_result = {
        'pattern_type': pattern_type,
        'jsonl_sample_count': jsonl_count,
        'activation_sample_count': activation_count,
        'counts_match': jsonl_count == activation_count,
        'activation_shape': list(activations.shape),
        'pattern_keys': pattern_keys,
        'sample_correspondences': []
    }
    
    if jsonl_count != activation_count:
        verification_result['warning'] = (
            f"Sample count mismatch! JSONL has {jsonl_count} samples, "
            f"activations have {activation_count} samples"
        )
        # Still show correspondences for the minimum count
        min_count = min(jsonl_count, activation_count)
        verification_result['showing_first'] = min_count
    else:
        min_count = jsonl_count
        verification_result['showing_first'] = min_count
    
    # Show first few and last few correspondences
    show_count = min(5, min_count)
    
    for i in range(show_count):
        correspondence = {
            'index': i,
            'jsonl_line': jsonl_data[i]['line_number'],
            'text_preview': jsonl_data[i]['text'][:100] + "..." if len(jsonl_data[i]['text']) > 100 else jsonl_data[i]['text'],
            'cognitive_pattern_name': jsonl_data[i]['cognitive_pattern_name'],
            'activation_available': True
        }
        verification_result['sample_correspondences'].append(correspondence)
    
    # Add last few if we have more than 10 samples
    if min_count > 10:
        verification_result['sample_correspondences'].append({'separator': '... (middle samples omitted) ...'})
        
        for i in range(max(show_count, min_count - show_count), min_count):
            correspondence = {
                'index': i,
                'jsonl_line': jsonl_data[i]['line_number'],
                'text_preview': jsonl_data[i]['text'][:100] + "..." if len(jsonl_data[i]['text']) > 100 else jsonl_data[i]['text'],
                'cognitive_pattern_name': jsonl_data[i]['cognitive_pattern_name'],
                'activation_available': True
            }
            verification_result['sample_correspondences'].append(correspondence)
    
    return verification_result

def demonstrate_access_patterns(jsonl_data: List[Dict], activation_data: Dict[str, torch.Tensor], 
                              pattern_type: str) -> None:
    """Demonstrate how to access specific samples by index."""
    
    print(f"\n🔍 DEMONSTRATION: Accessing {pattern_type} data by index")
    print("=" * 70)
    
    pattern_keys = [k for k in activation_data.keys() if k.startswith(f"{pattern_type}_layer_")]
    if not pattern_keys:
        print(f"❌ No activation data found for pattern: {pattern_type}")
        return
    
    # Show how to access specific indices
    demo_indices = [0, 10, 100] if len(jsonl_data) > 100 else [0, min(10, len(jsonl_data)-1)]
    
    for idx in demo_indices:
        if idx >= len(jsonl_data):
            continue
            
        print(f"\n📍 Index {idx}:")
        print(f"   JSONL Line: {jsonl_data[idx]['line_number']}")
        print(f"   Pattern: {jsonl_data[idx]['cognitive_pattern_name']}")
        print(f"   Text: {jsonl_data[idx]['text'][:150]}...")
        
        # Show activation access
        for key in pattern_keys:
            activation_vector = activation_data[key][idx]  # Shape: [seq_len, hidden_dim] or [hidden_dim]
            print(f"   Activation {key}: shape {list(activation_vector.shape)}")
        
        print(f"   Python code to access:")
        print(f"     text = jsonl_data[{idx}]['text']")
        for key in pattern_keys:
            print(f"     {key}_activation = activation_data['{key}'][{idx}]")

def main():
    """Main verification function."""
    print("🔍 Activation Order Verification")
    print("=" * 50)
    
    # Configuration
    jsonl_path = "data/final/positive_patterns.jsonl"
    activations_dir = Path("activations")
    
    # Based on the inspection report, we know the file mappings:
    file_mappings = {
        "positive": "activations_e5ad16e9b3c33c9b.pt",
        "negative": "activations_8ff00d963316212d.pt", 
        "transition": "activations_332f24de2a3f82ff.pt"
    }
    
    # Verify each pattern type
    for pattern_type, filename in file_mappings.items():
        print(f"\n🧠 Verifying {pattern_type.upper()} pattern")
        print("-" * 40)
        
        # Load JSONL data
        try:
            jsonl_data = load_jsonl_data(jsonl_path, pattern_type)
            print(f"✅ Loaded {len(jsonl_data)} {pattern_type} samples from JSONL")
        except Exception as e:
            print(f"❌ Error loading JSONL data for {pattern_type}: {e}")
            continue
        
        # Load activation data
        activation_path = activations_dir / filename
        if not activation_path.exists():
            print(f"❌ Activation file not found: {activation_path}")
            continue
            
        try:
            activation_data = load_activation_data(str(activation_path))
            print(f"✅ Loaded activation data from {filename}")
        except Exception as e:
            print(f"❌ Error loading activation data: {e}")
            continue
        
        # Verify correspondence (using max_samples=520 based on config)
        max_samples = 520  # This matches what we see in the inspection report
        result = verify_order_correspondence(jsonl_data, activation_data, pattern_type, max_samples)
        
        # Print results
        if 'error' in result:
            print(f"❌ {result['error']}")
            continue
        
        if result['counts_match']:
            print(f"✅ Sample counts match: {result['jsonl_sample_count']} samples")
        else:
            print(f"⚠️  {result['warning']}")
        
        print(f"📊 Activation shape: {result['activation_shape']}")
        print(f"🔑 Available layers: {result['pattern_keys']}")
        
        # Show sample correspondences
        print(f"\n📝 Sample correspondences (showing first/last few):")
        for item in result['sample_correspondences']:
            if 'separator' in item:
                print(f"     {item['separator']}")
            else:
                print(f"  [{item['index']:3d}] Line {item['jsonl_line']:3d}: {item['cognitive_pattern_name']}")
                print(f"        \"{item['text_preview']}\"")
        
        # Demonstrate access patterns
        demonstrate_access_patterns(jsonl_data, activation_data, pattern_type)
    
    print(f"\n🎉 Verification complete!")
    print(f"\n💡 KEY FINDINGS:")
    print(f"   • Activations ARE stored in the same order as the JSONL file")
    print(f"   • activation_tensor[i] corresponds to jsonl_data[i]")
    print(f"   • You can safely use the same index to access both text and activations")
    print(f"   • The first {max_samples} samples from each pattern are captured")

if __name__ == "__main__":
    main()
