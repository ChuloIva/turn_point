#!/usr/bin/env python3
"""
Script to convert activation files from float32 to float16 for memory savings.
"""

import torch
import os
from pathlib import Path
import json

def convert_activations_to_fp16(input_path: str, output_path: str = None):
    """
    Convert activation tensors from float32 to float16.
    
    Args:
        input_path: Path to the .pt file with float32 activations
        output_path: Path to save float16 version (optional)
    """
    if output_path is None:
        # Create output path by adding _fp16 suffix
        path = Path(input_path)
        output_path = path.parent / f"{path.stem}_fp16{path.suffix}"
    
    print(f"Loading activations from: {input_path}")
    
    # Load the activations
    activations = torch.load(input_path, map_location='cpu')
    
    print(f"Original data type: {type(activations)}")
    
    # Convert tensors to float16
    if isinstance(activations, dict):
        converted_activations = {}
        total_memory_before = 0
        total_memory_after = 0
        
        for key, tensor in activations.items():
            if isinstance(tensor, torch.Tensor):
                # Calculate memory usage
                memory_before = tensor.numel() * tensor.element_size()
                total_memory_before += memory_before
                
                # Convert to float16
                converted_tensor = tensor.half()  # Convert to float16
                
                memory_after = converted_tensor.numel() * converted_tensor.element_size()
                total_memory_after += memory_after
                
                converted_activations[key] = converted_tensor
                
                print(f"  {key}: {tensor.shape} {tensor.dtype} -> {converted_tensor.dtype}")
                print(f"    Memory: {memory_before/1024/1024:.1f}MB -> {memory_after/1024/1024:.1f}MB")
            else:
                # Keep non-tensor data as is
                converted_activations[key] = tensor
        
        print(f"\nTotal memory reduction: {total_memory_before/1024/1024:.1f}MB -> {total_memory_after/1024/1024:.1f}MB")
        print(f"Space saved: {(total_memory_before - total_memory_after)/1024/1024:.1f}MB ({((total_memory_before - total_memory_after)/total_memory_before)*100:.1f}%)")
        
    else:
        # Handle case where activations is a single tensor
        converted_activations = activations.half()
    
    # Save converted activations
    print(f"Saving converted activations to: {output_path}")
    torch.save(converted_activations, output_path)
    
    return output_path

def convert_all_activations_in_directory(directory: str = "activations"):
    """Convert all activation files in a directory to float16."""
    activation_dir = Path(directory)
    
    if not activation_dir.exists():
        print(f"Directory {directory} does not exist!")
        return
    
    activation_files = list(activation_dir.glob("activations_*.pt"))
    
    if not activation_files:
        print(f"No activation files found in {directory}")
        return
    
    print(f"Found {len(activation_files)} activation files to convert:")
    
    converted_files = []
    for file_path in activation_files:
        print(f"\n--- Converting {file_path.name} ---")
        try:
            output_path = convert_activations_to_fp16(str(file_path))
            converted_files.append(output_path)
        except Exception as e:
            print(f"Error converting {file_path.name}: {e}")
    
    print(f"\n✅ Successfully converted {len(converted_files)} files:")
    for file_path in converted_files:
        print(f"  - {file_path}")

def verify_conversion(original_path: str, converted_path: str, tolerance: float = 1e-3):
    """
    Verify that the conversion maintained reasonable accuracy.
    
    Args:
        original_path: Path to original float32 file
        converted_path: Path to converted float16 file
        tolerance: Maximum allowed difference (due to precision loss)
    """
    print(f"Verifying conversion accuracy...")
    
    original = torch.load(original_path, map_location='cpu')
    converted = torch.load(converted_path, map_location='cpu')
    
    if isinstance(original, dict) and isinstance(converted, dict):
        for key in original.keys():
            if isinstance(original[key], torch.Tensor):
                # Convert back to float32 for comparison
                converted_back = converted[key].float()
                
                # Calculate max absolute difference
                max_diff = torch.max(torch.abs(original[key] - converted_back)).item()
                
                print(f"  {key}: max difference = {max_diff:.6f}")
                
                if max_diff > tolerance:
                    print(f"    ⚠️  Warning: Large difference detected (>{tolerance})")
                else:
                    print(f"    ✅ Conversion accurate (within {tolerance})")

if __name__ == "__main__":
    # Convert all activation files in the activations directory
    convert_all_activations_in_directory()
    
    # Example: Verify one conversion
    # verify_conversion(
    #     "activations/activations_8ff00d963316212d.pt",
    #     "activations/activations_8ff00d963316212d_fp16.pt"
    # )
