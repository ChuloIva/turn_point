#!/usr/bin/env python3
"""
Example usage of the Gemma SelfIE Adapter

This script demonstrates various ways to use the adapter for interpreting
pre-captured Gemma activations with SelfIE.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from selfie_gemma_integration.gemma_selfie_adapter import GemmaSelfieAdapter
from selfie_gemma_integration.config import *


def example_basic_usage():
    """Basic usage example - analyze one cache file."""
    print("🔬 Example 1: Basic Usage")
    print("-" * 30)
    
    # Initialize adapter
    adapter = GemmaSelfieAdapter()
    
    # Load cached activations
    cache_data = adapter.load_cached_activations()
    
    # Extract cognitive patterns
    patterns = adapter.extract_cognitive_patterns()
    
    # Load model for interpretation
    adapter.load_model_and_tokenizer()
    
    # Run analysis on first available cache
    cache_key = list(cache_data.keys())[0]
    results_file = adapter.run_full_analysis(
        cache_key=cache_key,
        layer=17,
        max_patterns=3
    )
    
    print(f"✅ Basic analysis complete. Results: {results_file}")


def example_custom_interpretation():
    """Example with custom interpretation template."""
    print("\n🎨 Example 2: Custom Interpretation Template")
    print("-" * 40)
    
    adapter = GemmaSelfieAdapter()
    
    # Load data
    cache_data = adapter.load_cached_activations()
    adapter.load_model_and_tokenizer()
    
    # Prepare activations
    cache_key = list(cache_data.keys())[0]
    prepared_data = adapter.prepare_activations_for_selfie(
        cache_key=cache_key,
        layer=17,
        max_patterns=2
    )
    
    # Use custom interpretation template focused on emotional states
    custom_template = INTERPRETATION_TEMPLATES["emotional_state"]
    
    results = adapter.interpret_activations_with_selfie(
        prepared_data=prepared_data,
        interpretation_template=custom_template,
        position='mean',  # Last token
        batch_size=1,
        max_new_tokens=150
    )
    
    # Save with custom filename
    results_file = adapter.save_interpretation_results(
        results, 
        filename="emotional_state_analysis"
    )
    
    print(f"✅ Custom interpretation complete. Results: {results_file}")


def example_multi_layer_analysis():
    """Example analyzing multiple layers."""
    print("\n🔍 Example 3: Multi-Layer Analysis")
    print("-" * 35)
    
    adapter = GemmaSelfieAdapter()
    
    # Load data
    cache_data = adapter.load_cached_activations()
    adapter.load_model_and_tokenizer()
    
    cache_key = list(cache_data.keys())[0]
    layers_to_analyze = [10, 15, 17, 20]
    
    all_results = []
    
    for layer in layers_to_analyze:
        print(f"   Analyzing layer {layer}...")
        
        try:
            # Prepare data for this layer
            prepared_data = adapter.prepare_activations_for_selfie(
                cache_key=cache_key,
                layer=layer,
                max_patterns=2
            )
            
            # Generate interpretations
            results = adapter.interpret_activations_with_selfie(
                prepared_data=prepared_data,
                position=-1,
                batch_size=1,
                max_new_tokens=30
            )
            
            all_results.extend(results)
            
        except Exception as e:
            print(f"     ❌ Layer {layer} failed: {e}")
    
    if all_results:
        # Save combined results
        results_file = adapter.save_interpretation_results(
            all_results,
            filename="multi_layer_analysis"
        )
        print(f"✅ Multi-layer analysis complete. Results: {results_file}")
    else:
        print("❌ No results generated")


def example_position_comparison():
    """Example comparing different token positions."""
    print("\n📍 Example 4: Token Position Comparison")
    print("-" * 38)
    
    adapter = GemmaSelfieAdapter()
    
    # Load data
    cache_data = adapter.load_cached_activations()
    adapter.load_model_and_tokenizer()
    
    cache_key = list(cache_data.keys())[0]
    positions = {"last": -1, "second_last": -2, "average": "mean"}
    
    all_results = []
    
    for pos_name, position in positions.items():
        print(f"   Analyzing {pos_name} position...")
        
        try:
            # Prepare data
            prepared_data = adapter.prepare_activations_for_selfie(
                cache_key=cache_key,
                layer=17,
                max_patterns=2
            )
            
            # Generate interpretations
            results = adapter.interpret_activations_with_selfie(
                prepared_data=prepared_data,
                position=position,
                batch_size=1,
                max_new_tokens=30
            )
            
            # Add position info to results
            for result in results:
                result['position_name'] = pos_name
            
            all_results.extend(results)
            
        except Exception as e:
            print(f"     ❌ Position {pos_name} failed: {e}")
    
    if all_results:
        results_file = adapter.save_interpretation_results(
            all_results,
            filename="position_comparison"
        )
        print(f"✅ Position comparison complete. Results: {results_file}")


def example_specific_patterns():
    """Example analyzing specific cognitive patterns."""
    print("\n🧠 Example 5: Specific Pattern Analysis")
    print("-" * 36)
    
    adapter = GemmaSelfieAdapter()
    
    # Load data
    cache_data = adapter.load_cached_activations()
    patterns = adapter.extract_cognitive_patterns()
    adapter.load_model_and_tokenizer()
    
    # Print available patterns
    print("   Available cognitive patterns:")
    for i, pattern in enumerate(patterns[:20]):  # Show first 20
        print(f"     {i+1}. {pattern['name']}")
    
    # Analyze patterns from a specific cache
    # target_patterns = ["depression", "anxiety", "rumination"]  # Example pattern names
    
    results = []
    
    for pattern_info in patterns[:3]:  # Analyze first 3 patterns
        cache_key = pattern_info['cache_key']
        pattern_name = pattern_info['name']
        
        print(f"   Analyzing pattern: {pattern_name}")
        
        try:
            # Prepare data for this specific pattern's cache
            prepared_data = adapter.prepare_activations_for_selfie(
                cache_key=cache_key,
                layer=17,
                max_patterns=1  # Just this pattern
            )
            
            # Use cognitive pattern template
            template = INTERPRETATION_TEMPLATES["cognitive_pattern"]
            
            pattern_results = adapter.interpret_activations_with_selfie(
                prepared_data=prepared_data,
                interpretation_template=template,
                position=-1,
                batch_size=1,
                max_new_tokens=40
            )
            
            results.extend(pattern_results)
            
        except Exception as e:
            print(f"     ❌ Pattern {pattern_name} failed: {e}")
    
    if results:
        results_file = adapter.save_interpretation_results(
            results,
            filename="specific_patterns_analysis"
        )
        print(f"✅ Pattern-specific analysis complete. Results: {results_file}")


def main():
    """Run all examples."""
    print("🚀 Gemma SelfIE Integration Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_interpretation()
        example_multi_layer_analysis()
        example_position_comparison()
        example_specific_patterns()
        
        print("\n🎉 All examples completed successfully!")
        print("\nCheck the outputs/ directory for results.")
        
    except Exception as e:
        print(f"\n💥 Examples failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()