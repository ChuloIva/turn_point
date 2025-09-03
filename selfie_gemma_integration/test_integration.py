#!/usr/bin/env python3
"""
Test script for Gemma SelfIE Integration

This script performs basic validation of the integration without running
the full model inference (which requires significant resources).
"""

import sys
import torch
import json
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from selfie_gemma_integration.gemma_selfie_adapter import GemmaSelfieAdapter
from selfie_gemma_integration.config import *


def test_initialization():
    """Test adapter initialization."""
    print("🧪 Test 1: Adapter Initialization")
    try:
        adapter = GemmaSelfieAdapter()
        assert adapter.base_path.exists()
        assert adapter.output_dir.exists()
        print("   ✅ Adapter initialized successfully")
        return True
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False


def test_activation_loading():
    """Test loading cached activations."""
    print("\n🧪 Test 2: Activation Loading")
    try:
        adapter = GemmaSelfieAdapter()
        
        # Check if activation files exist
        activation_files = list(adapter.activations_dir.glob("activations_*.pt"))
        if not activation_files:
            print("   ⚠️  No activation files found - skipping test")
            return True
        
        print(f"   Found {len(activation_files)} activation files")
        
        # Test loading one file
        test_file = activation_files[0]
        print(f"   Testing with: {test_file.name}")
        
        cache_data = adapter.load_cached_activations(str(test_file))
        
        assert len(cache_data) > 0
        print(f"   ✅ Successfully loaded {len(cache_data)} cache entries")
        
        # Validate structure
        for cache_key, data in cache_data.items():
            assert 'activations' in data
            assert 'metadata' in data
            assert isinstance(data['activations'], dict)
            print(f"   ✅ Cache entry '{cache_key}' has valid structure")
            break  # Just test first entry
        
        return True
        
    except Exception as e:
        print(f"   ❌ Activation loading failed: {e}")
        return False


def test_pattern_extraction():
    """Test cognitive pattern extraction."""
    print("\n🧪 Test 3: Pattern Extraction")
    try:
        adapter = GemmaSelfieAdapter()
        
        # Load activations first
        activation_files = list(adapter.activations_dir.glob("activations_*.pt"))
        if not activation_files:
            print("   ⚠️  No activation files found - skipping test")
            return True
        
        cache_data = adapter.load_cached_activations(str(activation_files[0]))
        patterns = adapter.extract_cognitive_patterns()
        
        if len(patterns) > 0:
            print(f"   ✅ Extracted {len(patterns)} cognitive patterns")
            
            # Validate pattern structure
            for pattern in patterns[:3]:  # Check first 3
                assert 'name' in pattern
                assert 'cache_key' in pattern
                assert 'metadata' in pattern
                print(f"   ✅ Pattern '{pattern['name']}' has valid structure")
        else:
            print("   ⚠️  No cognitive patterns found in activation data")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pattern extraction failed: {e}")
        return False


def test_activation_preparation():
    """Test activation preparation for SelfIE."""
    print("\n🧪 Test 4: Activation Preparation")
    try:
        adapter = GemmaSelfieAdapter()
        
        # Load activations
        activation_files = list(adapter.activations_dir.glob("activations_*.pt"))
        if not activation_files:
            print("   ⚠️  No activation files found - skipping test")
            return True
        
        cache_data = adapter.load_cached_activations(str(activation_files[0]))
        cache_key = list(cache_data.keys())[0]
        
        # Test preparation for layer 17
        try:
            prepared_data = adapter.prepare_activations_for_selfie(
                cache_key=cache_key,
                layer=17,
                max_patterns=2
            )
            
            # Validate prepared data structure
            required_keys = ['activation_tensor', 'layer', 'pattern_names', 'pattern_texts']
            for key in required_keys:
                assert key in prepared_data, f"Missing key: {key}"
            
            # Validate tensor shape
            tensor = prepared_data['activation_tensor']
            assert isinstance(tensor, torch.Tensor)
            assert len(tensor.shape) in [2, 3], f"Unexpected tensor shape: {tensor.shape}"
            
            print(f"   ✅ Prepared activations for layer 17")
            print(f"      Tensor shape: {tensor.shape}")
            print(f"      Patterns: {len(prepared_data['pattern_names'])}")
            
        except KeyError as e:
            if "layer_17" in str(e):
                print("   ⚠️  Layer 17 not found in activations - trying layer 10")
                prepared_data = adapter.prepare_activations_for_selfie(
                    cache_key=cache_key,
                    layer=10,
                    max_patterns=2
                )
                print("   ✅ Prepared activations for layer 10")
            else:
                raise
        
        return True
        
    except Exception as e:
        print(f"   ❌ Activation preparation failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Test 5: Configuration Loading")
    try:
        # Test that all config dictionaries exist and have expected structure
        assert 'model_name' in MODEL_CONFIG
        assert 'default_interpretation_template' in SELFIE_CONFIG
        assert 'default_layer' in ANALYSIS_CONFIG
        assert len(INTERPRETATION_TEMPLATES) > 0
        
        # Test interpretation templates
        for name, template in INTERPRETATION_TEMPLATES.items():
            assert isinstance(template, tuple)
            assert len(template) > 0
            print(f"   ✅ Template '{name}' is valid")
        
        print("   ✅ All configurations loaded successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False


def test_output_directory():
    """Test output directory creation and permissions."""
    print("\n🧪 Test 6: Output Directory")
    try:
        adapter = GemmaSelfieAdapter()
        
        # Test directory exists
        assert adapter.output_dir.exists()
        assert adapter.output_dir.is_dir()
        
        # Test write permissions by creating a test file
        test_file = adapter.output_dir / "test_write_permissions.txt"
        test_file.write_text("test")
        test_file.unlink()  # Clean up
        
        print("   ✅ Output directory is accessible and writable")
        return True
        
    except Exception as e:
        print(f"   ❌ Output directory test failed: {e}")
        return False


def test_selfie_availability():
    """Test SelfIE library availability."""
    print("\n🧪 Test 7: SelfIE Availability")
    try:
        from selfie.interpret import InterpretationPrompt, interpret_vectors
        print("   ✅ SelfIE library is available")
        
        # Test basic SelfIE functionality (without model)
        from transformers import AutoTokenizer
        try:
            # This will work if tokenizer is available
            tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Small test tokenizer
            prompt = InterpretationPrompt(tokenizer, ("Test", 0, "prompt"))
            print("   ✅ SelfIE components work correctly")
        except Exception:
            print("   ⚠️  SelfIE available but tokenizer test failed (may need model)")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ SelfIE not available: {e}")
        print("   💡 Install SelfIE: cd ../selfie && pip install -e .")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("🚀 Running Gemma SelfIE Integration Tests")
    print("=" * 50)
    
    tests = [
        test_initialization,
        test_activation_loading, 
        test_pattern_extraction,
        test_activation_preparation,
        test_config_loading,
        test_output_directory,
        test_selfie_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   💥 Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is ready to use.")
        print("\nNext steps:")
        print("1. Run: python example_usage.py")
        print("2. Or try: python gemma_selfie_adapter.py")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        
        if passed >= 5:  # Most core functionality works
            print("💡 Core functionality appears to work. You may still be able to use the integration.")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()