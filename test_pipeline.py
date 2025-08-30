#!/usr/bin/env python3
"""
Test script for the complete bad > good narrative generation pipeline
"""

import asyncio
import sys
from pathlib import Path
from narrative_generator import BadGoodNarrativeGenerator, NarrativeConfig, list_models
from cognitive_pattern_dataset_generator import CognitivePatternDatasetGenerator, DatasetConfig

async def test_complete_pipeline():
    """Test the complete pipeline from cognitive patterns to bad > good narratives"""
    
    print("🧠 Testing Complete Bad > Good Narrative Generation Pipeline")
    print("=" * 60)
    
    # Step 1: Check available models
    print("📋 Step 1: Checking available models...")
    available_models = list_models()
    
    if not available_models:
        print("❌ No Ollama models found. Please install Ollama and pull a model first.")
        print("   Run: ollama pull llama3.2:8b")
        return False
    
    # Find the best available model
    preferred_models = ['llama3.2:8b', 'llama3.2:3b', 'mistral:7b', 'gemma2:9b']
    selected_model = None
    
    for model in preferred_models:
        if model in available_models:
            selected_model = model
            break
    
    if not selected_model:
        selected_model = available_models[0]  # Use first available
    
    print(f"✅ Selected model: {selected_model}")
    
    # Step 2: Generate cognitive patterns dataset (if not exists)
    print("\n📋 Step 2: Generating cognitive patterns dataset...")
    
    cognitive_dataset_file = "cognitive_patterns_dataset.jsonl"
    
    if not Path(cognitive_dataset_file).exists():
        print("   Cognitive dataset not found. Generating...")
        
        # Check if required files exist
        if not Path('data/cognitive_patterns_short14.csv').exists():
            print("❌ Required file 'data/cognitive_patterns_short14.csv' not found")
            return False
            
        if not Path('data/cognitive_pattern_questions.md').exists():
            print("❌ Required file 'data/cognitive_pattern_questions.md' not found")
            return False
        
        # Generate cognitive patterns
        config = DatasetConfig(
            model=selected_model,
            output_file=cognitive_dataset_file,
            batch_size=3,  # Small batch for testing
            max_concurrent=2,
            temperature=0.8,
            max_tokens=200
        )
        
        try:
            generator = CognitivePatternDatasetGenerator(config)
            
            # Generate just a few patterns for testing
            print("   Generating small test dataset (first 3 patterns)...")
            
            # Limit to first 3 patterns for testing
            original_patterns = generator.cognitive_patterns.copy()
            generator.cognitive_patterns = generator.cognitive_patterns.head(3)
            
            results = await generator.generate_complete_dataset()
            final_results = generator.save_final_dataset(results)
            
            if not final_results:
                print("❌ Failed to generate cognitive patterns dataset")
                return False
                
            print(f"✅ Generated {len(final_results)} cognitive patterns")
            
        except Exception as e:
            print(f"❌ Error generating cognitive patterns: {e}")
            return False
    else:
        print(f"✅ Using existing cognitive dataset: {cognitive_dataset_file}")
    
    # Step 3: Generate bad > good narratives
    print("\n📋 Step 3: Generating bad > good narratives...")
    
    try:
        # Configuration for narrative generation
        narrative_config = NarrativeConfig(
            model=selected_model,
            output_file="test_bad_good_narratives.jsonl",
            max_concurrent=2,
            temperature=0.7,
            max_tokens=300
        )
        
        # Initialize narrative generator
        narrative_generator = BadGoodNarrativeGenerator(narrative_config)
        
        print(f"   Loaded {len(narrative_generator.exercises)} therapeutic exercises")
        
        # Generate narratives (just 1 exercise per pattern for testing)
        narratives = await narrative_generator.generate_complete_narrative_dataset(
            cognitive_dataset_file,
            narratives_per_pattern=1  # Just 1 exercise per pattern for testing
        )
        
        if not narratives:
            print("❌ Failed to generate narratives")
            return False
        
        # Save and analyze
        narrative_generator.save_final_dataset(narratives)
        narrative_generator.analyze_narratives(narratives)
        
        print(f"✅ Generated {len(narratives)} bad > good narratives")
        
        # Step 4: Show sample results
        print("\n📋 Step 4: Sample Results")
        print("-" * 40)
        
        if narratives:
            sample = narratives[0]
            print(f"Cognitive Pattern: {sample['cognitive_pattern_name']}")
            print(f"Exercise: {sample['exercise_category']} - {sample['exercise_title']}")
            print(f"\nOriginal (Bad): {sample['original_thought_pattern'][:150]}...")
            print(f"\nTransformed (Good): {sample['transformed_thought_pattern'][:150]}...")
            
        print("\n✅ Pipeline test completed successfully!")
        print(f"📁 Files generated:")
        print(f"   - {cognitive_dataset_file}")
        print(f"   - {narrative_config.output_file}")
        print(f"   - {Path(narrative_config.output_file).with_suffix('.csv')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in narrative generation: {e}")
        return False

def test_individual_components():
    """Test individual components separately"""
    
    print("\n🔧 Testing Individual Components")
    print("=" * 40)
    
    # Test 1: Exercise loading
    try:
        from narrative_generator import BadGoodNarrativeGenerator
        config = NarrativeConfig()
        generator = BadGoodNarrativeGenerator(config)
        
        print(f"✅ Loaded {len(generator.exercises)} exercises")
        
        # Show sample exercise
        if generator.exercises:
            sample_ex = generator.exercises[0]
            print(f"   Sample: {sample_ex['category']} - {sample_ex['title']}")
            
    except Exception as e:
        print(f"❌ Exercise loading failed: {e}")
        return False
    
    # Test 2: Prompt loading
    try:
        prompt = generator.load_prompt()
        print(f"✅ Loaded transformation prompt ({len(prompt)} chars)")
        
    except Exception as e:
        print(f"❌ Prompt loading failed: {e}")
        return False
    
    print("✅ All components test passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting Pipeline Test")
    
    # Test components first
    if not test_individual_components():
        print("❌ Component tests failed")
        sys.exit(1)
    
    # Test complete pipeline
    try:
        success = asyncio.run(test_complete_pipeline())
        if success:
            print("\n🎉 All tests passed! Pipeline is ready to use.")
            sys.exit(0)
        else:
            print("\n❌ Pipeline test failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)