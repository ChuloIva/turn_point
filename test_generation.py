import asyncio
import ollama
from cognitive_pattern_dataset_generator import CognitivePatternDatasetGenerator, DatasetConfig

async def test_ollama_connection():
    """Test if Ollama is running and model is available"""
    try:
        models = ollama.list()
        print("Available models:")
        for model in models['models']:
            print(f"  - {model['name']}")
        
        # Test a simple generation
        response = ollama.generate(model='llama3.2', prompt='Say hello')
        print(f"Test generation successful: {response['response'][:50]}...")
        return True
    except Exception as e:
        print(f"Ollama connection error: {e}")
        print("Make sure Ollama is running and llama3.2 model is pulled")
        print("Run: ollama pull llama3.2")
        return False

async def test_single_generation():
    """Test generating a single thought pattern"""
    config = DatasetConfig(
        model="llama3.2",  # Use base model for testing
        temperature=0.8,
        max_tokens=150
    )
    
    generator = CognitivePatternDatasetGenerator(config)
    
    # Test with first pattern
    pattern_row = generator.cognitive_patterns.iloc[0]
    pattern_name = pattern_row['Concept Name']
    questions = list(generator.questions_by_pattern.values())[0]
    
    print(f"Testing generation for: {pattern_name}")
    print(f"Using question: {questions[0][:60]}...")
    
    result = await generator.generate_single_thought_pattern(
        pattern_name,
        pattern_row['Description'],
        questions[0],
        pattern_row['Cognitive Pattern']
    )
    
    if result.get('thought_pattern'):
        print(f"✅ Success! Generated thought pattern:")
        print(f"   {result['thought_pattern']}")
        return True
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        return False

async def main():
    print("🧠 Testing Cognitive Pattern Dataset Generation Setup")
    
    # Test 1: Ollama connection
    print("\n1. Testing Ollama connection...")
    if not await test_ollama_connection():
        return
    
    # Test 2: Single generation
    print("\n2. Testing single thought pattern generation...")
    if await test_single_generation():
        print("\n✅ All tests passed! Ready to generate full dataset.")
        print("Run: python cognitive_pattern_dataset_generator.py")
    else:
        print("\n❌ Generation test failed. Check configuration.")

if __name__ == "__main__":
    asyncio.run(main())