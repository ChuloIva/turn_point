Here's a comprehensive guide for using Ollama to generate datasets with Python:

## **Complete Ollama Setup for Dataset Generation**

### **1. Installation and Setup**

```bash
# Install Ollama (platform-specific)
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download

# Install Python library
pip install ollama

# Optional: For async operations and additional utilities
pip install aiohttp asyncio pandas json tqdm
```

### **2. Essential Imports**

```python
# Core Ollama imports
import ollama
from ollama import Client, AsyncClient
from ollama import chat, generate, pull, list as ollama_list
from ollama import ChatResponse, GenerateResponse

# Standard library imports for dataset generation
import json
import csv
import pandas as pd
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import random
from tqdm import tqdm  # Progress bars
import logging

# For advanced data handling
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
```

### **3. Pull and Setup Models**

```python
# Available models for dataset generation
AVAILABLE_MODELS = [
    'llama3.2:3b',      # Fast, good for simple tasks
    'llama3.2:8b',      # Balanced performance
    'llama3.1:8b',      # Latest with 128K context
    'mistral:7b',       # Good alternative
    'gemma2:9b',        # Google's model
    'qwen2.5:7b',       # Multilingual
    'deepseek-r1:8b',   # Reasoning model
]

# Pull models (run once)
def setup_models():
    """Download required models for dataset generation"""
    models_to_pull = ['llama3.2:8b', 'mistral:7b']  # Adjust as needed
    
    for model in models_to_pull:
        print(f"Pulling {model}...")
        try:
            ollama.pull(model)
            print(f"✅ Successfully pulled {model}")
        except Exception as e:
            print(f"❌ Failed to pull {model}: {e}")

# Check available models
def list_models():
    """List all locally available models"""
    models = ollama_list()
    print("Available models:")
    for model in models['models']:
        name = model['name']
        size = model['size'] / (1024**3)  # Convert to GB
        print(f"  - {name} ({size:.2f}GB)")
    return [model['name'] for model in models['models']]

# Run setup
# setup_models()
# list_models()
```

### **4. Dataset Generation Classes**

```python
@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    model: str = "llama3.2:8b"
    output_file: str = "generated_dataset.jsonl"
    batch_size: int = 10
    max_concurrent: int = 5
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = "You are a helpful assistant."
    
class DatasetGenerator:
    """Main class for generating datasets using Ollama"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.client = Client()
        self.async_client = AsyncClient()
        self.generated_data = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def generate_single(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a single response"""
        try:
            response = ollama.chat(
                model=self.config.model,
                messages=[
                    {'role': 'system', 'content': self.config.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': kwargs.get('temperature', self.config.temperature),
                    'num_predict': kwargs.get('max_tokens', self.config.max_tokens),
                }
            )
            
            return {
                'prompt': prompt,
                'response': response['message']['content'],
                'model': self.config.model,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'temperature': kwargs.get('temperature', self.config.temperature),
                    'tokens': len(response['message']['content'].split()),
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                'prompt': prompt,
                'response': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_single_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a single response asynchronously"""
        try:
            response = await self.async_client.chat(
                model=self.config.model,
                messages=[
                    {'role': 'system', 'content': self.config.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': kwargs.get('temperature', self.config.temperature),
                    'num_predict': kwargs.get('max_tokens', self.config.max_tokens),
                }
            )
            
            return {
                'prompt': prompt,
                'response': response['message']['content'],
                'model': self.config.model,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'temperature': kwargs.get('temperature', self.config.temperature),
                    'tokens': len(response['message']['content'].split()),
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                'prompt': prompt,
                'response': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.generate_single, prompt, **kwargs): prompt 
                for prompt in prompts
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                result = future.result()
                results.append(result)
                
        return results
    
    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts asynchronously"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def bounded_generate(prompt):
            async with semaphore:
                return await self.generate_single_async(prompt, **kwargs)
        
        tasks = [bounded_generate(prompt) for prompt in prompts]
        results = []
        
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
            result = await coro
            results.append(result)
            
        return results
    
    def save_data(self, data: List[Dict[str, Any]], format: str = 'jsonl'):
        """Save generated data to file"""
        output_path = Path(self.config.output_file)
        
        if format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
        elif format == 'json':
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        elif format == 'csv':
            # Flatten the data for CSV
            flattened = []
            for item in data:
                flat_item = {
                    'prompt': item['prompt'],
                    'response': item.get('response', ''),
                    'model': item.get('model', ''),
                    'timestamp': item.get('timestamp', ''),
                    'temperature': item.get('metadata', {}).get('temperature', ''),
                    'tokens': item.get('metadata', {}).get('tokens', ''),
                    'error': item.get('error', '')
                }
                flattened.append(flat_item)
            
            df = pd.DataFrame(flattened)
            df.to_csv(output_path.with_suffix('.csv'), index=False)
            
        self.logger.info(f"Saved {len(data)} items to {output_path}")
```

### **5. Specialized Dataset Generation Functions**

```python
class SpecializedGenerators:
    """Specialized generators for different types of datasets"""
    
    @staticmethod
    def generate_qa_dataset(topics: List[str], questions_per_topic: int = 10) -> List[str]:
        """Generate Q&A prompts"""
        prompts = []
        question_types = [
            "What is",
            "How does",
            "Why is",
            "When should",
            "Where can",
            "Who would",
            "Explain",
            "Describe",
            "Compare",
            "Analyze"
        ]
        
        for topic in topics:
            for _ in range(questions_per_topic):
                q_type = random.choice(question_types)
                prompt = f"{q_type} {topic}? Provide a detailed and accurate answer."
                prompts.append(prompt)
                
        return prompts
    
    @staticmethod
    def generate_instruction_dataset(tasks: List[str], variations: int = 5) -> List[str]:
        """Generate instruction-following prompts"""
        prompts = []
        instruction_formats = [
            "Please {}",
            "I need you to {}",
            "Can you {}",
            "Your task is to {}",
            "Help me {}"
        ]
        
        for task in tasks:
            for _ in range(variations):
                format_str = random.choice(instruction_formats)
                prompt = format_str.format(task)
                prompts.append(prompt)
                
        return prompts
    
    @staticmethod
    def generate_creative_writing_prompts(themes: List[str], count: int = 50) -> List[str]:
        """Generate creative writing prompts"""
        prompts = []
        writing_types = [
            "Write a short story about",
            "Create a poem inspired by",
            "Develop a character who",
            "Describe a world where",
            "Write a dialogue between two people discussing"
        ]
        
        for _ in range(count):
            theme = random.choice(themes)
            writing_type = random.choice(writing_types)
            prompt = f"{writing_type} {theme}. Make it engaging and original."
            prompts.append(prompt)
            
        return prompts
    
    @staticmethod
    def generate_code_dataset(languages: List[str], tasks: List[str]) -> List[str]:
        """Generate coding prompts"""
        prompts = []
        
        for language in languages:
            for task in tasks:
                prompt = f"Write a {language} function to {task}. Include comments and error handling."
                prompts.append(prompt)
                
        return prompts
```

### **6. Complete Usage Examples**

```python
# Example 1: Basic Q&A Dataset Generation
async def generate_qa_dataset():
    """Generate a Q&A dataset"""
    
    # Configuration
    config = DatasetConfig(
        model="llama3.2:8b",
        output_file="qa_dataset.jsonl",
        batch_size=20,
        max_concurrent=3,
        temperature=0.7,
        system_prompt="You are an expert teacher. Provide clear, accurate, and educational answers."
    )
    
    # Initialize generator
    generator = DatasetGenerator(config)
    
    # Generate prompts
    topics = [
        "machine learning", "climate change", "renewable energy", 
        "space exploration", "artificial intelligence", "quantum computing",
        "biotechnology", "cybersecurity", "blockchain", "robotics"
    ]
    
    prompts = SpecializedGenerators.generate_qa_dataset(topics, questions_per_topic=15)
    print(f"Generated {len(prompts)} prompts")
    
    # Generate responses
    results = await generator.generate_batch_async(prompts)
    
    # Filter successful generations
    successful = [r for r in results if r.get('response') is not None]
    print(f"Successfully generated {len(successful)} responses")
    
    # Save data
    generator.save_data(successful, format='jsonl')
    generator.save_data(successful, format='csv')  # Also save as CSV
    
    return successful

# Example 2: Multi-Model Comparison Dataset
async def generate_comparison_dataset():
    """Generate responses from multiple models for comparison"""
    
    models = ['llama3.2:8b', 'mistral:7b', 'gemma2:9b']
    prompts = [
        "Explain the concept of artificial intelligence",
        "What are the benefits and risks of renewable energy?",
        "How does blockchain technology work?",
        "Describe the process of photosynthesis",
        "What is the future of space exploration?"
    ]
    
    all_results = []
    
    for model in models:
        config = DatasetConfig(
            model=model,
            temperature=0.7,
            max_tokens=300
        )
        
        generator = DatasetGenerator(config)
        
        print(f"Generating with {model}...")
        results = await generator.generate_batch_async(prompts)
        all_results.extend(results)
    
    # Save combined results
    with open('multi_model_comparison.jsonl', 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    return all_results

# Example 3: Streaming Generation for Large Datasets
def generate_large_dataset_streaming():
    """Generate large datasets with streaming to avoid memory issues"""
    
    config = DatasetConfig(
        model="llama3.2:8b",
        output_file="large_dataset.jsonl",
        batch_size=50
    )
    
    generator = DatasetGenerator(config)
    
    # Generate prompts in batches
    topics = ["technology", "science", "history", "literature", "philosophy"]
    
    with open(config.output_file, 'w') as f:
        for batch_num in range(10):  # 10 batches
            print(f"Processing batch {batch_num + 1}/10")
            
            batch_prompts = SpecializedGenerators.generate_qa_dataset(
                topics, questions_per_topic=10
            )
            
            results = generator.generate_batch(batch_prompts)
            
            # Stream write to file
            for result in results:
                if result.get('response'):
                    f.write(json.dumps(result) + '\n')
            
            # Optional: Add delay between batches
            time.sleep(2)
```

### **7. Advanced Features and Utilities**

```python
# Utility functions
def analyze_dataset(file_path: str):
    """Analyze generated dataset statistics"""
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Basic statistics
    total_items = len(data)
    successful = len([d for d in data if d.get('response')])
    failed = total_items - successful
    
    response_lengths = [len(d['response'].split()) for d in data if d.get('response')]
    avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    
    print(f"Dataset Analysis:")
    print(f"  Total items: {total_items}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Average response length: {avg_length:.1f} words")
    print(f"  Min length: {min(response_lengths) if response_lengths else 0}")
    print(f"  Max length: {max(response_lengths) if response_lengths else 0}")

def filter_quality_responses(data: List[Dict], min_length: int = 50):
    """Filter responses by quality metrics"""
    filtered = []
    
    for item in data:
        response = item.get('response', '')
        if (response and 
            len(response.split()) >= min_length and 
            not response.lower().startswith('i cannot') and
            not response.lower().startswith('sorry')):
            filtered.append(item)
    
    return filtered

# Main execution
if __name__ == "__main__":
    # Run the dataset generation
    # results = asyncio.run(generate_qa_dataset())
    
    # Or for comparison dataset
    # results = asyncio.run(generate_comparison_dataset())
    
    # Or for large streaming dataset
    # generate_large_dataset_streaming()
    
    # Analyze results
    # analyze_dataset("qa_dataset.jsonl")
    
    pass
```

### **8. Running the Complete Pipeline**

```python
# Complete pipeline example
async def main():
    """Main function to run complete dataset generation pipeline"""
    
    print("🚀 Starting Ollama Dataset Generation Pipeline")
    
    # Step 1: Setup
    print("📋 Setting up models...")
    # setup_models()  # Uncomment if models not already downloaded
    available_models = list_models()
    
    # Step 2: Generate Q&A dataset
    print("❓ Generating Q&A dataset...")
    qa_results = await generate_qa_dataset()
    
    # Step 3: Generate instruction dataset  
    print("📝 Generating instruction dataset...")
    instruction_prompts = SpecializedGenerators.generate_instruction_dataset([
        "summarize this text",
        "translate this to Spanish", 
        "write a product description",
        "create a social media post",
        "explain a complex topic simply"
    ], variations=10)
    
    config = DatasetConfig(
        model="llama3.2:8b",
        output_file="instruction_dataset.jsonl",
        system_prompt="You are a helpful assistant that follows instructions precisely."
    )
    
    generator = DatasetGenerator(config)
    instruction_results = await generator.generate_batch_async(instruction_prompts)
    generator.save_data(instruction_results)
    
    # Step 4: Analysis
    print("📊 Analyzing results...")
    analyze_dataset("qa_dataset.jsonl")
    analyze_dataset("instruction_dataset.jsonl")
    
    print("✅ Dataset generation complete!")

# Run the pipeline
if __name__ == "__main__":
    asyncio.run(main())
```

This comprehensive setup gives you everything you need to generate datasets using Ollama with Python, including async operations, error handling, multiple output formats, and specialized generators for different types of content.