# Core Ollama imports
import ollama
from ollama import Client, AsyncClient
from ollama import chat, generate, pull, list as ollama_list
from ollama import ChatResponse, GenerateResponse

# Standard library imports
import json
import pandas as pd
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Union
from pathlib import Path
import random
from tqdm import tqdm
import logging
import re
from dataclasses import dataclass

@dataclass
class PositivePatternConfig:
    """Configuration for positive/healthy pattern generation"""
    model: str = "mannix/llama3.1-8b-abliterated:latest"
    output_file: str = "positive_patterns.jsonl"
    max_concurrent: int = 3
    temperature: float = 0.6
    max_tokens: int = 300
    system_prompt: str = """You are a cognitive therapist generating healthy, positive thought patterns. Given a cognitive pattern name and description, create a realistic first-person thought pattern that represents a healthy, balanced version of that cognitive style. The response should be natural, authentic, and demonstrate psychological wellness while maintaining the core cognitive pattern type but in a healthy expression."""

class PositivePatternGenerator:
    """Generate healthy/positive versions of cognitive patterns"""
    
    def __init__(self, config: PositivePatternConfig):
        self.config = config
        self.client = Client()
        self.async_client = AsyncClient()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def load_bad_good_narratives(self, filepath: str) -> List[Dict[str, Any]]:
        """Load existing bad > good narratives dataset"""
        narratives = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    narratives.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(narratives)} bad > good narratives from {filepath}")
        return narratives
    
    def extract_unique_patterns(self, narratives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique cognitive patterns from narratives"""
        pattern_map = {}
        
        for narrative in narratives:
            pattern_key = (
                narrative.get('cognitive_pattern_name', ''),
                narrative.get('cognitive_pattern_type', '')
            )
            
            if pattern_key not in pattern_map:
                pattern_map[pattern_key] = {
                    'cognitive_pattern_name': narrative.get('cognitive_pattern_name', ''),
                    'cognitive_pattern_type': narrative.get('cognitive_pattern_type', ''),
                    'pattern_description': narrative.get('pattern_description', ''),
                    'source_question': narrative.get('source_question', ''),
                    'example_negative': narrative.get('original_thought_pattern', ''),
                    'example_transformed': narrative.get('transformed_thought_pattern', '')
                }
        
        unique_patterns = list(pattern_map.values())
        self.logger.info(f"Found {len(unique_patterns)} unique cognitive patterns")
        return unique_patterns
    
    def create_positive_pattern_prompt(self, pattern_data: Dict[str, Any]) -> str:
        """Create prompt for generating healthy version of cognitive pattern"""
        
        return f"""Cognitive Pattern: {pattern_data['cognitive_pattern_name']}
Pattern Type: {pattern_data['cognitive_pattern_type']}
Description: {pattern_data['pattern_description']}

Source Question Context: {pattern_data['source_question']}

Example of unhealthy version:
"{pattern_data['example_negative']}"

Example of therapeutic transformation:
"{pattern_data['example_transformed']}"

Now generate a healthy, positive first-person thought pattern that represents this cognitive style in a psychologically well-adjusted person. The thought should:
- Demonstrate the same cognitive pattern type but in a healthy, balanced way
- Show emotional intelligence and self-awareness
- Be realistic and authentic (not overly optimistic)
- Reflect good mental health practices
- Be around 2-4 sentences in first person

Healthy version:"""
    
    async def generate_positive_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a healthy/positive version of a cognitive pattern"""
        try:
            prompt = self.create_positive_pattern_prompt(pattern_data)
            
            response = await self.async_client.chat(
                model=self.config.model,
                messages=[
                    {'role': 'system', 'content': self.config.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_tokens,
                }
            )
            
            positive_pattern = response['message']['content'].strip()
            
            # Clean up formatting
            positive_pattern = re.sub(r'^["\'`](.*)["\'`]$', r'\1', positive_pattern.strip())
            positive_pattern = re.sub(r'^(Healthy version:?\s*)', '', positive_pattern, flags=re.IGNORECASE)
            positive_pattern = re.sub(r'^(Positive thought:?\s*)', '', positive_pattern, flags=re.IGNORECASE)
            
            return {
                'positive_thought_pattern': positive_pattern,
                'cognitive_pattern_name': pattern_data['cognitive_pattern_name'],
                'cognitive_pattern_type': pattern_data['cognitive_pattern_type'],
                'pattern_description': pattern_data['pattern_description'],
                'source_question': pattern_data['source_question'],
                'reference_negative_example': pattern_data['example_negative'],
                'reference_transformed_example': pattern_data['example_transformed'],
                'model': self.config.model,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'temperature': self.config.temperature,
                    'word_count': len(positive_pattern.split())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating positive pattern: {e}")
            return {
                'positive_thought_pattern': None,
                'cognitive_pattern_name': pattern_data.get('cognitive_pattern_name', ''),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_complete_positive_dataset(self, bad_good_narratives_path: str) -> List[Dict[str, Any]]:
        """Generate complete positive patterns dataset"""
        
        # Load all narratives (not just unique patterns)
        narratives = self.load_bad_good_narratives(bad_good_narratives_path)
        
        if not narratives:
            self.logger.error("No narratives found")
            return []
        
        # Convert narratives to pattern format for processing
        all_patterns = []
        for narrative in narratives:
            pattern_data = {
                'cognitive_pattern_name': narrative.get('cognitive_pattern_name', ''),
                'cognitive_pattern_type': narrative.get('cognitive_pattern_type', ''),
                'pattern_description': narrative.get('pattern_description', ''),
                'source_question': narrative.get('source_question', ''),
                'example_negative': narrative.get('original_thought_pattern', ''),
                'example_transformed': narrative.get('transformed_thought_pattern', '')
            }
            all_patterns.append(pattern_data)
        
        positive_patterns = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        self.logger.info(f"Generating positive patterns for {len(all_patterns)} narrative examples")
        
        async def bounded_generate(pattern_data):
            async with semaphore:
                return await self.generate_positive_pattern(pattern_data)
        
        # Create tasks
        tasks = [bounded_generate(pattern) for pattern in all_patterns]
        
        # Execute with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), 
                        total=len(tasks), 
                        desc="Generating positive patterns"):
            result = await coro
            if result and result.get('positive_thought_pattern'):
                results.append(result)
                positive_patterns.append(result)
            
            # Save incrementally every 10 results
            if len(results) % 10 == 0:
                self.save_incremental_patterns(results[-10:])
        
        return positive_patterns
    
    def save_incremental_patterns(self, patterns: List[Dict[str, Any]]):
        """Save patterns incrementally"""
        # Create positive_full directory if it doesn't exist
        positive_full_dir = Path("data/positive_full")
        positive_full_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = positive_full_dir / f"incremental_positive_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for pattern in patterns:
                f.write(json.dumps(pattern, ensure_ascii=False) + '\n')
    
    def save_final_dataset(self, patterns: List[Dict[str, Any]]):
        """Save final positive patterns dataset"""
        
        # Save as JSONL
        output_path = Path(self.config.output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for pattern in patterns:
                f.write(json.dumps(pattern, ensure_ascii=False) + '\n')
        
        # Save as CSV for easy viewing
        csv_data = []
        for pattern in patterns:
            csv_data.append({
                'positive_thought': pattern['positive_thought_pattern'],
                'cognitive_pattern': pattern['cognitive_pattern_name'],
                'pattern_type': pattern['cognitive_pattern_type'],
                'pattern_description': pattern['pattern_description'],
                'source_question': pattern['source_question'],
                'word_count': pattern.get('metadata', {}).get('word_count', 0),
                'timestamp': pattern['timestamp']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        self.logger.info(f"Saved {len(patterns)} positive patterns to {output_path}")
        self.logger.info(f"CSV version saved to {output_path.with_suffix('.csv')}")
        
        return patterns
    
    def analyze_patterns(self, patterns: List[Dict[str, Any]]):
        """Analyze generated positive patterns dataset"""
        
        pattern_counts = {}
        type_counts = {}
        word_counts = []
        
        for pattern in patterns:
            name = pattern['cognitive_pattern_name']
            type_name = pattern['cognitive_pattern_type']
            
            pattern_counts[name] = pattern_counts.get(name, 0) + 1
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            word_count = pattern.get('metadata', {}).get('word_count', 0)
            if word_count:
                word_counts.append(word_count)
        
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        
        print(f"\n🌟 Positive Pattern Analysis:")
        print(f"  Total positive patterns generated: {len(patterns)}")
        print(f"  Average word count: {avg_words:.1f}")
        print(f"  Unique cognitive patterns: {len(pattern_counts)}")
        print(f"  Pattern types covered: {len(type_counts)}")
        
        print(f"\n📈 Top pattern types:")
        for pattern_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pattern_type}: {count}")

async def main():
    """Main function to generate positive patterns"""
    print("🌟 Starting Positive/Healthy Pattern Generation")
    
    # Configuration
    config = PositivePatternConfig(
        model="mannix/llama3.1-8b-abliterated:latest",
        output_file="data/positive_patterns.jsonl",
        max_concurrent=3,
        temperature=0.6,
        max_tokens=300
    )
    
    # Check if bad_good_narratives exists
    narratives_file = "data/bad_good_narratives.jsonl"
    if not Path(narratives_file).exists():
        print(f"❌ Bad > good narratives dataset not found: {narratives_file}")
        print("Run narrative_generator.py first to generate the source data.")
        return
    
    # Initialize generator
    generator = PositivePatternGenerator(config)
    
    # Generate positive patterns
    print(f"📋 Using narratives dataset: {narratives_file}")
    positive_patterns = await generator.generate_complete_positive_dataset(narratives_file)
    
    if positive_patterns:
        # Save and analyze
        generator.save_final_dataset(positive_patterns)
        generator.analyze_patterns(positive_patterns)
        
        print("✅ Positive pattern generation complete!")
        
        # Show sample
        if positive_patterns:
            print(f"\n📝 Sample positive pattern:")
            sample = positive_patterns[0]
            print(f"Pattern: {sample['cognitive_pattern_name']}")
            print(f"Positive thought: {sample['positive_thought_pattern']}")
    else:
        print("❌ No positive patterns generated")
    
    return positive_patterns

if __name__ == "__main__":
    asyncio.run(main())