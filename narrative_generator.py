# Core Ollama imports
import ollama
from ollama import Client, AsyncClient
from ollama import chat, generate, pull, list as ollama_list
from ollama import ChatResponse, GenerateResponse

# Standard library imports for dataset generation
import json
import pandas as pd
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Union
from pathlib import Path
import random
from tqdm import tqdm  # Progress bars
import logging
import re
from dataclasses import dataclass

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

def list_models():
    """List all locally available models"""
    try:
        models = ollama_list()
        print("Available models:")
        for model in models['models']:
            name = model['name']
            size = model['size'] / (1024**3)  # Convert to GB
            print(f"  - {name} ({size:.2f}GB)")
        return [model['name'] for model in models['models']]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

@dataclass
class NarrativeConfig:
    """Configuration for bad > good narrative generation"""
    model: str = "mannix/llama3.1-8b-abliterated:latest"
    output_file: str = "bad_good_narratives.jsonl"
    max_concurrent: int = 3
    temperature: float = 0.7
    max_tokens: int = 400
    system_prompt: str = """You are generating therapeutic narrative transformations. Given a negative thought pattern and a therapeutic exercise, show how the thought pattern changes when the exercise is applied. Write only the transformed first-person thoughts - no additional explanation."""

class BadGoodNarrativeGenerator:
    """Generate bad > good narratives by applying therapeutic exercises to cognitive patterns"""
    
    def __init__(self, config: NarrativeConfig):
        self.config = config
        self.client = Client()
        self.async_client = AsyncClient()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Load components
        self.exercises = self.load_exercises()
        self.transformation_prompt = self.load_prompt()
        
    def load_exercises(self) -> List[Dict[str, str]]:
        """Parse exercises from markdown file"""
        exercises = []
        
        with open('data/excercises.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by exercise sections
        sections = re.split(r'\n## \d+\.\s*\*\*([^*]+)\*\*', content)[1:]
        
        for i in range(0, len(sections), 2):
            if i + 1 < len(sections):
                exercise_type = sections[i].strip()
                exercise_content = sections[i + 1]
                
                # Extract specific exercises (Exercise A, B, etc.)
                sub_exercises = re.split(r'\n### Exercise [A-Z]:', exercise_content)
                
                for sub_ex in sub_exercises[1:]:  # Skip first empty split
                    # Get the exercise title and content
                    lines = sub_ex.strip().split('\n')
                    if lines:
                        title = lines[0].strip()
                        content_text = '\n'.join(lines[1:])
                        
                        # Extract "RIGHT NOW" emergency interventions if present
                        emergency_match = re.search(r'\*\*RIGHT NOW[^*]*\*\*\*(.*?)\*', content_text, re.DOTALL)
                        emergency_intervention = emergency_match.group(1).strip() if emergency_match else ""
                        
                        exercises.append({
                            'category': exercise_type,
                            'title': title,
                            'content': content_text,
                            'emergency_intervention': emergency_intervention
                        })
        
        self.logger.info(f"Loaded {len(exercises)} therapeutic exercises")
        return exercises
    
    def load_prompt(self) -> str:
        """Load the transformation prompt"""
        with open('data/prompts/prompt_thought_pattern.md', 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def load_cognitive_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load cognitive patterns dataset (JSONL format)"""
        patterns = []
        
        if Path(dataset_path).suffix == '.jsonl':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        patterns.append(json.loads(line))
        elif Path(dataset_path).suffix == '.csv':
            df = pd.read_csv(dataset_path)
            patterns = df.to_dict('records')
        
        self.logger.info(f"Loaded {len(patterns)} cognitive patterns from {dataset_path}")
        return patterns
    
    def create_transformation_prompt(self, thought_pattern: str, exercise: Dict[str, str]) -> str:
        """Create prompt for transforming bad thought to good thought using exercise"""
        
        exercise_instruction = f"""
Exercise Category: {exercise['category']}
Exercise: {exercise['title']}

Exercise Details:
{exercise['content']}

Emergency Intervention (if applicable):
{exercise['emergency_intervention']}
"""
        
        return f"""{self.transformation_prompt}

Original negative thought pattern:
"{thought_pattern}"

Therapeutic exercise to apply:
{exercise_instruction}

Show how the thought pattern transforms when this exercise is applied. Write only the new first-person thoughts - no explanations or descriptions."""
    
    async def generate_bad_good_narrative(self, pattern_data: Dict[str, Any], exercise: Dict[str, str]) -> Dict[str, Any]:
        """Generate a bad > good narrative transformation"""
        try:
            original_thought = pattern_data.get('thought_pattern', '')
            if not original_thought:
                return None
                
            prompt = self.create_transformation_prompt(original_thought, exercise)
            
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
            
            transformed_thought = response['message']['content'].strip()
            
            # Clean up formatting
            transformed_thought = re.sub(r'^["\'\`](.*)["\'`]$', r'\1', transformed_thought.strip())
            transformed_thought = re.sub(r'^(Transformed thoughts?:?\s*)', '', transformed_thought, flags=re.IGNORECASE)
            
            return {
                'original_thought_pattern': original_thought,
                'transformed_thought_pattern': transformed_thought,
                'cognitive_pattern_name': pattern_data.get('cognitive_pattern_name', ''),
                'cognitive_pattern_type': pattern_data.get('cognitive_pattern_type', ''),
                'pattern_description': pattern_data.get('pattern_description', ''),
                'source_question': pattern_data.get('source_question', ''),
                'exercise_category': exercise['category'],
                'exercise_title': exercise['title'],
                'exercise_content': exercise['content'],
                'emergency_intervention': exercise['emergency_intervention'],
                'model': self.config.model,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'temperature': self.config.temperature,
                    'original_word_count': len(original_thought.split()),
                    'transformed_word_count': len(transformed_thought.split())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating narrative transformation: {e}")
            return {
                'original_thought_pattern': pattern_data.get('thought_pattern', ''),
                'transformed_thought_pattern': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_complete_narrative_dataset(self, cognitive_dataset_path: str, narratives_per_pattern: int = 2) -> List[Dict[str, Any]]:
        """Generate complete bad > good narrative dataset"""
        
        # Load cognitive patterns
        cognitive_patterns = self.load_cognitive_dataset(cognitive_dataset_path)
        
        if not cognitive_patterns:
            self.logger.error(f"No cognitive patterns found in {cognitive_dataset_path}")
            return []
        
        all_narratives = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        self.logger.info(f"Generating narratives for {len(cognitive_patterns)} patterns with {len(self.exercises)} exercises")
        
        async def bounded_generate(pattern_data, exercise):
            async with semaphore:
                return await self.generate_bad_good_narrative(pattern_data, exercise)
        
        # Create tasks for all pattern-exercise combinations
        tasks = []
        for pattern_data in cognitive_patterns:
            # Select random exercises for each pattern
            selected_exercises = random.sample(self.exercises, min(narratives_per_pattern, len(self.exercises)))
            
            for exercise in selected_exercises:
                tasks.append(bounded_generate(pattern_data, exercise))
        
        # Execute all tasks with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), 
                        total=len(tasks), 
                        desc="Generating narratives"):
            result = await coro
            if result and result.get('transformed_thought_pattern'):
                results.append(result)
                all_narratives.append(result)
            
            # Save incrementally every 10 results
            if len(results) % 10 == 0:
                self.save_incremental_narratives(results[-10:])
        
        return all_narratives
    
    def save_incremental_narratives(self, narratives: List[Dict[str, Any]]):
        """Save narratives incrementally"""
        filepath = Path(f"incremental_narratives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for narrative in narratives:
                f.write(json.dumps(narrative, ensure_ascii=False) + '\n')
    
    def save_final_dataset(self, narratives: List[Dict[str, Any]]):
        """Save final narrative dataset"""
        
        # Save as JSONL
        output_path = Path(self.config.output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for narrative in narratives:
                f.write(json.dumps(narrative, ensure_ascii=False) + '\n')
        
        # Save as CSV for easy viewing
        csv_data = []
        for narrative in narratives:
            csv_data.append({
                'original_thought': narrative['original_thought_pattern'],
                'transformed_thought': narrative['transformed_thought_pattern'],
                'cognitive_pattern': narrative['cognitive_pattern_name'],
                'pattern_type': narrative['cognitive_pattern_type'],
                'exercise_category': narrative['exercise_category'],
                'exercise_title': narrative['exercise_title'],
                'source_question': narrative['source_question'],
                'original_word_count': narrative.get('metadata', {}).get('original_word_count', 0),
                'transformed_word_count': narrative.get('metadata', {}).get('transformed_word_count', 0),
                'timestamp': narrative['timestamp']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        self.logger.info(f"Saved {len(narratives)} bad > good narratives to {output_path}")
        self.logger.info(f"CSV version saved to {output_path.with_suffix('.csv')}")
        
        return narratives
    
    def analyze_narratives(self, narratives: List[Dict[str, Any]]):
        """Analyze generated narrative dataset"""
        
        # Group by categories
        pattern_counts = {}
        exercise_counts = {}
        word_changes = []
        
        for narrative in narratives:
            pattern = narrative['cognitive_pattern_name']
            exercise = narrative['exercise_category']
            
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            exercise_counts[exercise] = exercise_counts.get(exercise, 0) + 1
            
            orig_words = narrative.get('metadata', {}).get('original_word_count', 0)
            trans_words = narrative.get('metadata', {}).get('transformed_word_count', 0)
            if orig_words and trans_words:
                word_changes.append(trans_words - orig_words)
        
        avg_word_change = sum(word_changes) / len(word_changes) if word_changes else 0
        
        print(f"\n📊 Bad > Good Narrative Analysis:")
        print(f"  Total narratives generated: {len(narratives)}")
        print(f"  Average word count change: {avg_word_change:+.1f}")
        print(f"  Cognitive patterns covered: {len(pattern_counts)}")
        print(f"  Exercise types used: {len(exercise_counts)}")
        
        print(f"\n📈 Top exercise categories:")
        for exercise, count in sorted(exercise_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {exercise}: {count}")

async def main():
    """Main function to generate bad > good narratives"""
    print("🔄 Starting Bad > Good Narrative Generation")
    
    # Configuration
    config = NarrativeConfig(
        model="mannix/llama3.1-8b-abliterated:latest",
        output_file="bad_good_narratives.jsonl",
        max_concurrent=3,
        temperature=0.7,
        max_tokens=400
    )
    
    # Check if cognitive dataset exists
    cognitive_dataset_file = "data/depressed_inductions/cognitive_patterns_dataset_cleaned.jsonl"
    if not Path(cognitive_dataset_file).exists():
        # Try CSV version
        cognitive_dataset_file = "data/depressed_inductions/cognitive_patterns_dataset_cleaned.csv"
        if not Path(cognitive_dataset_file).exists():
            print(f"❌ Cognitive patterns dataset not found. Run cognitive_pattern_dataset_generator.py first.")
            return
    
    # Initialize generator
    generator = BadGoodNarrativeGenerator(config)
    
    # Generate narratives
    print(f"📋 Using cognitive dataset: {cognitive_dataset_file}")
    narratives = await generator.generate_complete_narrative_dataset(
        cognitive_dataset_file, 
        narratives_per_pattern=2  # 2 different exercises per pattern
    )
    
    if narratives:
        # Save and analyze
        generator.save_final_dataset(narratives)
        generator.analyze_narratives(narratives)
        
        print("✅ Bad > Good narrative generation complete!")
        
        # Show sample
        if narratives:
            print(f"\n📝 Sample transformation:")
            sample = narratives[0]
            print(f"Original: {sample['original_thought_pattern'][:100]}...")
            print(f"Transformed: {sample['transformed_thought_pattern'][:100]}...")
            print(f"Exercise: {sample['exercise_category']} - {sample['exercise_title']}")
    else:
        print("❌ No narratives generated")
    
    return narratives

if __name__ == "__main__":
    asyncio.run(main())