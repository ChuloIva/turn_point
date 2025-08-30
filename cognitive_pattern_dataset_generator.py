import ollama
import pandas as pd
import json
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import logging

@dataclass
class DatasetConfig:
    """Configuration for cognitive pattern dataset generation"""
    model: str = "mannix/llama3.1-8b-abliterated:latest"
    output_file: str = "cognitive_patterns_dataset.jsonl"
    batch_size: int = 5
    max_concurrent: int = 3
    temperature: float = 0.8
    max_tokens: int = 300
    system_prompt: str = """You are generating authentic first-person thought patterns of someone experiencing specific cognitive patterns related to mental health struggles. Generate realistic internal monologue that captures the essence of the cognitive pattern. Write in first person, present tense, as if these are actual thoughts going through someone's mind. Be authentic and psychologically accurate while being respectful of mental health experiences."""

class CognitivePatternDatasetGenerator:
    """Generate dataset of first-person thought patterns for cognitive patterns"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.async_client = ollama.AsyncClient()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Load cognitive patterns and questions
        self.cognitive_patterns = self.load_cognitive_patterns()
        self.questions_by_pattern = self.load_questions_by_pattern()
        
    def load_cognitive_patterns(self) -> pd.DataFrame:
        """Load cognitive patterns from CSV"""
        return pd.read_csv('data/cognitive_patterns_short14.csv')
    
    def load_questions_by_pattern(self) -> Dict[str, List[str]]:
        """Parse questions markdown file and organize by pattern"""
        questions_by_pattern = {}
        
        with open('data/cognitive_pattern_questions.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by pattern sections (## followed by number)
        pattern_sections = re.split(r'\n## \d+\.', content)[1:]  # Skip header
        
        for i, section in enumerate(pattern_sections):
            if i < len(self.cognitive_patterns):
                # Extract pattern name (first line before *)
                lines = section.strip().split('\n')
                pattern_name = lines[0].strip()
                
                # Extract questions (lines that start with numbers)
                questions = []
                for line in lines:
                    if re.match(r'^\d+\.', line):
                        question = re.sub(r'^\d+\.\s*', '', line).strip()
                        questions.append(question)
                
                questions_by_pattern[pattern_name] = questions
        
        return questions_by_pattern
    
    def create_generation_prompt(self, pattern_name: str, pattern_description: str, question: str) -> str:
        """Create a prompt for generating first-person thought patterns"""
        return f"""Based on this cognitive pattern and question, generate authentic first-person thoughts that someone might have when experiencing this pattern:

Cognitive Pattern: {pattern_name}
Description: {pattern_description}
Context Question: {question}

Generate 2-3 sentences of realistic first-person internal thoughts that capture this cognitive pattern. Write as if you are thinking these thoughts yourself. Be authentic and specific, showing the actual mental experience rather than describing it.

Example format: "I keep thinking about... My mind won't stop... I can't help but..."

First-person thoughts:"""

    async def generate_single_thought_pattern(self, pattern_name: str, pattern_description: str, question: str, pattern_type: str) -> Dict[str, Any]:
        """Generate a single first-person thought pattern"""
        try:
            prompt = self.create_generation_prompt(pattern_name, pattern_description, question)
            
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
            
            thought_pattern = response['message']['content'].strip()
            
            # Clean up any formatting artifacts
            thought_pattern = re.sub(r'^(First-person thoughts?:?\s*)', '', thought_pattern, flags=re.IGNORECASE)
            thought_pattern = re.sub(r'^["\'`](.*)["\'`]$', r'\1', thought_pattern.strip())
            
            return {
                'thought_pattern': thought_pattern,
                'cognitive_pattern_name': pattern_name,
                'cognitive_pattern_type': pattern_type,
                'pattern_description': pattern_description,
                'source_question': question,
                'model': self.config.model,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'temperature': self.config.temperature,
                    'word_count': len(thought_pattern.split())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating thought pattern: {e}")
            return {
                'thought_pattern': None,
                'cognitive_pattern_name': pattern_name,
                'cognitive_pattern_type': pattern_type,
                'pattern_description': pattern_description,
                'source_question': question,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_for_pattern(self, pattern_row: pd.Series, questions: List[str]) -> List[Dict[str, Any]]:
        """Generate thought patterns for a specific cognitive pattern"""
        pattern_name = pattern_row['Concept Name']
        pattern_type = pattern_row['Cognitive Pattern']
        pattern_description = pattern_row['Description']
        
        self.logger.info(f"Generating for pattern: {pattern_name} ({len(questions)} questions)")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def bounded_generate(question):
            async with semaphore:
                return await self.generate_single_thought_pattern(
                    pattern_name, pattern_description, question, pattern_type
                )
        
        # Generate for all questions for this pattern
        tasks = [bounded_generate(question) for question in questions]
        results = []
        
        for coro in tqdm(asyncio.as_completed(tasks), 
                        total=len(tasks), 
                        desc=f"Generating {pattern_name[:30]}..."):
            result = await coro
            results.append(result)
            
            # Save incrementally to avoid data loss
            if len(results) % 5 == 0:
                self.save_incremental(results, pattern_name)
        
        return results
    
    def save_incremental(self, results: List[Dict[str, Any]], pattern_name: str):
        """Save results incrementally"""
        filename = f"incremental_{pattern_name.replace(' ', '_').replace('&', 'and').lower()}.jsonl"
        filepath = Path(filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in results:
                if result.get('thought_pattern'):
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    async def generate_complete_dataset(self) -> List[Dict[str, Any]]:
        """Generate complete dataset for all cognitive patterns"""
        all_results = []
        
        self.logger.info(f"Starting dataset generation for {len(self.cognitive_patterns)} cognitive patterns")
        
        for idx, pattern_row in self.cognitive_patterns.iterrows():
            pattern_name = pattern_row['Concept Name']
            
            # Find matching questions for this pattern
            matching_questions = None
            for q_pattern_name, questions in self.questions_by_pattern.items():
                if q_pattern_name in pattern_name or pattern_name in q_pattern_name:
                    matching_questions = questions
                    break
            
            if not matching_questions:
                # Fallback: use first available questions if no exact match
                matching_questions = list(self.questions_by_pattern.values())[idx] if idx < len(self.questions_by_pattern) else []
            
            if matching_questions:
                pattern_results = await self.generate_for_pattern(pattern_row, matching_questions)
                all_results.extend(pattern_results)
                
                # Brief pause between patterns to be gentle on the API
                await asyncio.sleep(1)
            else:
                self.logger.warning(f"No questions found for pattern: {pattern_name}")
        
        return all_results
    
    def save_final_dataset(self, results: List[Dict[str, Any]]):
        """Save final combined dataset"""
        # Filter successful generations
        successful_results = [r for r in results if r.get('thought_pattern')]
        
        # Save as JSONL
        output_path = Path(self.config.output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in successful_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Also save as CSV for easy viewing
        csv_data = []
        for result in successful_results:
            csv_data.append({
                'thought_pattern': result['thought_pattern'],
                'cognitive_pattern_name': result['cognitive_pattern_name'],
                'cognitive_pattern_type': result['cognitive_pattern_type'],
                'pattern_description': result['pattern_description'],
                'source_question': result['source_question'],
                'word_count': result.get('metadata', {}).get('word_count', 0),
                'timestamp': result['timestamp']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        self.logger.info(f"Saved {len(successful_results)} thought patterns to {output_path}")
        self.logger.info(f"CSV version saved to {output_path.with_suffix('.csv')}")
        
        return successful_results
    
    def analyze_dataset(self, results: List[Dict[str, Any]]):
        """Analyze generated dataset"""
        successful = [r for r in results if r.get('thought_pattern')]
        failed = len(results) - len(successful)
        
        # Analyze by pattern
        pattern_counts = {}
        word_counts = []
        
        for result in successful:
            pattern = result['cognitive_pattern_name']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            word_counts.append(result.get('metadata', {}).get('word_count', 0))
        
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        
        print(f"\n📊 Dataset Generation Analysis:")
        print(f"  Total generated: {len(successful)}")
        print(f"  Failed: {failed}")
        print(f"  Average words per thought: {avg_words:.1f}")
        print(f"  Patterns covered: {len(pattern_counts)}")
        
        print(f"\n📋 Examples per pattern:")
        for pattern, count in sorted(pattern_counts.items()):
            print(f"  {pattern}: {count}")

async def main():
    """Main function to run the complete pipeline"""
    print("🧠 Starting Cognitive Pattern Dataset Generation")
    
    # Configuration
    config = DatasetConfig(
        model="mannix/llama3.1-8b-abliterated:latest",  # Change to available model
        output_file="cognitive_patterns_dataset.jsonl",
        batch_size=5,
        max_concurrent=3,
        temperature=0.8,
        max_tokens=300
    )
    
    # Initialize generator
    generator = CognitivePatternDatasetGenerator(config)
    
    # Generate complete dataset
    print(f"📋 Loaded {len(generator.cognitive_patterns)} cognitive patterns")
    print(f"📋 Loaded questions for {len(generator.questions_by_pattern)} patterns")
    
    results = await generator.generate_complete_dataset()
    
    # Save and analyze
    final_results = generator.save_final_dataset(results)
    generator.analyze_dataset(results)
    
    print("✅ Dataset generation complete!")
    return final_results

if __name__ == "__main__":
    asyncio.run(main())