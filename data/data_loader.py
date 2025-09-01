import json
import os
from typing import List, Dict, Optional, Union
from pathlib import Path


class DataLoader:
    """Modular data loading for cognitive pattern datasets."""
    
    def __init__(self, base_path: str = "./data/cognitive_patterns/"):
        self.base_path = Path(base_path)
        self.cognitive_patterns = {}
        
    def load_cognitive_patterns(self, pattern_names: List[str]) -> Dict[str, List[str]]:
        """
        Load cognitive pattern datasets.
        
        Args:
            pattern_names: List of pattern names to load
            
        Returns:
            Dictionary mapping pattern names to string lists
        """
        loaded_patterns = {}
        
        for pattern_name in pattern_names:
            pattern_file = self.base_path / f"{pattern_name}.jsonl"
            
            if pattern_file.exists():
                loaded_patterns[pattern_name] = self._load_jsonl(pattern_file)
            else:
                # Try alternative extensions
                alternatives = [f"{pattern_name}.json", f"{pattern_name}.txt"]
                loaded = False
                
                for alt in alternatives:
                    alt_file = self.base_path / alt
                    if alt_file.exists():
                        if alt.endswith('.json'):
                            loaded_patterns[pattern_name] = self._load_json(alt_file)
                        elif alt.endswith('.txt'):
                            loaded_patterns[pattern_name] = self._load_txt(alt_file)
                        loaded = True
                        break
                
                if not loaded:
                    print(f"Warning: Could not find data file for pattern '{pattern_name}'")
                    loaded_patterns[pattern_name] = []
        
        self.cognitive_patterns = loaded_patterns
        return loaded_patterns
    
    def _load_jsonl(self, filepath: Path) -> List[str]:
        """Load strings from JSONL file."""
        strings = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Extract text field (flexible field names)
                if isinstance(data, dict):
                    # Check for cognitive pattern specific fields first
                    text = (data.get('positive_thought_pattern', 
                           data.get('negative_thought_pattern',
                           data.get('transition_thought_pattern',
                           data.get('text', 
                           data.get('content', 
                           data.get('string', str(data))))))))
                else:
                    text = str(data)
                strings.append(text)
        return strings
    
    def _load_json(self, filepath: Path) -> List[str]:
        """Load strings from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [str(item) for item in data]
        elif isinstance(data, dict):
            # Try to extract strings from various possible structures
            if 'strings' in data:
                return data['strings']
            elif 'data' in data:
                return [str(item) for item in data['data']]
            else:
                return [str(v) for v in data.values()]
        else:
            return [str(data)]
    
    def _load_txt(self, filepath: Path) -> List[str]:
        """Load strings from text file (one per line)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def get_pattern_strings(self, pattern_name: str) -> List[str]:
        """Get strings for a specific cognitive pattern."""
        return self.cognitive_patterns.get(pattern_name, [])
    
    def get_all_patterns(self) -> Dict[str, List[str]]:
        """Get all loaded cognitive patterns."""
        return self.cognitive_patterns
    
    def filter_patterns(self, pattern_names: List[str]) -> Dict[str, List[str]]:
        """Filter loaded patterns to specific subset."""
        return {name: self.cognitive_patterns.get(name, []) 
                for name in pattern_names 
                if name in self.cognitive_patterns}
    
    def combine_patterns(self, pattern_names: List[str], new_name: str) -> List[str]:
        """Combine multiple patterns into one dataset."""
        combined = []
        for pattern_name in pattern_names:
            if pattern_name in self.cognitive_patterns:
                combined.extend(self.cognitive_patterns[pattern_name])
        
        self.cognitive_patterns[new_name] = combined
        return combined
    
    def save_pattern(self, pattern_name: str, strings: List[str], format: str = 'jsonl') -> None:
        """Save a cognitive pattern to file."""
        if format == 'jsonl':
            filepath = self.base_path / f"{pattern_name}.jsonl"
            with open(filepath, 'w', encoding='utf-8') as f:
                for string in strings:
                    json.dump({'text': string}, f)
                    f.write('\n')
        elif format == 'json':
            filepath = self.base_path / f"{pattern_name}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({'strings': strings}, f, indent=2)
        elif format == 'txt':
            filepath = self.base_path / f"{pattern_name}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(strings))
    
    def list_available_patterns(self) -> List[str]:
        """List available pattern files in the data directory."""
        pattern_files = []
        for ext in ['.jsonl', '.json', '.txt']:
            pattern_files.extend([
                f.stem for f in self.base_path.glob(f"*{ext}")
            ])
        return list(set(pattern_files))
    
    def load_cognitive_pattern_types(self, jsonl_filepath: str) -> Dict[str, List[str]]:
        """
        Load cognitive patterns separated by type from your final dataset format.
        
        Args:
            jsonl_filepath: Path to the JSONL file containing pattern data
            
        Returns:
            Dictionary with 'positive', 'negative', 'transition' pattern lists
        """
        patterns = {'positive': [], 'negative': [], 'transition': []}
        
        filepath = Path(jsonl_filepath)
        if not filepath.exists():
            print(f"Warning: Could not find file '{jsonl_filepath}'")
            return patterns
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract positive pattern
                    if 'positive_thought_pattern' in data:
                        patterns['positive'].append(data['positive_thought_pattern'])
                    
                    # Extract negative pattern
                    if 'reference_negative_example' in data:
                        patterns['negative'].append(data['reference_negative_example'])
                    
                    # Extract transition/transformation pattern
                    if 'reference_transformed_example' in data:
                        patterns['transition'].append(data['reference_transformed_example'])
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
        
        # Update internal state
        self.cognitive_patterns.update(patterns)
        return patterns
    
    def get_pattern_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get statistics for loaded patterns."""
        stats = {}
        for pattern_name, strings in self.cognitive_patterns.items():
            if strings:
                lengths = [len(s) for s in strings]
                stats[pattern_name] = {
                    'count': len(strings),
                    'avg_length': sum(lengths) / len(lengths),
                    'min_length': min(lengths),
                    'max_length': max(lengths)
                }
            else:
                stats[pattern_name] = {'count': 0, 'avg_length': 0, 'min_length': 0, 'max_length': 0}
        return stats