#!/usr/bin/env python3
"""
Activation Inspector - Analyze and understand saved activation files.

This script helps you understand:
1. What cognitive patterns each activation file corresponds to
2. The structure and metadata of each activation file
3. Which layers and positions are captured
4. File sizes and tensor shapes
"""

import torch
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class ActivationInspector:
    """Inspector for analyzing saved activation files."""
    
    def __init__(self, activations_dir: str = "./activations/"):
        self.activations_dir = Path(activations_dir)
        self.cache_info = {}
        
    def scan_activations_directory(self) -> Dict[str, Any]:
        """Scan the activations directory and gather file information."""
        if not self.activations_dir.exists():
            print(f"❌ Activations directory not found: {self.activations_dir}")
            return {}
        
        activation_files = list(self.activations_dir.glob("activations_*.pt"))
        
        print(f"🔍 Found {len(activation_files)} activation files in {self.activations_dir}")
        print("=" * 60)
        
        file_info = {}
        
        for file_path in activation_files:
            cache_key = file_path.stem.replace("activations_", "")
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            file_info[cache_key] = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'cache_key': cache_key,
                'size_mb': round(file_size_mb, 2),
                'modified': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                'loaded': False,
                'content': None
            }
        
        return file_info
    
    def load_activation_file(self, file_path: str) -> Dict[str, Any]:
        """Load and analyze a single activation file."""
        try:
            print(f"📂 Loading: {file_path}")
            data = torch.load(file_path, map_location='cpu')
            
            analysis = {
                'file_path': file_path,
                'data_type': type(data).__name__,
                'keys': list(data.keys()) if isinstance(data, dict) else ['single_tensor'],
                'total_tensors': len(data) if isinstance(data, dict) else 1,
                'tensor_info': {}
            }
            
            if isinstance(data, dict):
                for key, tensor in data.items():
                    if torch.is_tensor(tensor):
                        analysis['tensor_info'][key] = {
                            'shape': list(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'device': str(tensor.device),
                            'requires_grad': tensor.requires_grad,
                            'memory_mb': tensor.numel() * tensor.element_size() / (1024 * 1024)
                        }
                    else:
                        analysis['tensor_info'][key] = {
                            'type': type(tensor).__name__,
                            'value': str(tensor)[:100] + "..." if len(str(tensor)) > 100 else str(tensor)
                        }
            elif torch.is_tensor(data):
                analysis['tensor_info']['main_tensor'] = {
                    'shape': list(data.shape),
                    'dtype': str(data.dtype),
                    'device': str(data.device),
                    'requires_grad': data.requires_grad,
                    'memory_mb': data.numel() * data.element_size() / (1024 * 1024)
                }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return {'error': str(e)}
    
    def reverse_engineer_cache_key(self, model_name: str = "google/gemma-2-2b-it", 
                                 layer_nums: List[int] = [17, 21], 
                                 position: str = "last") -> Dict[str, str]:
        """
        Try to reverse engineer what cognitive patterns might correspond to cache keys.
        This attempts to match the cache key generation logic from ActivationCapturer.
        """
        
        # Load config to get expected patterns
        config_path = Path("./config/config.yaml")
        expected_patterns = ["positive", "negative", "transition"]  # Default from config
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    expected_patterns = config.get('data', {}).get('cognitive_patterns', expected_patterns)
            except:
                pass
        
        cache_key_mapping = {}
        
        # Try to generate cache keys for expected patterns
        for pattern in expected_patterns:
            for num_strings in [1, 10, 50, 100, 500, 1000]:  # Common sample sizes
                content = f"{model_name}_{layer_nums}_{pattern}_{position}_{num_strings}"
                # We can't recreate the exact strings hash, but we can show the pattern
                cache_key_partial = hashlib.md5(content.encode()).hexdigest()[:16]
                cache_key_mapping[pattern] = {
                    'expected_pattern': cache_key_partial,
                    'full_content_template': content + "_<strings_hash>"
                }
        
        return cache_key_mapping
    
    def analyze_tensor_patterns(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze the tensor data to infer cognitive patterns."""
        analysis = {
            'inferred_patterns': [],
            'layer_info': {},
            'sample_counts': {},
            'tensor_relationships': {}
        }
        
        # Extract pattern names from keys
        for key in data.keys():
            if isinstance(key, str):
                # Keys typically look like: "positive_layer_17", "negative_layer_21", etc.
                parts = key.split('_')
                if len(parts) >= 3 and parts[-2] == 'layer':
                    pattern = '_'.join(parts[:-2])  # Everything before "_layer_X"
                    layer = parts[-1]
                    
                    if pattern not in analysis['inferred_patterns']:
                        analysis['inferred_patterns'].append(pattern)
                    
                    if pattern not in analysis['layer_info']:
                        analysis['layer_info'][pattern] = []
                    analysis['layer_info'][pattern].append(int(layer))
                    
                    # Get sample count from tensor shape
                    if torch.is_tensor(data[key]) and len(data[key].shape) > 0:
                        sample_count = data[key].shape[0]
                        analysis['sample_counts'][pattern] = sample_count
        
        return analysis
    
    def create_inspection_report(self) -> Dict[str, Any]:
        """Create a comprehensive inspection report."""
        print("🚀 Creating Activation Inspection Report")
        print("=" * 60)
        
        # Scan directory
        file_info = self.scan_activations_directory()
        if not file_info:
            return {'error': 'No activation files found'}
        
        # Load and analyze each file
        detailed_analysis = {}
        
        for cache_key, info in file_info.items():
            print(f"\n📊 Analyzing {info['filename']}...")
            
            # Load file
            file_analysis = self.load_activation_file(info['filepath'])
            
            if 'error' not in file_analysis:
                # Load the actual data for pattern analysis
                try:
                    data = torch.load(info['filepath'], map_location='cpu')
                    pattern_analysis = self.analyze_tensor_patterns(data)
                    file_analysis['pattern_analysis'] = pattern_analysis
                    
                    print(f"  ✓ Detected patterns: {pattern_analysis['inferred_patterns']}")
                    print(f"  ✓ Layers: {pattern_analysis['layer_info']}")
                    print(f"  ✓ Sample counts: {pattern_analysis['sample_counts']}")
                    
                except Exception as e:
                    file_analysis['pattern_analysis'] = {'error': str(e)}
            
            detailed_analysis[cache_key] = file_analysis
        
        # Try to reverse engineer cache keys
        cache_key_mapping = self.reverse_engineer_cache_key()
        
        report = {
            'scan_timestamp': datetime.now().isoformat(),
            'activations_directory': str(self.activations_dir),
            'total_files': len(file_info),
            'file_info': file_info,
            'detailed_analysis': detailed_analysis,
            'cache_key_patterns': cache_key_mapping,
            'summary': self._create_summary(detailed_analysis)
        }
        
        return report
    
    def _create_summary(self, detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the analysis."""
        all_patterns = set()
        all_layers = set()
        total_size_mb = 0
        file_count = 0
        
        for cache_key, analysis in detailed_analysis.items():
            if 'error' not in analysis and 'pattern_analysis' in analysis:
                pattern_info = analysis['pattern_analysis']
                all_patterns.update(pattern_info.get('inferred_patterns', []))
                
                for pattern, layers in pattern_info.get('layer_info', {}).items():
                    all_layers.update(layers)
                
                # Sum up memory usage
                for tensor_info in analysis.get('tensor_info', {}).values():
                    if isinstance(tensor_info, dict) and 'memory_mb' in tensor_info:
                        total_size_mb += tensor_info['memory_mb']
                
                file_count += 1
        
        return {
            'cognitive_patterns_found': list(all_patterns),
            'layers_captured': sorted(list(all_layers)),
            'total_memory_mb': round(total_size_mb, 2),
            'successfully_analyzed_files': file_count
        }
    
    def save_report(self, report: Dict[str, Any], output_path: str = "activation_inspection_report.json"):
        """Save the inspection report to a JSON file."""
        # Convert any non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, torch.dtype):
                return str(obj)
            elif isinstance(obj, torch.device):
                return str(obj)
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        # Deep copy and convert
        import copy
        serializable_report = copy.deepcopy(report)
        
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return make_serializable(d)
        
        serializable_report = convert_dict(serializable_report)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to: {output_path}")
    
    def print_summary_table(self, report: Dict[str, Any]):
        """Print a nice summary table of the findings."""
        print("\n" + "=" * 80)
        print("📋 ACTIVATION FILES SUMMARY")
        print("=" * 80)
        
        if 'summary' in report:
            summary = report['summary']
            print(f"🧠 Cognitive Patterns Found: {', '.join(summary['cognitive_patterns_found'])}")
            print(f"🔢 Layers Captured: {summary['layers_captured']}")
            print(f"💾 Total Memory Usage: {summary['total_memory_mb']} MB")
            print(f"✅ Successfully Analyzed: {summary['successfully_analyzed_files']} files")
        
        print("\n📁 FILE DETAILS:")
        print("-" * 80)
        
        if 'detailed_analysis' in report:
            for cache_key, analysis in report['detailed_analysis'].items():
                if 'error' in analysis:
                    print(f"❌ {cache_key}: ERROR - {analysis['error']}")
                else:
                    patterns = analysis.get('pattern_analysis', {}).get('inferred_patterns', ['unknown'])
                    sample_counts = analysis.get('pattern_analysis', {}).get('sample_counts', {})
                    
                    print(f"✅ {cache_key}:")
                    print(f"   📊 Patterns: {', '.join(patterns)}")
                    print(f"   📈 Samples: {sample_counts}")
                    print(f"   💽 Tensors: {analysis.get('total_tensors', 0)}")


def main():
    """Main function to run the activation inspection."""
    print("🔍 Activation Inspector")
    print("=" * 50)
    
    inspector = ActivationInspector()
    
    # Create comprehensive report
    report = inspector.create_inspection_report()
    
    if 'error' in report:
        print(f"❌ {report['error']}")
        return
    
    # Print summary
    inspector.print_summary_table(report)
    
    # Save detailed report
    inspector.save_report(report)
    
    print("\n🎉 Inspection complete!")
    print("\nTo load and work with specific activations:")
    print("```python")
    print("import torch")
    print("data = torch.load('activations/activations_<cache_key>.pt')")
    print("print(data.keys())  # See available patterns and layers")
    print("```")


if __name__ == "__main__":
    main()
