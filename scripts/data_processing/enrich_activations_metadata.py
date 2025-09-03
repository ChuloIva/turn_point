#!/usr/bin/env python3
"""
Enrich Activations Metadata - Combine all three datasets and add metadata to activations.

This script:
1. Loads and aligns the three datasets (cognitive_patterns, bad_good_narratives, positive_patterns)
2. Creates enriched metadata combining all sources
3. Adds the metadata to activation files

Based on your analysis, the activations order matches positive_patterns.jsonl order.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd

def load_jsonl_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                item['_line_number'] = line_num
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    return data

def find_text_matches(datasets: Dict[str, List[Dict]], max_samples: int = 520) -> Dict[str, Any]:
    """
    Find matches across the three datasets and create alignment mapping.
    
    The datasets have different structures:
    - cognitive_patterns: has 'thought_pattern'
    - bad_good_narratives: has 'original_thought_pattern' and 'transformed_thought_pattern'  
    - positive_patterns: has 'positive_thought_pattern' and 'reference_negative_example'
    
    We need to find which entries correspond to each other.
    """
    
    # Extract all unique thought patterns from each dataset
    cognitive_patterns = set()
    bad_good_originals = set()  
    positive_references = set()
    
    for item in datasets['cognitive_patterns'][:max_samples]:
        if 'thought_pattern' in item:
            cognitive_patterns.add(item['thought_pattern'].strip())
    
    for item in datasets['bad_good_narratives'][:max_samples]:
        if 'original_thought_pattern' in item:
            bad_good_originals.add(item['original_thought_pattern'].strip())
    
    for item in datasets['positive_patterns'][:max_samples]:
        if 'reference_negative_example' in item:
            positive_references.add(item['reference_negative_example'].strip())
    
    # Find intersections
    cognitive_vs_bad_good = cognitive_patterns & bad_good_originals
    cognitive_vs_positive = cognitive_patterns & positive_references
    bad_good_vs_positive = bad_good_originals & positive_references
    all_three_match = cognitive_patterns & bad_good_originals & positive_references
    
    print(f"🔍 Text Pattern Alignment Analysis:")
    print(f"   Cognitive patterns dataset: {len(cognitive_patterns)} unique patterns")
    print(f"   Bad/good narratives dataset: {len(bad_good_originals)} unique original patterns")
    print(f"   Positive patterns dataset: {len(positive_references)} unique reference negative patterns")
    print(f"   Cognitive ∩ Bad/Good: {len(cognitive_vs_bad_good)} matches")
    print(f"   Cognitive ∩ Positive: {len(cognitive_vs_positive)} matches") 
    print(f"   Bad/Good ∩ Positive: {len(bad_good_vs_positive)} matches")
    print(f"   All three datasets: {len(all_three_match)} matches")
    
    return {
        'cognitive_patterns': cognitive_patterns,
        'bad_good_originals': bad_good_originals,
        'positive_references': positive_references,
        'matches': {
            'cognitive_vs_bad_good': cognitive_vs_bad_good,
            'cognitive_vs_positive': cognitive_vs_positive,
            'bad_good_vs_positive': bad_good_vs_positive,
            'all_three': all_three_match
        }
    }

def create_enriched_metadata(datasets: Dict[str, List[Dict]], alignment_info: Dict[str, Any], max_samples: int = 520) -> List[Dict[str, Any]]:
    """
    Create enriched metadata by combining information from all three datasets.
    
    Since activations are aligned with positive_patterns.jsonl, we'll use that as the base
    and enrich it with information from the other datasets where matches exist.
    """
    
    enriched_data = []
    
    # Create lookup dictionaries for faster matching
    cognitive_lookup = {}
    for item in datasets['cognitive_patterns'][:max_samples]:
        if 'thought_pattern' in item:
            cognitive_lookup[item['thought_pattern'].strip()] = item
    
    # Create lookup for bad_good_narratives using both original AND transformed patterns as key
    bad_good_lookup = {}
    for item in datasets['bad_good_narratives'][:max_samples]:
        if 'original_thought_pattern' in item and 'transformed_thought_pattern' in item:
            # Use tuple of (original, transformed) as key to handle multiple transformations of same negative pattern
            key = (item['original_thought_pattern'].strip(), item['transformed_thought_pattern'].strip())
            bad_good_lookup[key] = item
    
    # Also create a simple lookup by original pattern for fallback
    bad_good_simple_lookup = {}
    for item in datasets['bad_good_narratives'][:max_samples]:
        if 'original_thought_pattern' in item:
            original = item['original_thought_pattern'].strip()
            # Keep track of all transformations for this negative pattern
            if original not in bad_good_simple_lookup:
                bad_good_simple_lookup[original] = []
            bad_good_simple_lookup[original].append(item)
    
    # Use positive_patterns as the base since activations are aligned with this
    positive_patterns = datasets['positive_patterns'][:max_samples]
    
    for idx, positive_item in enumerate(positive_patterns):
        # Start with the positive pattern data
        enriched_entry = {
            'activation_index': idx,
            'source_line_positive_patterns': positive_item.get('_line_number'),
            'positive_thought_pattern': positive_item.get('positive_thought_pattern'),
            'reference_negative_example': positive_item.get('reference_negative_example'),
            'reference_transformed_example': positive_item.get('reference_transformed_example'),
            'cognitive_pattern_name': positive_item.get('cognitive_pattern_name'),
            'cognitive_pattern_type': positive_item.get('cognitive_pattern_type'),
            'pattern_description': positive_item.get('pattern_description'),
            'source_question': positive_item.get('source_question'),
            'model': positive_item.get('model'),
            'timestamp': positive_item.get('timestamp'),
            'metadata': positive_item.get('metadata', {}),
            'word_count': positive_item.get('metadata', {}).get('word_count'),
            'temperature': positive_item.get('metadata', {}).get('temperature'),
        }
        
        # Try to match with cognitive_patterns dataset
        reference_text = positive_item.get('reference_negative_example', '').strip()
        if reference_text in cognitive_lookup:
            cognitive_match = cognitive_lookup[reference_text]
            enriched_entry['cognitive_patterns_match'] = {
                'source_line_cognitive_patterns': cognitive_match.get('_line_number'),
                'thought_pattern': cognitive_match.get('thought_pattern'),
                'cognitive_pattern_name_from_cognitive': cognitive_match.get('cognitive_pattern_name'),
                'cognitive_pattern_type_from_cognitive': cognitive_match.get('cognitive_pattern_type'),
                'pattern_description_from_cognitive': cognitive_match.get('pattern_description'),
                'source_question_from_cognitive': cognitive_match.get('source_question'),
                'model_from_cognitive': cognitive_match.get('model'),
                'timestamp_from_cognitive': cognitive_match.get('timestamp'),
                'metadata_from_cognitive': cognitive_match.get('metadata', {})
            }
        else:
            enriched_entry['cognitive_patterns_match'] = None
        
        # Try to match with bad_good_narratives dataset using precise matching
        reference_transformed_text = positive_item.get('reference_transformed_example', '').strip()
        
        # First try: exact match using both negative and transformed patterns
        precise_key = (reference_text, reference_transformed_text)
        bad_good_match = None
        
        if precise_key in bad_good_lookup:
            bad_good_match = bad_good_lookup[precise_key]
            print(f"✅ Precise match found for activation {idx}")
        elif reference_text in bad_good_simple_lookup:
            # Fallback: find the entry where transformed pattern matches
            candidates = bad_good_simple_lookup[reference_text]
            for candidate in candidates:
                if candidate.get('transformed_thought_pattern', '').strip() == reference_transformed_text:
                    bad_good_match = candidate
                    print(f"✅ Fallback match found for activation {idx}")
                    break
            
            if not bad_good_match:
                # Last resort: use the first candidate and log the mismatch
                bad_good_match = candidates[0]
                print(f"⚠️  Using first available match for activation {idx} (potential mismatch)")
        
        if bad_good_match:
            enriched_entry['bad_good_narratives_match'] = {
                'source_line_bad_good_narratives': bad_good_match.get('_line_number'),
                'original_thought_pattern': bad_good_match.get('original_thought_pattern'),
                'transformed_thought_pattern': bad_good_match.get('transformed_thought_pattern'),
                'cognitive_pattern_name_from_bad_good': bad_good_match.get('cognitive_pattern_name'),
                'cognitive_pattern_type_from_bad_good': bad_good_match.get('cognitive_pattern_type'),
                'pattern_description_from_bad_good': bad_good_match.get('pattern_description'),
                'source_question_from_bad_good': bad_good_match.get('source_question'),
                'exercise_category': bad_good_match.get('exercise_category'),
                'exercise_title': bad_good_match.get('exercise_title'),
                'exercise_content': bad_good_match.get('exercise_content'),
                'emergency_intervention': bad_good_match.get('emergency_intervention'),
                'model_from_bad_good': bad_good_match.get('model'),
                'timestamp_from_bad_good': bad_good_match.get('timestamp'),
                'metadata_from_bad_good': bad_good_match.get('metadata', {}),
                'original_word_count': bad_good_match.get('metadata', {}).get('original_word_count'),
                'transformed_word_count': bad_good_match.get('metadata', {}).get('transformed_word_count')
            }
        else:
            enriched_entry['bad_good_narratives_match'] = None
        
        # Add data source summary
        enriched_entry['data_sources'] = {
            'has_positive_patterns': True,
            'has_cognitive_patterns_match': enriched_entry['cognitive_patterns_match'] is not None,
            'has_bad_good_narratives_match': enriched_entry['bad_good_narratives_match'] is not None,
            'total_sources': 1 + (1 if enriched_entry['cognitive_patterns_match'] else 0) + (1 if enriched_entry['bad_good_narratives_match'] else 0)
        }
        
        enriched_data.append(enriched_entry)
    
    return enriched_data

def add_metadata_to_activations(enriched_metadata: List[Dict[str, Any]], activations_dir: Path) -> None:
    """
    Add enriched metadata to activation files.
    
    Based on verify_activation_order.py, the file mappings are:
    - positive: activations_e5ad16e9b3c33c9b.pt
    - negative: activations_8ff00d963316212d.pt  
    - transition: activations_332f24de2a3f82ff.pt
    """
    
    file_mappings = {
        "positive": "activations_e5ad16e9b3c33c9b.pt",
        "negative": "activations_8ff00d963316212d.pt", 
        "transition": "activations_332f24de2a3f82ff.pt"
    }
    
    for pattern_type, filename in file_mappings.items():
        activation_path = activations_dir / filename
        
        if not activation_path.exists():
            print(f"❌ Activation file not found: {activation_path}")
            continue
        
        print(f"🔄 Processing {pattern_type} activations: {filename}")
        
        # Load existing activation data
        try:
            activation_data = torch.load(activation_path, map_location='cpu')
            print(f"✅ Loaded existing activation data")
        except Exception as e:
            print(f"❌ Error loading activation data: {e}")
            continue
        
        # Add enriched metadata
        activation_data['enriched_metadata'] = enriched_metadata
        activation_data['metadata_info'] = {
            'total_samples': len(enriched_metadata),
            'sources_combined': ['positive_patterns.jsonl', 'cognitive_patterns_dataset_cleaned.jsonl', 'bad_good_narratives.jsonl'],
            'alignment_base': 'positive_patterns.jsonl',
            'enriched_timestamp': pd.Timestamp.now().isoformat(),
            'enrichment_version': '1.0'
        }
        
        # Create backup
        backup_path = activation_path.with_suffix('.pt.backup')
        if not backup_path.exists():
            print(f"📋 Creating backup: {backup_path}")
            torch.save(torch.load(activation_path, map_location='cpu'), backup_path)
        
        # Save enriched data
        print(f"💾 Saving enriched activation data...")
        torch.save(activation_data, activation_path)
        print(f"✅ Successfully enriched {pattern_type} activations")
        
        # Print some statistics
        activation_keys = [k for k in activation_data.keys() if k.startswith(f"{pattern_type}_layer_")]
        print(f"📊 Activation layers: {len(activation_keys)}")
        if activation_keys:
            sample_shape = activation_data[activation_keys[0]].shape
            print(f"📊 Sample shape: {sample_shape}")
        
        # Count metadata sources
        total_with_cognitive = sum(1 for item in enriched_metadata if item['data_sources']['has_cognitive_patterns_match'])
        total_with_bad_good = sum(1 for item in enriched_metadata if item['data_sources']['has_bad_good_narratives_match'])
        total_with_all_three = sum(1 for item in enriched_metadata if item['data_sources']['total_sources'] == 3)
        
        print(f"📊 Metadata coverage:")
        print(f"   - Total samples: {len(enriched_metadata)}")
        print(f"   - With cognitive_patterns match: {total_with_cognitive}")
        print(f"   - With bad_good_narratives match: {total_with_bad_good}")
        print(f"   - With all three sources: {total_with_all_three}")

def main():
    """Main enrichment function."""
    print("🚀 Activation Metadata Enrichment")
    print("=" * 50)
    
    # Configuration
    data_dir = Path("data/final")
    activations_dir = Path("activations")
    max_samples = 520  # Based on verification script
    
    # Dataset file paths
    dataset_files = {
        'cognitive_patterns': data_dir / "cognitive_patterns_dataset_cleaned.jsonl",
        'bad_good_narratives': data_dir / "bad_good_narratives.jsonl",
        'positive_patterns': data_dir / "positive_patterns.jsonl"
    }
    
    # Load all datasets
    datasets = {}
    for name, path in dataset_files.items():
        if not path.exists():
            print(f"❌ Dataset file not found: {path}")
            return
        
        try:
            datasets[name] = load_jsonl_data(str(path))
            print(f"✅ Loaded {len(datasets[name])} samples from {name}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            return
    
    # Analyze alignment across datasets
    print(f"\n🔍 Analyzing data alignment...")
    alignment_info = find_text_matches(datasets, max_samples)
    
    # Create enriched metadata
    print(f"\n🔨 Creating enriched metadata...")
    enriched_metadata = create_enriched_metadata(datasets, alignment_info, max_samples)
    
    print(f"✅ Created enriched metadata for {len(enriched_metadata)} samples")
    
    # Add metadata to activation files
    print(f"\n💾 Adding metadata to activation files...")
    add_metadata_to_activations(enriched_metadata, activations_dir)
    
    # Save standalone enriched metadata file
    metadata_file = data_dir / "enriched_metadata.json"
    print(f"\n📄 Saving standalone metadata file: {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_metadata, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n🎉 Enrichment complete!")
    print(f"\n💡 SUMMARY:")
    print(f"   • Combined data from 3 datasets")
    print(f"   • Enriched {len(enriched_metadata)} activation samples")
    print(f"   • Added metadata to all activation files")
    print(f"   • Created backups of original files")
    print(f"   • Saved standalone metadata file")
    
    # Show some sample enriched data
    print(f"\n📋 Sample enriched entry (first entry):")
    if enriched_metadata:
        sample = enriched_metadata[0]
        print(f"   Activation Index: {sample['activation_index']}")
        print(f"   Cognitive Pattern: {sample['cognitive_pattern_name']}")
        print(f"   Data Sources: {sample['data_sources']['total_sources']}")
        print(f"   Has Cognitive Match: {sample['data_sources']['has_cognitive_patterns_match']}")
        print(f"   Has Bad/Good Match: {sample['data_sources']['has_bad_good_narratives_match']}")

if __name__ == "__main__":
    main()