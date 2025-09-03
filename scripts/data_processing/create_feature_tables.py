#!/usr/bin/env python3
"""
Script to create feature interpretation tables from SAE analysis JSON files.
Creates tables showing token strings and their activating features with interpretations.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import sys
import torch
from transformers import AutoTokenizer

def load_json_data(file_path):
    """Load and parse the JSON data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)

def get_tokens_from_text(text, model_name="google/gemma-2-2b-it"):
    """Tokenize text using the same tokenizer as the original analysis."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(text, return_tensors="pt")[0]
        token_strings = [tokenizer.decode([token]) for token in tokens]
        return token_strings
    except Exception as e:
        print(f"Warning: Could not load tokenizer, falling back to word splitting: {e}")
        return text.split()

def get_token_from_position(token_strings, position):
    """Get token string from tokenized list at given position."""
    if position < len(token_strings):
        return token_strings[position]
    return f"<token_{position}>"

def create_pattern_table(pattern_name, pattern_data, all_feature_info, external_features=None):
    """Create a table for a single pattern showing tokens and their features."""
    
    print(f"\n{'='*80}")
    print(f"PATTERN: {pattern_name}")
    print(f"{'='*80}")
    print(f"Text: {pattern_data['pattern_text']}")
    print(f"Sequence Length: {pattern_data['sequence_length']}")
    print(f"Average L0 Norm: {pattern_data['avg_l0_norm']:.2f}")
    print(f"Reconstruction MSE: {pattern_data['reconstruction_mse']:.4f}")
    print()
    
    # Get pattern-specific feature info
    pattern_features = pattern_data.get('top_features', [])
    feature_lookup = {f['feature_index']: f for f in pattern_features}
    
    # Tokenize the pattern text once
    pattern_text = pattern_data['pattern_text']
    token_strings = get_tokens_from_text(pattern_text)
    
    print(f"Tokenized text into {len(token_strings)} tokens:")
    for i, token in enumerate(token_strings):
        print(f"  {i}: '{token}'")
    print()
    
    # Create a comprehensive table for all tokens
    all_data = []
    
    for token_info in pattern_data['token_analysis']:
        position = token_info['position']
        relative_pos = token_info['relative_position']
        l0_norm = token_info['l0_norm']
        
        # Get proper token text from tokenizer
        # For last_10 analysis, map position within last 10 to actual token position
        if relative_pos < 0:  # This indicates last_10 analysis
            # relative_pos is -10 to -1, so actual position is len(tokens) + relative_pos
            actual_token_pos = len(token_strings) + relative_pos
            token_text = get_token_from_position(token_strings, actual_token_pos)
        else:
            # For other analysis types, use position directly
            token_text = get_token_from_position(token_strings, position)
        
        # Get top features for this token
        for rank, (feature_idx, activation) in enumerate(token_info['top_features'][:10], 1):
            # Find feature info - try external first, then pattern-specific, then global
            feature_desc = "No description available"
            feature_url = ""
            
            # First try external features (most comprehensive)
            if external_features and feature_idx in external_features:
                feature_info = external_features[feature_idx]
                feature_desc = feature_info.get('description', 'No description available')
                feature_url = feature_info.get('neuronpedia_url', '')
            elif feature_idx in feature_lookup:
                # Then try pattern-specific features
                feature_info = feature_lookup[feature_idx]
                feature_desc = feature_info.get('description', 'No description available')
                feature_url = feature_info.get('neuronpedia_url', '')
            else:
                # Finally look in global feature info
                for feature in all_feature_info:
                    if feature.get('feature_index') == feature_idx:
                        feature_desc = feature.get('description', 'No description available')
                        feature_url = feature.get('neuronpedia_url', '')
                        break
            
            # Show both actual and relative positions for clarity
            if relative_pos < 0:
                actual_pos = len(token_strings) + relative_pos
                position_display = f"{actual_pos} (rel: {relative_pos})"
            else:
                actual_pos = position
                position_display = f"{position}"
            
            all_data.append({
                'Token_Position': position_display,
                'Relative_Position': relative_pos,
                'Token_Text': token_text,
                'L0_Norm': l0_norm,
                'Feature_Rank': rank,
                'Feature_Index': feature_idx,
                'Activation': f"{activation:.2f}",
                'Feature_Description': feature_desc,
                'Neuronpedia_URL': feature_url
            })
    
    # Create DataFrame and display
    df = pd.DataFrame(all_data)
    
    # Group by token position for better display
    for pos in sorted(df['Token_Position'].unique(), key=lambda x: int(x.split()[0])):
        token_df = df[df['Token_Position'] == pos]
        token_text = token_df.iloc[0]['Token_Text']
        l0_norm = token_df.iloc[0]['L0_Norm']
        rel_pos = token_df.iloc[0]['Relative_Position']
        
        print(f"\nTOKEN {pos}: '{token_text}' [L0: {l0_norm}]")
        print("-" * 60)
        
        # Show top features for this token
        feature_table = token_df[['Feature_Rank', 'Feature_Index', 'Activation', 'Feature_Description']].copy()
        print(feature_table.to_string(index=False, max_colwidth=50))
        print()
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create feature interpretation tables from SAE analysis JSON')
    parser.add_argument('json_file', help='Path to the JSON file to analyze')
    parser.add_argument('--output', '-o', help='Output CSV file prefix (optional)')
    parser.add_argument('--pattern', '-p', help='Analyze specific pattern only')
    parser.add_argument('--features', '-f', help='Path to feature descriptions JSON file (optional)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_json_data(args.json_file)
    
    # Load external feature descriptions if provided
    external_features = {}
    if args.features:
        try:
            external_feature_data = load_json_data(args.features)
            # Convert to lookup dict with feature_index as int key
            external_features = {
                int(feature_idx): feature_info 
                for feature_idx, feature_info in external_feature_data.items()
            }
            print(f"Loaded {len(external_features)} external feature descriptions")
        except Exception as e:
            print(f"Warning: Could not load external features file: {e}")
    
    # Auto-detect feature descriptions file if not provided
    if not args.features and not external_features:
        json_path = Path(args.json_file)
        feature_files = list(json_path.parent.glob('feature_descriptions_*.json'))
        if feature_files:
            try:
                # Use most recent feature descriptions file
                latest_features_file = max(feature_files, key=lambda f: f.stat().st_mtime)
                external_feature_data = load_json_data(str(latest_features_file))
                external_features = {
                    int(feature_idx): feature_info 
                    for feature_idx, feature_info in external_feature_data.items()
                }
                print(f"Auto-detected and loaded {len(external_features)} feature descriptions from: {latest_features_file.name}")
            except Exception as e:
                print(f"Warning: Could not load auto-detected features file: {e}")
    
    print("SAE FEATURE INTERPRETATION ANALYSIS")
    print("=" * 80)
    
    # Display metadata
    metadata = data.get('analysis_metadata', {})
    print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"Total Patterns: {metadata.get('total_patterns', 'N/A')}")
    print(f"Total Unique Features: {metadata.get('total_unique_features', 'N/A')}")
    print(f"Target State: {metadata.get('target_state', 'N/A')}")
    
    sae_config = metadata.get('sae_config', {})
    print(f"SAE Release: {sae_config.get('release', 'N/A')}")
    print(f"SAE ID: {sae_config.get('sae_id', 'N/A')}")
    print(f"Layer: {sae_config.get('layer', 'N/A')}")
    
    # Get patterns and feature info
    patterns = data.get('patterns', {})
    feature_info = data.get('feature_info', [])
    
    print(f"\nFound {len(patterns)} patterns and {len(feature_info)} unique features")
    
    all_pattern_data = []
    
    # Process each pattern
    for pattern_name, pattern_data in patterns.items():
        if args.pattern and pattern_name != args.pattern:
            continue
            
        # Create table for this pattern
        pattern_df = create_pattern_table(pattern_name, pattern_data, feature_info, external_features)
        pattern_df['Pattern_Name'] = pattern_name
        all_pattern_data.append(pattern_df)
    
    # Save to CSV if requested
    if args.output and all_pattern_data:
        combined_df = pd.concat(all_pattern_data, ignore_index=True)
        output_file = f"{args.output}_feature_analysis.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")
        
        # Also save pattern-specific files
        for i, (pattern_name, df) in enumerate(zip(patterns.keys(), all_pattern_data)):
            if args.pattern and pattern_name != args.pattern:
                continue
            pattern_file = f"{args.output}_{pattern_name.replace(' ', '_').replace('&', 'and')}.csv"
            df.to_csv(pattern_file, index=False)
            print(f"Pattern '{pattern_name}' saved to: {pattern_file}")

if __name__ == "__main__":
    main()