#!/usr/bin/env python3
"""
Script to remove notes from cognitive patterns dataset files.
Notes are denoted with "(Note:" and need to be completely removed from the thought_pattern field.
"""

import csv
import json
import re
import os
from typing import Dict, Any

def clean_thought_pattern(text: str) -> str:
    """
    Remove notes from thought pattern text.
    Notes start with "(Note:" and end with the closing parenthesis.
    """
    if not text:
        return text
    
    # Remove notes that start with "(Note:" and continue until the closing parenthesis
    # This handles multi-line notes
    pattern = r'\s*\n*\s*\(Note:.*?\)\s*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # Clean up any extra whitespace or newlines
    cleaned = re.sub(r'\n+$', '', cleaned)  # Remove trailing newlines
    cleaned = re.sub(r'^\s*\n+', '', cleaned)  # Remove leading newlines
    cleaned = cleaned.strip()
    
    return cleaned

def clean_csv_file(input_path: str, output_path: str):
    """Clean notes from CSV file."""
    print(f"Cleaning CSV file: {input_path}")
    
    rows_cleaned = 0
    total_rows = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                total_rows += 1
                original_thought = row['thought_pattern']
                cleaned_thought = clean_thought_pattern(original_thought)
                
                if original_thought != cleaned_thought:
                    rows_cleaned += 1
                    print(f"Row {total_rows}: Cleaned note")
                
                row['thought_pattern'] = cleaned_thought
                writer.writerow(row)
    
    print(f"CSV cleaning complete: {rows_cleaned}/{total_rows} rows had notes removed")

def clean_jsonl_file(input_path: str, output_path: str):
    """Clean notes from JSONL file."""
    print(f"Cleaning JSONL file: {input_path}")
    
    rows_cleaned = 0
    total_rows = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                total_rows += 1
                data = json.loads(line.strip())
                
                original_thought = data['thought_pattern']
                cleaned_thought = clean_thought_pattern(original_thought)
                
                if original_thought != cleaned_thought:
                    rows_cleaned += 1
                    print(f"Row {total_rows}: Cleaned note")
                
                data['thought_pattern'] = cleaned_thought
                outfile.write(json.dumps(data) + '\n')
    
    print(f"JSONL cleaning complete: {rows_cleaned}/{total_rows} rows had notes removed")

def main():
    """Main function to clean both files."""
    base_path = "/home/koalacrown/Desktop/Code/Projects/turnaround/turn_point/data/depressed_inductions"
    
    # Clean CSV file
    csv_input = os.path.join(base_path, "cognitive_patterns_dataset.csv")
    csv_output = os.path.join(base_path, "cognitive_patterns_dataset_cleaned.csv")
    clean_csv_file(csv_input, csv_output)
    
    # Clean JSONL file
    jsonl_input = os.path.join(base_path, "cognitive_patterns_dataset.jsonl")
    jsonl_output = os.path.join(base_path, "cognitive_patterns_dataset_cleaned.jsonl")
    clean_jsonl_file(jsonl_input, jsonl_output)
    
    print("\nCleaning complete!")
    print(f"Original files preserved, cleaned versions saved as:")
    print(f"  - {csv_output}")
    print(f"  - {jsonl_output}")

if __name__ == "__main__":
    main()
