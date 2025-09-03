#!/usr/bin/env python3
"""
Script to interpret features that don't have explanations in the feature summary JSON.
Uses the Neuronpedia Auto-Interp API to generate feature interpretations.
"""

import json
import requests
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NeuronpediaClient:
    """Client for fetching and generating feature explanations from Neuronpedia API."""
    
    def __init__(self, base_url: str = "https://www.neuronpedia.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Add API key authentication if available
        api_key = os.getenv("NEURONPEDIA_KEY")
        if api_key:
            self.session.headers.update({"x-api-key": api_key})
            print(f"   🔑 Using Neuronpedia API key authentication")
        else:
            print(f"   ⚠️ No Neuronpedia API key found, using unauthenticated requests")
    
    def generate_auto_interp_explanation(
        self,
        feature_idx: int,
        model_id: str = "gemma-2-2b",
        layer: str = "17-gemmascope-res-65k", 
        explanation_type: str = "oai_token-act-pair",
        explanation_model: str = "claude-3-5-sonnet-20250219"
    ) -> Dict[str, Any]:
        """
        Generate feature interpretation using Neuronpedia Auto-Interp API.
        
        Args:
            feature_idx: The feature index
            model_id: Model ID (default: gemma-2-2b)
            layer: Layer specification (default: 17-gemmascope-res-65k)
            explanation_type: Type of explanation (default: oai_token-act-pair)
            explanation_model: Model to use for explanation (default: gpt-4-turbo)
        
        Returns:
            Dictionary with interpretation data or error info
        """
        url = f"{self.base_url}/api/explanation/generate"
        
        payload = {
            "modelId": model_id,
            "layer": layer,
            "index": feature_idx,
            "explanationType": explanation_type,
            "explanationModelName": explanation_model
        }
        
        try:
            print(f"  Making Auto-Interp API request for feature {feature_idx}...")
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'explanation': data.get('explanation', ''),
                    'raw_response': data
                }
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('error', 'Unknown error')}"
                except:
                    error_msg += f": {response.text}"
                
                print(f"  Auto-Interp API request failed for feature {feature_idx}: {error_msg}")
                return {
                    'success': False, 
                    'error': error_msg,
                    'status_code': response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            print(f"  Request error for feature {feature_idx}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_feature_explanation(self, model_id: str, layer: str, feature_idx: int) -> Dict[str, Any]:
        """Fetch existing feature explanation from Neuronpedia API as fallback."""
        url = f"{self.base_url}/api/feature/{model_id}/{layer}/{feature_idx}"
        try:
            response = self.fetch_with_retry(url)
            return response
        except Exception as e:
            print(f"  Failed to fetch existing explanation for feature {feature_idx}: {e}")
            return {}
    
    def fetch_with_retry(self, url: str, max_retries: int = 3, delay: float = 1.0) -> Dict[str, Any]:
        """Fetch with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay * (2 ** attempt))

def load_feature_summary(file_path: str) -> List[Dict[str, Any]]:
    """Load the feature summary JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_features_needing_interpretation(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter features that need interpretation (empty description/autointerp_explanation)."""
    return [
        feature for feature in features
        if not feature['description'].strip() or not feature['autointerp_explanation'].strip()
    ]

def update_feature_with_interpretation(feature: Dict[str, Any], interpretation_data: Dict[str, Any], fallback_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Update a feature with new interpretation data, with fallback to existing data."""
    updated_feature = feature.copy()
    
    if interpretation_data['success'] and interpretation_data.get('explanation'):
        explanation = interpretation_data['explanation']
        updated_feature['description'] = explanation
        updated_feature['autointerp_explanation'] = explanation
        updated_feature['autointerp_score'] = 1.0  # Mark as successfully interpreted
        print(f"  ✓ Generated explanation: {explanation[:100]}...")
    elif fallback_data:
        # Try to use existing explanation from Neuronpedia
        existing_explanation = ""
        if 'explanations' in fallback_data and fallback_data['explanations']:
            first_explanation = fallback_data['explanations'][0]
            existing_explanation = first_explanation.get('description', '')
        elif 'description' in fallback_data:
            existing_explanation = fallback_data['description']
        
        if existing_explanation:
            updated_feature['description'] = existing_explanation
            updated_feature['autointerp_explanation'] = existing_explanation  
            updated_feature['autointerp_score'] = 0.5  # Mark as existing explanation
            print(f"  ↻ Using existing explanation: {existing_explanation[:100]}...")
        else:
            # Generate from activating tokens
            pos_tokens = fallback_data.get('pos_str', [])
            if pos_tokens:
                description = f"Activates on: {', '.join(pos_tokens[:5])}"
                updated_feature['description'] = description
                updated_feature['autointerp_explanation'] = description
                updated_feature['autointerp_score'] = 0.3  # Mark as token-based
                print(f"  ↻ Generated from tokens: {description[:100]}...")
            else:
                error_msg = interpretation_data.get('error', 'Unknown error')
                updated_feature['description'] = f"Auto-interp failed: {error_msg}"
                updated_feature['autointerp_explanation'] = f"Auto-interp failed: {error_msg}"
                print(f"  ✗ Failed to generate explanation: {error_msg}")
    else:
        error_msg = interpretation_data.get('error', 'Unknown error')
        updated_feature['description'] = f"Auto-interp failed: {error_msg}"
        updated_feature['autointerp_explanation'] = f"Auto-interp failed: {error_msg}"
        print(f"  ✗ Failed to generate explanation: {error_msg}")
    
    return updated_feature

def main():
    """Main function to process features and add interpretations."""
    input_file = "/Users/ivanculo/Desktop/Projects/turn_point/sae_test_outputs/feature_summary_20250902_030109.json"
    output_file = "/Users/ivanculo/Desktop/Projects/turn_point/sae_test_outputs/feature_summary_with_interpretations.json"
    
    print("=" * 60)
    print("NEURONPEDIA AUTO-INTERP FEATURE INTERPRETER")
    print("=" * 60)
    
    print(f"\nLoading feature summary from: {input_file}")
    features = load_feature_summary(input_file)
    
    print(f"Total features loaded: {len(features)}")
    
    # Find features needing interpretation
    features_to_interpret = get_features_needing_interpretation(features)
    print(f"Features needing interpretation: {len(features_to_interpret)}")
    
    if not features_to_interpret:
        print("No features need interpretation!")
        return
    
    # Show which features will be processed
    print("\nFeatures to be interpreted:")
    unique_features = set()
    for feature in features_to_interpret:
        if feature['feature_idx'] not in unique_features:
            print(f"  - Feature {feature['feature_idx']} (rank {feature['rank']}, {feature['state']}) - activation: {feature['activation_value']:.2f}")
            unique_features.add(feature['feature_idx'])
    
    print(f"\nUnique features to process: {len(unique_features)}")
    
    # Ask for confirmation
    if input(f"\nProceed with interpreting {len(unique_features)} unique features? (y/N): ").lower() != 'y':
        print("Cancelled.")
        return
    
    # Initialize Neuronpedia client
    client = NeuronpediaClient()
    
    # Process each unique feature first to get interpretations
    print(f"\nProcessing unique features...")
    print("-" * 40)
    
    feature_interpretations = {}
    successful_interpretations = 0
    
    for i, feature_idx in enumerate(unique_features):
        print(f"\nProcessing feature {feature_idx} ({i+1}/{len(unique_features)})...")
        
        # Try Auto-Interp API first
        interpretation_data = client.generate_auto_interp_explanation(feature_idx)
        
        # If Auto-Interp fails, try to get existing explanation as fallback
        fallback_data = None
        if not interpretation_data['success']:
            print(f"  Auto-Interp failed, trying existing explanation...")
            fallback_data = client.get_feature_explanation("gemma-2-2b", "17-gemmascope-res-65k", feature_idx)
        
        feature_interpretations[feature_idx] = {
            'interpretation': interpretation_data,
            'fallback': fallback_data
        }
        
        if interpretation_data['success']:
            successful_interpretations += 1
        
        # Be respectful to the API - add a delay between requests
        time.sleep(1.5)
    
    # Now update all features with the interpretations
    print(f"\nUpdating feature records...")
    updated_features = []
    
    for feature in features:
        if not feature['description'].strip():
            feature_idx = feature['feature_idx']
            if feature_idx in feature_interpretations:
                interp_data = feature_interpretations[feature_idx]['interpretation']
                fallback_data = feature_interpretations[feature_idx]['fallback']
                updated_feature = update_feature_with_interpretation(feature, interp_data, fallback_data)
                updated_features.append(updated_feature)
            else:
                # This shouldn't happen, but just in case
                updated_features.append(feature)
        else:
            # Feature already has interpretation
            updated_features.append(feature)
    
    # Save updated features
    print(f"\nSaving updated features to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(updated_features, f, indent=2)
    
    print("\n" + "=" * 60)
    print("FEATURE INTERPRETATION COMPLETE!")
    print("=" * 60)
    print(f"Successfully interpreted: {successful_interpretations}/{len(unique_features)} unique features")
    print(f"Results saved to: {output_file}")
    
    # Show summary of interpretation sources
    auto_interp_count = sum(1 for f in updated_features if f.get('autointerp_score', 0) == 1.0)
    existing_count = sum(1 for f in updated_features if f.get('autointerp_score', 0) == 0.5)
    token_count = sum(1 for f in updated_features if f.get('autointerp_score', 0) == 0.3)
    
    print(f"\nInterpretation sources:")
    print(f"  Auto-Interp generated: {auto_interp_count}")
    print(f"  Existing explanations: {existing_count}")
    print(f"  Token-based: {token_count}")
    print(f"  Failed: {len(features_to_interpret) - auto_interp_count - existing_count - token_count}")

if __name__ == "__main__":
    main()