I'll analyze the files to understand the Neuronpedia integration capabilities and create a comprehensive plan for fetching neuron descriptions in the SAE test.

[1 tool called]

Now let me examine the tutorial to see how they fetch explanations from Neuronpedia:

[2 tools called]

Let me search for the specific section about downloading explanations:

[2 tools called]

Let me find the exact lines with the API calls:

[1 tool called]

Now let me look at the current SAE test to understand its structure:

[1 tool called]

Now let me check what SAE configuration is being used:

[1 tool called]

Based on my analysis of the files, here's a comprehensive plan for integrating Neuronpedia feature description fetching into the SAE test:

# Comprehensive Plan: Neuronpedia Feature Description Integration

## Overview
The goal is to enhance `test_single_pattern_sae.py` to fetch and display human-readable descriptions of the most important SAE features identified during cognitive pattern analysis.

## Available Neuronpedia Integration Capabilities

### From `neuronpedia_integration.py`:
1. **`get_neuronpedia_feature()`** - Fetches individual feature data via API
2. **`NeuronpediaFeature` class** - Represents a feature with description and autointerp explanation
3. **URL generation functions** - Create dashboard links

### From `tutorial_2_0.ipynb`:
1. **Bulk explanation export API**: `https://www.neuronpedia.org/api/explanation/export?modelId=<model>&saeId=<sae_id>`
2. **Individual feature API**: `https://www.neuronpedia.org/api/feature/{modelId}/{layer}/{index}`
3. **Pandas DataFrame integration** for explanation search and filtering

## Implementation Plan

### Phase 1: Core Integration Infrastructure

#### 1.1 Add Neuronpedia API Client Class
Create a new class `NeuronpediaClient` in `test_single_pattern_sae.py`:

```python
class NeuronpediaClient:
    def __init__(self, base_url: str = "https://www.neuronpedia.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def get_feature_explanation(self, model_id: str, layer: str, feature_idx: int) -> Dict[str, Any]:
        """Fetch individual feature explanation from Neuronpedia API"""
        
    def get_bulk_explanations(self, model_id: str, sae_id: str) -> pd.DataFrame:
        """Fetch all explanations for an SAE as DataFrame"""
        
    def batch_get_features(self, model_id: str, layer: str, feature_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch multiple features with rate limiting and error handling"""
```

#### 1.2 SAE-to-Neuronpedia ID Mapping
Add method to convert SAE configuration to Neuronpedia API parameters:

```python
def get_neuronpedia_identifiers(self) -> Tuple[str, str]:
    """Convert SAE config to Neuronpedia model_id and sae_id format"""
    # Handle different SAE releases and convert to Neuronpedia format
    # Example: "gemma-scope-2b-pt-res-canonical" -> "gemma-2b"
    # "layer_20/width_16k/canonical" -> "20-res-16k"
```

### Phase 2: Feature Description Fetching

#### 2.1 Enhance `analyze_feature_differences()` 
Modify to include description fetching:

```python
def analyze_feature_differences_with_descriptions(self, sae_results: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced feature analysis with Neuronpedia descriptions"""
    # Existing differential analysis code...
    
    # New: Collect all unique feature indices
    all_feature_indices = set()
    for comparison_name, features in analysis.items():
        for feature in features[:10]:  # Top 10 per comparison
            all_feature_indices.add(feature['feature_idx'])
    
    # Fetch descriptions in batch
    descriptions = self.fetch_feature_descriptions(list(all_feature_indices))
    
    # Enrich analysis with descriptions
    for comparison_name, features in analysis.items():
        for feature in features:
            feature_idx = feature['feature_idx']
            feature['description'] = descriptions.get(feature_idx, {})
    
    return analysis
```

#### 2.2 New Method: `fetch_feature_descriptions()`

```python
def fetch_feature_descriptions(self, feature_indices: List[int]) -> Dict[int, Dict[str, Any]]:
    """Fetch descriptions for a list of feature indices"""
    print(f"\n📖 Fetching descriptions for {len(feature_indices)} features...")
    
    # Get Neuronpedia identifiers
    model_id, sae_id = self.get_neuronpedia_identifiers()
    
    # Initialize client
    client = NeuronpediaClient()
    
    descriptions = {}
    failed_fetches = []
    
    # Batch fetch with rate limiting
    for i, feature_idx in enumerate(feature_indices):
        try:
            print(f"   Fetching {i+1}/{len(feature_indices)}: Feature {feature_idx}")
            
            feature_data = client.get_feature_explanation(model_id, sae_id, feature_idx)
            descriptions[feature_idx] = {
                'description': feature_data.get('description', 'No description available'),
                'autointerp_explanation': feature_data.get('explanation', ''),
                'autointerp_score': feature_data.get('explanationScore', 0.0),
                'neuronpedia_url': f"https://neuronpedia.org/{model_id}/{sae_id}/{feature_idx}"
            }
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"     ⚠️ Failed to fetch Feature {feature_idx}: {e}")
            failed_fetches.append(feature_idx)
            descriptions[feature_idx] = {
                'description': f'Failed to fetch: {e}',
                'autointerp_explanation': '',
                'autointerp_score': 0.0,
                'neuronpedia_url': f"https://neuronpedia.org/{model_id}/{sae_id}/{feature_idx}"
            }
    
    if failed_fetches:
        print(f"   ⚠️ Failed to fetch {len(failed_fetches)} features: {failed_fetches}")
    
    print(f"✅ Successfully fetched descriptions for {len(descriptions) - len(failed_fetches)} features")
    return descriptions
```

### Phase 3: Enhanced Reporting and Output

#### 3.1 Update `save_results()` Method
Enhance to include descriptions in all output formats:

```python
def save_results(self, sae_results: Dict[str, Any], differential_analysis: Dict[str, Any],
                neuronpedia_urls: Dict[str, List[str]], descriptions: Dict[int, Dict[str, Any]],
                release: str, sae_id: str) -> str:
    # Existing code...
    
    # Add descriptions to results_data
    results_data['feature_descriptions'] = descriptions
    
    # Enhanced differential analysis with descriptions
    results_data['differential_analysis_with_descriptions'] = differential_analysis
```

#### 3.2 Update `_create_report()` Method
Enhance markdown report to include feature descriptions:

```python
def _create_report(self, report_file: Path, results_data: Dict[str, Any]) -> None:
    # Existing report sections...
    
    # New: Feature Descriptions Section
    f.write("## Top Feature Descriptions\n")
    for transition, features in results_data['differential_analysis_with_descriptions'].items():
        f.write(f"### {transition.replace('_', ' ').title()}\n")
        for feature in features[:5]:
            feature_idx = feature['feature_idx']
            desc_data = feature.get('description', {})
            
            f.write(f"#### Feature {feature_idx} ({feature['direction']} {feature['abs_diff']:.4f})\n")
            f.write(f"**Description:** {desc_data.get('description', 'No description')}\n\n")
            
            if desc_data.get('autointerp_explanation'):
                f.write(f"**AutoInterp:** {desc_data['autointerp_explanation']}\n")
                f.write(f"**Score:** {desc_data.get('autointerp_score', 0.0):.2f}\n\n")
            
            f.write(f"**Dashboard:** [{desc_data.get('neuronpedia_url', 'N/A')}]({desc_data.get('neuronpedia_url', '#')})\n\n")
```

#### 3.3 Add Feature Summary Table
Create a CSV export of top features with descriptions:

```python
def create_feature_summary_csv(self, differential_analysis: Dict[str, Any], 
                              descriptions: Dict[int, Dict[str, Any]]) -> str:
    """Create CSV summary of top differential features with descriptions"""
    
    summary_data = []
    for transition, features in differential_analysis.items():
        for feature in features[:10]:  # Top 10 per transition
            feature_idx = feature['feature_idx']
            desc_data = descriptions.get(feature_idx, {})
            
            summary_data.append({
                'transition_type': transition,
                'feature_idx': feature_idx,
                'rank': feature['rank'],
                'direction': feature['direction'],
                'abs_diff': feature['abs_diff'],
                'raw_diff': feature['raw_diff'],
                'description': desc_data.get('description', ''),
                'autointerp_explanation': desc_data.get('autointerp_explanation', ''),
                'autointerp_score': desc_data.get('autointerp_score', 0.0),
                'neuronpedia_url': desc_data.get('neuronpedia_url', '')
            })
    
    # Save as CSV
    df = pd.DataFrame(summary_data)
    csv_file = self.output_dir / f"feature_summary_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    return str(csv_file)
```

### Phase 4: Error Handling and Robustness

#### 4.1 Fallback Mechanisms
```python
def get_neuronpedia_identifiers(self) -> Tuple[str, str]:
    """Convert SAE config to Neuronpedia identifiers with fallbacks"""
    
    # Primary mapping
    mapping_attempts = [
        # Try direct neuronpedia_id from SAE config
        lambda: (self.sae.cfg.neuronpedia_id.split('/')[0], self.sae.cfg.neuronpedia_id.split('/')[1]),
        
        # Try model name extraction
        lambda: self._extract_from_model_name(),
        
        # Try release name parsing
        lambda: self._extract_from_release_name(),
        
        # Manual mapping for known cases
        lambda: self._manual_mapping()
    ]
    
    for attempt in mapping_attempts:
        try:
            model_id, sae_id = attempt()
            if model_id and sae_id:
                return model_id, sae_id
        except Exception:
            continue
    
    raise ValueError("Could not determine Neuronpedia identifiers")
```

#### 4.2 Rate Limiting and Retry Logic
```python
def fetch_with_retry(self, url: str, max_retries: int = 3, delay: float = 1.0) -> Dict[str, Any]:
    """Fetch with exponential backoff retry"""
    
    for attempt in range(max_retries):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))
```

### Phase 5: Integration into Main Flow

#### 5.1 Update `main()` Function
```python
def main():
    # Existing steps 1-5...
    
    # Step 6: Enhanced analysis with descriptions
    differential_analysis = test.analyze_feature_differences_with_descriptions(sae_results)
    
    # Step 7: Extract descriptions for separate handling
    descriptions = test.extract_descriptions_from_analysis(differential_analysis)
    
    # Step 8: Generate enhanced Neuronpedia dashboard
    neuronpedia_urls = test.create_neuronpedia_dashboard(
        differential_analysis, release, sae_id
    )
    
    # Step 9: Save enhanced results
    report_path = test.save_results(
        sae_results, differential_analysis, neuronpedia_urls, descriptions,
        release, sae_id
    )
    
    # Step 10: Create feature summary CSV
    csv_path = test.create_feature_summary_csv(differential_analysis, descriptions)
    
    print(f"\n🎉 Enhanced SAE analysis completed!")
    print(f"📋 Report: {report_path}")
    print(f"📊 Feature CSV: {csv_path}")
```

## Expected Output Enhancements

### Console Output
```
📖 Fetching descriptions for 25 features...
   Fetching 1/25: Feature 1234
   Fetching 2/25: Feature 5678
   ...
✅ Successfully fetched descriptions for 23 features

🔍 Top Features with Descriptions:
   negative_to_positive:
     1. Feature 1234: ↑ 0.8542 - "Religious and spiritual concepts"
     2. Feature 5678: ↓ 0.7231 - "Negative emotional expressions"
```

### Enhanced Markdown Report
```markdown
## Top Feature Descriptions

### Negative To Positive
#### Feature 1234 (↑ 0.8542)
**Description:** Religious and spiritual concepts, particularly references to God, prayer, and faith-based thinking.

**AutoInterp:** This feature activates on tokens related to religious concepts, spiritual practices, and faith-based reasoning.
**Score:** 0.87

**Dashboard:** [https://neuronpedia.org/gemma-2b/20-res-16k/1234](https://neuronpedia.org/gemma-2b/20-res-16k/1234)
```

### CSV Export Columns
- `transition_type`, `feature_idx`, `rank`, `direction`, `abs_diff`, `raw_diff`
- `description`, `autointerp_explanation`, `autointerp_score`, `neuronpedia_url`

## Implementation Considerations

1. **API Rate Limits**: Implement delays and batch processing
2. **Error Handling**: Graceful degradation when descriptions aren't available
3. **Caching**: Store fetched descriptions to avoid re-fetching
4. **Model Compatibility**: Handle different SAE releases and model formats
5. **Async Processing**: Consider `asyncio` for faster batch fetching

This plan provides a comprehensive framework for another LLM to implement feature description fetching that enhances the cognitive pattern analysis with human-interpretable explanations from Neuronpedia.