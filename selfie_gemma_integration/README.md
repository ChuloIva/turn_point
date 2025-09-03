# Gemma SelfIE Integration

This directory contains an integration between your cognitive pattern analysis pipeline and the SelfIE (Self-Interpretation of Embeddings) library. It allows you to generate natural language interpretations of pre-captured Gemma-2-2b activations.

## Overview

SelfIE enables Large Language Models to interpret their own internal representations in natural language. This integration adapts SelfIE to work with your pre-captured activations from the cognitive pattern analysis, providing human-readable explanations of what neural patterns represent.

## Key Features

- **Load Pre-captured Activations**: Work with your existing activation cache files
- **Multi-layer Analysis**: Interpret activations from different transformer layers
- **Position-specific Analysis**: Analyze specific token positions or averaged representations
- **Custom Interpretation Templates**: Use different prompts for various analysis goals
- **Cognitive Pattern Focus**: Specifically designed for cognitive pattern analysis
- **Multiple Output Formats**: JSON data and Markdown reports

## Installation

The integration requires SelfIE to be available in your project:

```bash
# SelfIE should already be cloned in your project directory
cd ../selfie
pip install -e .
```

Additional requirements:
- transformers
- torch
- numpy
- pandas (optional, for CSV output)

## Quick Start

```python
from gemma_selfie_adapter import GemmaSelfieAdapter

# Initialize adapter
adapter = GemmaSelfieAdapter()

# Run complete analysis
results_file = adapter.run_full_analysis(
    layer=17,           # Analyze layer 17
    max_patterns=5,     # Process first 5 patterns
    position=-1         # Use last token position
)

print(f"Results saved to: {results_file}")
```

## File Structure

```
selfie_gemma_integration/
├── gemma_selfie_adapter.py    # Main adapter class
├── config.py                  # Configuration settings
├── example_usage.py           # Usage examples
├── README.md                  # This file
└── outputs/                   # Generated results directory
```

## Core Components

### GemmaSelfieAdapter Class

The main class that handles the integration:

- `load_cached_activations()` - Load your pre-captured activation files
- `extract_cognitive_patterns()` - Extract cognitive pattern metadata
- `load_model_and_tokenizer()` - Load Gemma model for interpretation
- `prepare_activations_for_selfie()` - Convert activations to SelfIE format
- `interpret_activations_with_selfie()` - Generate natural language interpretations
- `save_interpretation_results()` - Save results to JSON and Markdown

### Configuration

Customizable settings in `config.py`:

- **Model Configuration**: Gemma model settings
- **SelfIE Configuration**: Batch sizes, token limits
- **Analysis Configuration**: Default layers, positions
- **Interpretation Templates**: Different prompts for various analysis types

## Usage Examples

### Basic Analysis

```python
adapter = GemmaSelfieAdapter()

# Load and analyze
cache_data = adapter.load_cached_activations()
adapter.load_model_and_tokenizer()

results = adapter.run_full_analysis(layer=17, max_patterns=3)
```

### Custom Interpretation Template

```python
from config import INTERPRETATION_TEMPLATES

# Use emotional state template
template = INTERPRETATION_TEMPLATES["emotional_state"]

prepared_data = adapter.prepare_activations_for_selfie(
    cache_key="your_cache_key",
    layer=17
)

results = adapter.interpret_activations_with_selfie(
    prepared_data=prepared_data,
    interpretation_template=template
)
```

### Multi-layer Analysis

```python
layers = [10, 15, 17, 20]
all_results = []

for layer in layers:
    prepared_data = adapter.prepare_activations_for_selfie(
        cache_key="your_cache_key",
        layer=layer
    )
    
    results = adapter.interpret_activations_with_selfie(
        prepared_data=prepared_data
    )
    
    all_results.extend(results)

adapter.save_interpretation_results(all_results, "multi_layer_analysis")
```

### Position Comparison

```python
positions = {"last": -1, "second_last": -2, "average": "mean"}

for pos_name, position in positions.items():
    results = adapter.interpret_activations_with_selfie(
        prepared_data=prepared_data,
        position=position
    )
    # Process results...
```

## Available Interpretation Templates

1. **cognitive_pattern** - For general cognitive pattern analysis
2. **emotional_state** - For emotional/affective analysis  
3. **general_concept** - For conceptual understanding
4. **decision_making** - For decision-making processes

## Output Format

### JSON Output
```json
{
  "metadata": {
    "timestamp": "2024-09-03T16:30:00",
    "total_interpretations": 5,
    "model_used": "google/gemma-2-2b-it"
  },
  "interpretations": [
    {
      "pattern_name": "Depression Pattern",
      "pattern_text": "I always mess everything up...",
      "interpretation": "negative self-evaluation and pessimistic thinking",
      "layer": 17,
      "position": -1
    }
  ]
}
```

### Markdown Report
Human-readable report with:
- Pattern names and original text
- Layer and position analyzed
- SelfIE interpretations
- Metadata

## Integration with Your Pipeline

This integration is designed to work seamlessly with your existing cognitive pattern analysis:

1. **Uses Existing Activations**: Reads from your `/activations/` directory
2. **Cognitive Pattern Aware**: Understands your enriched metadata format
3. **Layer 17 Focus**: Defaults to layer 17 (same as your SAE analysis)
4. **Compatible Output**: Generates analysis reports similar to your existing pipeline

## Advanced Usage

### Batch Processing Multiple Cache Files

```python
adapter = GemmaSelfieAdapter()
cache_data = adapter.load_cached_activations()  # Loads all cache files

for cache_key in cache_data.keys():
    results = adapter.run_full_analysis(
        cache_key=cache_key,
        max_patterns=3
    )
```

### Custom Vector Analysis

```python
# If you have specific activation vectors
from selfie.interpret import interpret_vectors

vectors = [your_activation_tensor]  # List of tensors
interpretations = interpret_vectors(
    vecs=vectors,
    model=adapter.model,
    interpretation_prompt=your_prompt,
    tokenizer=adapter.tokenizer
)
```

## Troubleshooting

### Common Issues

1. **SelfIE Import Error**: Make sure SelfIE is installed (`pip install -e .` in selfie directory)
2. **Model Loading Issues**: Ensure sufficient GPU memory or use CPU
3. **No Activations Found**: Check that activation cache files exist in `/activations/`
4. **Layer Not Found**: Verify the layer exists in your cached activations

### Memory Management

- Use smaller batch sizes for GPU memory constraints
- Set `torch_dtype="float32"` for CPU usage
- Limit `max_patterns` for large datasets

## Performance Notes

- **GPU Recommended**: SelfIE interpretation benefits from GPU acceleration
- **Batch Size**: Start with small batch sizes (1-2) and increase as memory allows  
- **Token Limits**: Higher `max_new_tokens` gives more detailed interpretations but takes longer

## Example Output

```
🔮 Generating SelfIE interpretations...
   Processing 3 patterns
   Activation tensor shape: torch.Size([3, 10, 2304])
   Selected vectors shape: torch.Size([3, 2304])
   Generating interpretations with SelfIE...
   ✅ Generated 3 interpretations

📊 Analysis Complete!
   Patterns analyzed: 3
   Layer analyzed: 17
   Results saved to: outputs/selfie_interpretations_20240903_163000.json

🔍 Sample interpretations:
   1. Depression Pattern:
      "negative self-evaluation and pessimistic thinking patterns..."
   2. Anxiety Pattern: 
      "worry and future-oriented threat assessment..."
   3. Rumination Pattern:
      "repetitive negative thought cycles and self-focused attention..."
```

## Next Steps

1. Run `python example_usage.py` to see all examples
2. Experiment with different layers and positions
3. Try custom interpretation templates for your specific use cases
4. Integrate results with your existing SAE analysis
5. Compare SelfIE interpretations with Neuronpedia feature descriptions

## Contributing

To extend this integration:
- Add new interpretation templates in `config.py`
- Implement additional output formats
- Add support for other model architectures
- Enhance error handling and logging