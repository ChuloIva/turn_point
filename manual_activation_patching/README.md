# Manual Activation Patching

This directory contains a manual implementation of activation patching using TransformerLens, following the approach outlined in `plan.md`. The implementation allows you to inject activations from positive thought patterns into negative/neutral prompts to observe their effect on text generation.

## Files

- `activation_patcher.py`: Main implementation with the `ActivationPatcher` class
- `experiment_notebook.ipynb`: Basic activation patching experiments with easy model switching
- `batch_activation_notebook.ipynb`: Advanced batch activation patching experiments
- `README.md`: This documentation file

## How It Works

1. **Load Model**: Uses `HookedTransformer` from TransformerLens to load GPT-2 small
2. **Get Clean Cache**: Runs the model on positive thought patterns to extract activations
3. **Create Hooks**: Defines custom hook functions to patch activations during forward pass
4. **Patch & Generate**: Injects positive activations into negative prompts and generates text

## Key Features

- **Multi-Model Support**: Easy switching between different language models (GPT-2, GPT-Neo, Pythia, OPT, etc.)
- **Flexible Patching**: Patch at any layer and any number of token positions
- **Batch Processing**: Extract and aggregate activations from multiple texts
- **Dataset Integration**: Uses the positive_patterns.jsonl dataset
- **Multiple Experiments**: Compare different layers, patterns, and configurations
- **Longer Sequences**: Generates 50-70 tokens to observe extended effects

## Usage

### Command Line
```python
python activation_patcher.py
```

### Jupyter Notebooks

#### Basic Experiments (`experiment_notebook.ipynb`)
- Easy model switching (just change MODEL_NAME and re-run)
- Different cognitive patterns
- Layer comparisons
- Varying numbers of patch positions
- Custom text inputs
- Baseline comparisons

#### Batch Experiments (`batch_activation_notebook.ipynb`)
- Aggregate activations from multiple texts
- Different aggregation methods (mean, median, max, random)
- Batch size comparisons
- Target word vs. full text extraction
- Advanced patching strategies

## Example

```python
from activation_patcher import ActivationPatcher

# Initialize
patcher = ActivationPatcher("gpt2-small")

# Define texts
clean_text = "I'm taking steps to acknowledge my feelings and find solutions."
corrupted_text = "I feel overwhelmed and don't know how to"
target_words = ["acknowledge", "feelings", "solutions"]

# Patch and generate
predicted_token, generated_text = patcher.patch_and_generate(
    clean_text=clean_text,
    corrupted_text=corrupted_text,
    target_words=target_words,
    max_new_tokens=60
)

print(generated_text)
```

## Technical Details

- **Model**: GPT-2 small (117M parameters)
- **Patching Location**: Residual stream at specified layers (default: last layer)
- **Patch Method**: Direct activation replacement at placeholder token positions
- **Generation**: Temperature sampling with configurable length (50-70 tokens)

## Experiments

The notebook includes several pre-configured experiments:
1. Basic activation patching with single pattern
2. Layer-by-layer comparison
3. Multiple cognitive patterns comparison
4. Different numbers of patch positions
5. Baseline comparison (no patching)
6. Custom pattern testing

## Dependencies

- PyTorch
- TransformerLens (included in third_party/)
- JSON (for dataset loading)
- Jupyter (for notebook experiments)

## Dataset Format

The positive_patterns.jsonl dataset contains cognitive patterns with:
- `positive_thought_pattern`: The clean text to extract activations from
- `cognitive_pattern_name`: Name of the cognitive pattern
- `reference_negative_example`: Example of negative thought pattern
- Additional metadata

## Next Steps

- Try different activation types (attention, MLP)
- Implement multi-layer patching
- Add quantitative evaluation metrics
- Experiment with larger models
- Test with longer generation sequences