# Psychology SelfIE Implementation

This directory contains an implementation of the SelfIE (Self-Interpretation of Embeddings) technique adapted for psychological cognitive pattern analysis.

## Overview

Unlike activation patching which modifies model behavior, SelfIE reveals what the model "thinks" its internal representations mean by having the model describe them in natural language. This provides interpretable insights into how the model processes different cognitive patterns.

## Files

- `selfie_patcher.py` - Main wrapper class providing SelfIE functionality with a similar interface to ActivationPatcher
- `selfie_experiment_notebook.ipynb` - Jupyter notebook with experiments similar to the manual activation patching workflow
- `requirements.txt` - Dependencies for the SelfIE environment (requires transformers==4.34.0)
- `README.md` - This file

## Setup

1. **Create a separate environment** (SelfIE requires transformers==4.34.0):
```bash
python3 -m venv selfie_psych_env
source selfie_psych_env/bin/activate
pip install -r requirements.txt
```

2. **Install SelfIE library**:
```bash
cd ../third_party/selfie
pip install -e .
```

3. **Use a LLaMA-compatible model** - SelfIE currently works best with LLaMA models:
   - `meta-llama/Llama-2-7b-chat-hf`
   - `meta-llama/Llama-3.1-8B-Instruct`
   - `huggyllama/llama-7b`

## Key Differences from Activation Patching

| Aspect | Activation Patching | SelfIE |
|--------|-------------------|--------|
| **Approach** | Modifies model behavior | Interprets model representations |
| **Output** | Generated text with patched behavior | Natural language descriptions of internal states |
| **Interpretability** | Indirect (observe behavior changes) | Direct (explicit descriptions) |
| **Model Requirements** | Works with most transformer models | Optimized for LLaMA models |
| **Use Case** | Steering model behavior | Understanding model cognition |

## Quick Start

```python
from selfie_patcher import SelfIEPatcher, TokenSelectionStrategy

# Initialize with LLaMA model
patcher = SelfIEPatcher("meta-llama/Llama-2-7b-chat-hf")

# Interpret cognitive patterns
results = patcher.interpret_text(
    text="I keep telling myself I'm not good enough.",
    layers_to_interpret=[-1, -2],
    interpretation_template="cognitive_pattern",
    max_new_tokens=40
)

print(results)
```

## Experiments Available

The notebook includes these experiments modeled after the activation patching workflow:

1. **Basic SelfIE Interpretation** - Interpret internal representations of cognitive patterns
2. **Positive vs Negative Comparison** - Compare how the model interprets different pattern types
3. **Multi-Layer Analysis** - Analyze how interpretations change across model layers
4. **Template Comparison** - Test different interpretation templates

## Future Features (Placeholders)

The implementation includes placeholders for advanced SelfIE features from the research paper:

- **Supervised Control** - Edit concepts in hidden representations using gradient-based methods
- **Reinforcement Control** - Use RLHF to remove harmful knowledge from embeddings
- **Batch Processing** - Efficiently process multiple cognitive patterns
- **Visualization** - Interactive plots of interpretation results
- **Export Tools** - Save results in various formats (JSON, CSV, HTML)

## Implementation Status

✅ **Completed:**
- Basic SelfIE wrapper class
- Experiment notebook with 4 core experiments
- Environment setup and requirements
- Interface compatibility with ActivationPatcher

🚧 **In Progress:**
- Advanced control features (supervised/reinforcement)
- Visualization tools
- Batch processing optimization

📋 **Planned:**
- Integration with existing cognitive pattern dataset
- Clinical analysis tools
- Performance optimizations
- Extended model support

## Troubleshooting

**Common Issues:**

1. **Model Loading Errors**: Ensure you're using a LLaMA-compatible model and have access
2. **Version Conflicts**: SelfIE requires transformers==4.34.0 (older than the activation patching environment)
3. **GPU Memory**: Use smaller batch sizes or reduce max_new_tokens if you encounter OOM errors
4. **Import Errors**: Make sure the SelfIE library is properly installed from third_party/selfie

**Environment Conflicts:**
Since SelfIE requires an older transformers version, use a separate virtual environment to avoid conflicts with the manual activation patching setup.

## Research References

- [SelfIE Paper](https://arxiv.org/abs/2403.10949): "SelfIE: Self-Interpretation of Large Language Model Embeddings"
- SelfIE enables LLMs to interpret their own embeddings in natural language
- Reveals internal reasoning in ethical decisions, prompt injection, and harmful knowledge recall
- Opens avenues for controlling LLM reasoning through text descriptions of hidden embeddings