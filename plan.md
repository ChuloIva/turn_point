## Project Summary
You want to analyze cognitive patterns by capturing model activations, performing arithmetic in semantic space, and interpreting the results using SAEs (Sparse Autoencoders) and other methods. The goal is to understand how different cognitive patterns are represented in neural networks.

## Detailed Implementation Requirements

### 1. Model Setup (Modular Architecture)
```python
class ActivationCapturer:
    def __init__(self, model_name="google/gemma-2-2b-it", device="auto"):
        # Support for multiple models
        # - google/gemma-2-2b-it (default)  
        # - google/gemma-2-9b-it
        # - meta-llama/Llama-3.1-8B-Instruct
        # - mistralai/Mistral-7B-Instruct-v0.3
```

### 2. Data Input System (Modular)
- **Configurable dataset parts**: Allow selection of specific cognitive patterns
- **String categories to capture**:
  - Cognitive patterns (various types)
  - Transitions between patterns
  - Healthy/baseline strings
- **Flexible input format**: Support different string collections

### 3. Activation Capture Implementation
```python
def capture_activations(self, strings, layer_nums=[23, 29]):
    # Use TransformersLens model.run_with_cache()
    # Capture activations for entire strings
    # Support multiple layers (default: 23, 29)
    # Return activations for each token position
    # Store by: model + cognitive_pattern + layer + position
```

### 4. Analysis Pipeline
```python
class ActivationAnalyzer:
    def compute_pca(self, activations):
        # PCA on collected activations per cognitive pattern
        
    def arithmetic_operations(self, activations):
        # Semantic space arithmetic between patterns
        
    def sae_interpretation(self, activations):
        # Run activations through SAE for interpretation
        
    def selfie_interpretation(self, activations):
        # Use model to interpret its own activations
        # Inject activations and ask model to describe
```

### 5. Interpretation Methods
1. **SAE Analysis**: 
   - Use pre-trained SAEs (NeuroNPedia compatible)
   - Custom feature interpretation with prompts
   
2. **Selfie Method**:
   - Inject activations during generation
   - Ask model to describe the activation
   - Validity check for captured concepts

3. **Model Steering**:
   - Use activations to steer model responses
   - Self-report questionnaires based on cognitive patterns

### 6. Required Directory Structure
```
project/
├── models/
│   ├── model_loader.py      # Modular model loading
│   └── activation_capture.py # Capture implementation
├── data/
│   ├── cognitive_patterns/   # Your string datasets
│   └── data_loader.py       # Modular data loading  
├── analysis/
│   ├── pca_analysis.py      # PCA computations
│   ├── sae_interface.py     # SAE integration
│   └── interpretation.py    # Selfie + SAE interpretation
├── config/
│   └── config.yaml          # Model, layer, dataset configs
└── main.py                  # Orchestration script
```

### 7. Key Implementation Features
- **Model Agnostic**: Easy switching between Gemma, Llama, Mistral
- **Layer Configurable**: Default layers 23, 29, but adjustable
- **Batch Processing**: Handle multiple cognitive patterns efficiently  
- **Token-Level Granularity**: Analyze which parts of strings activate what
- **Validation Pipeline**: Cross-check SAE vs Selfie interpretations

### 8. Configuration Example
```yaml
model:
  name: "google/gemma-2-2b-it"
  layers: [23, 29]
  
data:
  cognitive_patterns: ["pattern1", "pattern2", "transitions"]
  base_path: "./data/cognitive_patterns/"
  
analysis:
  methods: ["pca", "sae", "selfie"]
  sae_model: "neuronpedia_compatible"
```

