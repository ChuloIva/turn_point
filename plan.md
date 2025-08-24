# Complete Project Plan: SAE Analysis of Therapeutic Language Transitions in LLMs

## Project Overview
**Goal**: Use Sparse Autoencoders to interpret how language models internally represent and process transitions from depressive to positive emotional states, revealing the mechanistic features underlying therapeutic language changes.

---

## Stage 1: Model Selection & Setup

### 1.1 Choose Target Model
**Criteria**:
- Open source with activation access (Llama, Mistral, etc.) OR API with activation hooks
- Sufficient size for complex representations (7B+ parameters)
- Good performance on emotional/conversational tasks

**Implementation**:
```python
# Example with Llama
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Set up activation extraction hooks
def extract_activations(model, layer_idx):
    activations = []
    def hook(module, input, output):
        activations.append(output[0].detach().cpu())
    
    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook)
    return activations, hook_handle
```

### 1.2 Validate Model Capabilities
- Test model's ability to generate depressive vs positive content
- Verify emotional steering works through activation patching
- Document baseline emotional response patterns

---

## Stage 2: Steering Vector Development

### 2.1 Create Emotional Steering Vectors
**Data Collection**:
```python
# Collect contrasting examples
depressive_prompts = [
    "I feel hopeless and everything seems pointless...",
    "Nothing I do matters and I'm worthless...",
    # 100+ examples
]

positive_prompts = [
    "I feel optimistic about the future and my possibilities...", 
    "I'm grateful for the good things in my life...",
    # 100+ examples  
]
```

**Vector Extraction**:
```python
def extract_steering_vectors(model, positive_prompts, negative_prompts, layer_idx):
    pos_activations = []
    neg_activations = []
    
    # Extract activations for each prompt type
    for prompt in positive_prompts:
        activations = get_activations(model, prompt, layer_idx)
        pos_activations.append(activations.mean(dim=1))  # Average over tokens
    
    for prompt in negative_prompts:
        activations = get_activations(model, prompt, layer_idx)
        neg_activations.append(activations.mean(dim=1))
    
    # Calculate steering vectors
    positive_vector = torch.stack(pos_activations).mean(dim=0)
    negative_vector = torch.stack(neg_activations).mean(dim=0)
    
    steering_vector = positive_vector - negative_vector
    return steering_vector, positive_vector, negative_vector
```

### 2.2 Validate Steering Effectiveness
- Test steering magnitude effects (0.5x, 1x, 2x, 3x)
- Verify behavioral changes are consistent and meaningful
- Choose optimal steering strength and layer

---

## Stage 3: Generate Transition Samples  

### 3.1 Create Depressive-to-Positive Transitions
```python
def generate_transition_samples(model, steering_vectors, n_samples=1000):
    transition_samples = []
    
    for i in range(n_samples):
        # Start with depressive steering
        initial_prompt = sample_neutral_prompt()  # "Tell me about your day"
        
        # Generate 20-30 tokens with negative steering
        with activation_patch(model, negative_vector, target_layer, strength=2.0):
            depressive_part = model.generate(
                initial_prompt, 
                max_new_tokens=25,
                do_sample=True,
                temperature=0.8
            )
        
        # Switch to positive steering for continuation
        with activation_patch(model, positive_vector, target_layer, strength=2.0):
            full_response = model.generate(
                depressive_part,
                max_new_tokens=25, 
                do_sample=True,
                temperature=0.8
            )
        
        transition_samples.append({
            'full_text': full_response,
            'depressive_part': depressive_part,
            'positive_part': full_response[len(depressive_part):]
        })
    
    return transition_samples
```

### 3.2 Quality Control
- Manual review of sample quality
- Filter out poor transitions
- Ensure clear emotional shifts are present

---

## Stage 4: Identify Transition Points

### 4.1 Automated Labeling with LLM
```python
def label_transition_points(samples, labeler_model):
    labeled_samples = []
    
    labeling_prompt = """
    Analyze the following text for emotional transitions from negative to positive sentiment.
    Mark the exact token position where the transition begins.
    
    Text: {text}
    
    Return JSON: {{"transition_start_token": <token_index>, "confidence": <0-1>}}
    """
    
    for sample in samples:
        tokens = tokenizer.tokenize(sample['full_text'])
        
        response = labeler_model.generate(
            labeling_prompt.format(text=sample['full_text']),
            max_tokens=100
        )
        
        try:
            label_data = json.loads(response)
            if label_data['confidence'] > 0.7:  # Quality threshold
                labeled_samples.append({
                    **sample,
                    'transition_token': label_data['transition_start_token'],
                    'tokens': tokens
                })
        except:
            continue  # Skip failed parses
    
    return labeled_samples
```

### 4.2 Human Validation
- Manually validate 10-20% of automated labels
- Refine labeling criteria if needed
- Calculate inter-annotator agreement

---

## Stage 5: Extract Transition Activations

### 5.1 Collect Target Activations
```python
def extract_transition_activations(model, labeled_samples, target_layers=[8, 12, 16, 20]):
    transition_data = {}
    
    for layer_idx in target_layers:
        layer_activations = []
        
        for sample in labeled_samples:
            # Get activations for full sequence
            activations, hook = extract_activations(model, layer_idx)
            
            # Forward pass
            inputs = tokenizer(sample['full_text'], return_tensors='pt')
            with torch.no_grad():
                model(**inputs)
            
            # Extract activations at transition point
            transition_idx = sample['transition_token']
            transition_activation = activations[0][0, transition_idx, :] # [batch, seq, hidden]
            
            layer_activations.append({
                'activation': transition_activation,
                'context': sample,
                'position': transition_idx
            })
            
            hook.remove()
        
        transition_data[f'layer_{layer_idx}'] = layer_activations
    
    return transition_data
```

---

## Stage 6: SAE Training or Selection

### 6.1 Option A: Use Pre-trained SAE
```python
# Check if suitable SAE exists for your model
def validate_pretrained_sae(sae, transition_activations):
    reconstruction_metrics = {}
    
    for layer, activations in transition_activations.items():
        test_activations = torch.stack([item['activation'] for item in activations[:100]])
        
        # Test reconstruction quality
        sparse_features = sae.encode(test_activations)
        reconstructed = sae.decode(sparse_features)
        
        mse_loss = F.mse_loss(test_activations, reconstructed)
        cosine_sim = F.cosine_similarity(test_activations, reconstructed, dim=-1).mean()
        
        reconstruction_metrics[layer] = {
            'mse_loss': mse_loss.item(),
            'cosine_similarity': cosine_sim.item()
        }
    
    return reconstruction_metrics

# Decision criteria
if cosine_similarity > 0.95 and mse_loss < threshold:
    use_pretrained_sae = True
else:
    train_custom_sae = True
```

### 6.2 Option B: Train Custom SAE

#### 6.2.1 Collect Training Data (100M+ activations)
```python
def collect_sae_training_data(model, target_layer, data_sources):
    training_activations = []
    
    data_composition = {
        'general_web': 0.6,      # Wikipedia, Common Crawl
        'emotional_content': 0.2, # Therapy conversations, emotional text  
        'transition_examples': 0.15, # Similar to your target data
        'control_samples': 0.05   # Neutral content
    }
    
    for data_type, proportion in data_composition.items():
        texts = sample_texts(data_sources[data_type], int(proportion * total_samples))
        
        for text in texts:
            activations = get_activations(model, text, target_layer)
            # Store all token activations, not just specific positions
            for token_activation in activations:
                training_activations.append(token_activation)
    
    return training_activations
```

#### 6.2.2 Train SAE
```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=32768, sparsity_penalty=0.001):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sparsity_penalty = sparsity_penalty
        
    def forward(self, x):
        hidden = F.relu(self.encoder(x))
        reconstructed = self.decoder(hidden)
        
        # L1 sparsity penalty
        sparsity_loss = self.sparsity_penalty * torch.mean(torch.abs(hidden))
        reconstruction_loss = F.mse_loss(x, reconstructed)
        
        total_loss = reconstruction_loss + sparsity_loss
        
        return {
            'reconstructed': reconstructed,
            'hidden': hidden,
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss
        }

def train_sae(training_data, epochs=100, batch_size=1024):
    sae = SparseAutoencoder()
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = sae(batch)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return sae
```

---

## Stage 7: SAE Analysis of Transitions

### 7.1 Feature Extraction and Analysis
```python
def analyze_transition_features(sae, transition_activations, top_k=20):
    analysis_results = {}
    
    for layer, activations in transition_activations.items():
        layer_results = []
        
        for item in activations:
            activation = item['activation']
            context = item['context']
            
            # Get sparse features
            sparse_features = sae.encode(activation.unsqueeze(0))
            
            # Find top activated features
            top_features = torch.topk(sparse_features.squeeze(), k=top_k)
            
            layer_results.append({
                'sample_id': context.get('id'),
                'context': context['full_text'][:100] + "...",
                'transition_token': context['transition_token'],
                'top_feature_indices': top_features.indices.tolist(),
                'top_feature_values': top_features.values.tolist(),
                'sparsity': (sparse_features == 0).float().mean().item()
            })
        
        analysis_results[layer] = layer_results
    
    return analysis_results
```

### 7.2 Feature Interpretation
```python
def interpret_features(sae, feature_indices, model, tokenizer, n_samples=100):
    """
    Analyze what specific SAE features represent by finding inputs that maximally activate them
    """
    feature_interpretations = {}
    
    for feature_idx in feature_indices:
        # Find texts that maximally activate this feature
        max_activating_texts = find_max_activating_inputs(
            sae, feature_idx, model, tokenizer, n_samples
        )
        
        # Analyze common patterns
        patterns = analyze_activation_patterns(max_activating_texts, feature_idx)
        
        feature_interpretations[feature_idx] = {
            'max_activating_examples': max_activating_texts[:5],
            'common_patterns': patterns,
            'average_activation': patterns['avg_activation']
        }
    
    return feature_interpretations
```

---

## Stage 8: Statistical Analysis

### 8.1 Feature Pattern Analysis
```python
def analyze_transition_patterns(analysis_results):
    # Find features that consistently activate during transitions
    feature_frequency = defaultdict(int)
    feature_contexts = defaultdict(list)
    
    for layer, results in analysis_results.items():
        for result in results:
            for feature_idx in result['top_feature_indices']:
                feature_frequency[f"{layer}_feature_{feature_idx}"] += 1
                feature_contexts[f"{layer}_feature_{feature_idx}"].append(
                    result['context']
                )
    
    # Statistical significance testing
    significant_features = []
    for feature, frequency in feature_frequency.items():
        if frequency > threshold:  # Appears in >10% of transitions
            significant_features.append({
                'feature': feature,
                'frequency': frequency,
                'examples': feature_contexts[feature][:3]
            })
    
    return significant_features
```

### 8.2 Control Comparisons
```python
def compare_with_controls(sae, transition_activations, control_activations):
    """
    Compare feature activation patterns in transitions vs non-transition contexts
    """
    transition_features = extract_features(sae, transition_activations)
    control_features = extract_features(sae, control_activations)
    
    # Statistical tests
    feature_differences = []
    for feature_idx in range(sae.hidden_dim):
        trans_values = [f[feature_idx] for f in transition_features]
        control_values = [f[feature_idx] for f in control_features]
        
        # T-test for significant differences
        t_stat, p_value = stats.ttest_ind(trans_values, control_values)
        
        if p_value < 0.05:  # Significant difference
            feature_differences.append({
                'feature_idx': feature_idx,
                'transition_mean': np.mean(trans_values),
                'control_mean': np.mean(control_values), 
                'p_value': p_value,
                'effect_size': (np.mean(trans_values) - np.mean(control_values)) / np.std(control_values)
            })
    
    return feature_differences
```

---

## Stage 9: Validation and Interpretation

### 9.1 Feature Validation
```python
def validate_discovered_features(sae, significant_features, model):
    validation_results = []
    
    for feature_info in significant_features:
        feature_idx = extract_feature_index(feature_info['feature'])
        
        # Test if artificially activating this feature produces therapeutic language
        test_results = test_feature_intervention(
            model, sae, feature_idx, 
            test_prompts=["I'm feeling down today..."] * 10
        )
        
        validation_results.append({
            'feature': feature_info['feature'],
            'intervention_success_rate': test_results['success_rate'],
            'example_outputs': test_results['examples']
        })
    
    return validation_results
```

### 9.2 Clinical Interpretation
```python
def clinical_interpretation_analysis(significant_features, feature_interpretations):
    """
    Map discovered features to known therapeutic mechanisms
    """
    therapeutic_mechanisms = {
        'reframing': [],
        'hope_injection': [],
        'perspective_shift': [],
        'emotional_regulation': [],
        'cognitive_restructuring': []
    }
    
    # Manual mapping based on feature interpretations
    for feature_info in significant_features:
        feature_idx = extract_feature_index(feature_info['feature'])
        interpretation = feature_interpretations.get(feature_idx, {})
        
        # Analyze patterns and classify
        mechanism = classify_therapeutic_mechanism(interpretation)
        therapeutic_mechanisms[mechanism].append({
            'feature': feature_info['feature'],
            'evidence': interpretation['common_patterns']
        })
    
    return therapeutic_mechanisms
```

---

## Stage 10: Results Documentation

### 10.1 Comprehensive Report
```python
def generate_final_report(all_results):
    report = {
        'executive_summary': {
            'total_transitions_analyzed': len(transition_samples),
            'significant_features_discovered': len(significant_features),
            'layers_analyzed': target_layers,
            'key_findings': extract_key_findings(all_results)
        },
        
        'methodology': {
            'model_used': model_name,
            'steering_approach': steering_method,
            'sae_configuration': sae_config,
            'validation_approach': validation_methods
        },
        
        'detailed_findings': {
            'layer_analysis': layer_by_layer_results,
            'feature_interpretations': feature_meanings,
            'therapeutic_mechanisms': clinical_mappings,
            'statistical_significance': significance_tests
        },
        
        'implications': {
            'therapeutic_applications': potential_applications,
            'ai_safety_considerations': safety_implications,
            'future_research_directions': next_steps
        }
    }
    
    return report
```

---

## Resource Requirements

### Computational Needs
- **GPU**: A100 40GB+ (for SAE training) or V100 32GB minimum
- **Storage**: 500GB+ for activation data and model weights
- **Time**: 2-4 weeks for full pipeline (including SAE training)

### Data Requirements  
- **Training corpus**: 100M+ diverse text samples for SAE training
- **Transition samples**: 10K+ high-quality labeled transitions
- **Validation data**: 1K+ manually validated examples

### Dependencies
```python
# Core libraries
torch >= 2.0
transformers >= 4.21
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.5
seaborn >= 0.11
pandas >= 1.3

# Specialized libraries
sparse_autoencoder  # Custom implementation
activation_patching  # Custom hooks
statistical_analysis # scipy.stats + custom functions
```

This comprehensive plan provides a complete roadmap for using SAEs to understand therapeutic language transitions in LLMs, from initial setup through final analysis and interpretation.
