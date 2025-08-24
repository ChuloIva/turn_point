# Small-Scale Testing Strategy

Great question! Here's how to validate your entire pipeline with minimal resources before committing to the full project:

## Phase 1: Proof-of-Concept with Pre-trained SAE (1-2 days)

### 1.1 Find Compatible Pre-trained SAE
```python
# Check available SAEs for your model
# Popular sources: Anthropic's published SAEs, Neel Nanda's SAEs, EleutherAI

def find_compatible_sae():
    # Example: If using Llama-7B, look for SAEs trained on similar models
    compatible_saes = [
        "https://huggingface.co/jbloom/GPT2-Small-SAEs",  # GPT2 example
        # Add actual URLs for your model family
    ]
    
    # Download and test basic functionality
    sae = load_pretrained_sae("path/to/sae")
    
    # Quick compatibility check
    test_input = torch.randn(1, 4096)  # Your model's hidden dim
    try:
        sparse_features = sae.encode(test_input)
        reconstructed = sae.decode(sparse_features)
        print(f"SAE works! Sparsity: {(sparse_features == 0).float().mean():.2f}")
        return sae
    except Exception as e:
        print(f"SAE incompatible: {e}")
        return None
```

### 1.2 Mini Steering Test (10 samples)
```python
def mini_steering_test(model, tokenizer, n_samples=10):
    """Test if basic emotional steering works"""
    
    # Simple steering vectors (manual)
    positive_words = ["happy", "joy", "optimistic", "hopeful", "grateful"]
    negative_words = ["sad", "hopeless", "depressed", "worthless", "empty"]
    
    # Quick steering test
    test_prompt = "Today I feel"
    
    results = []
    for i in range(n_samples):
        # Normal generation
        normal_output = model.generate(test_prompt, max_length=50, do_sample=True)
        
        # Try simple prompting-based "steering" first
        positive_prompt = f"{test_prompt} (thinking positively)"
        positive_output = model.generate(positive_prompt, max_length=50, do_sample=True)
        
        negative_prompt = f"{test_prompt} (feeling down)"  
        negative_output = model.generate(negative_prompt, max_length=50, do_sample=True)
        
        results.append({
            'normal': normal_output,
            'positive': positive_output,
            'negative': negative_output
        })
    
    # Manual inspection
    print("=== STEERING TEST RESULTS ===")
    for i, result in enumerate(results[:3]):  # Show first 3
        print(f"\nSample {i+1}:")
        print(f"Normal: {result['normal']}")
        print(f"Positive: {result['positive']}")  
        print(f"Negative: {result['negative']}")
    
    return results
```

## Phase 2: Minimal Pipeline Test (2-3 days)

### 2.1 Generate 50 Transition Samples
```python
def generate_mini_transitions(model, tokenizer, n_samples=50):
    """Create minimal transition dataset for testing"""
    
    transition_samples = []
    
    # Simple approach: prompt-based transitions
    base_prompts = [
        "I'm having a tough day because",
        "Everything feels overwhelming when", 
        "I struggle with feeling like",
        "It's hard to stay motivated when",
        "I feel hopeless about"
    ]
    
    for i in range(n_samples):
        base_prompt = np.random.choice(base_prompts)
        
        # Generate negative part
        negative_prompt = f"{base_prompt} [continue negatively]"
        negative_part = model.generate(
            negative_prompt, 
            max_length=30,
            do_sample=True,
            temperature=0.8
        )
        
        # Generate positive continuation 
        transition_prompt = f"{negative_part} However, I realize that"
        full_response = model.generate(
            transition_prompt,
            max_length=60, 
            do_sample=True,
            temperature=0.8
        )
        
        transition_samples.append({
            'full_text': full_response,
            'negative_part': negative_part,
            'transition_word': 'However',  # Known transition point
            'sample_id': i
        })
    
    return transition_samples
```

### 2.2 Manual Transition Point Labeling (Skip LLM labeler)
```python
def manual_label_mini_dataset(transition_samples):
    """Manually label transition points for small dataset"""
    
    labeled_samples = []
    
    for sample in transition_samples[:10]:  # Only label 10 for testing
        print(f"\n=== Sample {sample['sample_id']} ===")
        print(sample['full_text'])
        
        # Simple manual labeling
        tokens = sample['full_text'].split()
        print(f"Tokens: {tokens}")
        
        # Find transition word position
        try:
            transition_idx = tokens.index('However')
        except ValueError:
            # Look for other transition words
            transition_words = ['but', 'however', 'yet', 'though', 'still', 'although']
            transition_idx = None
            for word in transition_words:
                if word.lower() in [t.lower() for t in tokens]:
                    transition_idx = [t.lower() for t in tokens].index(word.lower())
                    break
        
        if transition_idx is not None:
            labeled_samples.append({
                **sample,
                'transition_token_idx': transition_idx,
                'tokens': tokens
            })
    
    return labeled_samples
```

### 2.3 Extract Activations (1 layer only)
```python
def extract_mini_activations(model, labeled_samples, target_layer=12):
    """Extract activations from just one layer for testing"""
    
    activation_data = []
    
    # Set up hook for target layer
    activations = []
    def activation_hook(module, input, output):
        activations.append(output[0].detach().cpu())
    
    hook = model.model.layers[target_layer].register_forward_hook(activation_hook)
    
    try:
        for sample in labeled_samples:
            # Clear previous activations
            activations.clear()
            
            # Forward pass
            inputs = tokenizer(sample['full_text'], return_tensors='pt')
            with torch.no_grad():
                model(**inputs)
            
            # Extract transition activation
            if activations and sample['transition_token_idx'] < activations[0].shape[1]:
                transition_activation = activations[0][0, sample['transition_token_idx'], :]
                
                activation_data.append({
                    'activation': transition_activation,
                    'sample_id': sample['sample_id'],
                    'context': sample['full_text'][:100] + "...",
                    'transition_token': sample['tokens'][sample['transition_token_idx']]
                })
    
    finally:
        hook.remove()
    
    return activation_data
```

## Phase 3: SAE Testing (1 day)

### 3.1 Test Pre-trained SAE on Your Data
```python
def test_sae_on_mini_data(sae, activation_data):
    """Test SAE reconstruction quality on your specific data"""
    
    if not activation_data:
        print("No activation data to test!")
        return None
    
    # Stack activations for batch processing
    activations = torch.stack([item['activation'] for item in activation_data])
    
    print(f"Testing SAE on {len(activations)} transition activations...")
    
    # Test reconstruction
    with torch.no_grad():
        sparse_features = sae.encode(activations)
        reconstructed = sae.decode(sparse_features)
    
    # Calculate metrics
    mse_loss = F.mse_loss(activations, reconstructed).item()
    cosine_sim = F.cosine_similarity(activations, reconstructed, dim=-1).mean().item()
    sparsity = (sparse_features == 0).float().mean().item()
    
    results = {
        'mse_loss': mse_loss,
        'cosine_similarity': cosine_sim,
        'sparsity': sparsity,
        'reconstruction_quality': 'good' if cosine_sim > 0.9 else 'poor'
    }
    
    print(f"=== SAE TEST RESULTS ===")
    print(f"MSE Loss: {mse_loss:.6f}")
    print(f"Cosine Similarity: {cosine_sim:.4f}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Quality: {results['reconstruction_quality']}")
    
    return results
```

### 3.2 Analyze Top Features
```python
def analyze_mini_features(sae, activation_data, top_k=10):
    """Look at which features activate most for transitions"""
    
    feature_analysis = []
    
    for item in activation_data:
        activation = item['activation'].unsqueeze(0)
        
        # Get sparse representation
        sparse_features = sae.encode(activation)
        
        # Find top activated features
        top_features = torch.topk(sparse_features.squeeze(), k=top_k)
        
        feature_analysis.append({
            'sample_id': item['sample_id'],
            'context': item['context'],
            'transition_token': item['transition_token'],
            'top_feature_indices': top_features.indices.tolist(),
            'top_feature_values': top_features.values.tolist()
        })
    
    # Look for common patterns
    all_features = []
    for analysis in feature_analysis:
        all_features.extend(analysis['top_feature_indices'])
    
    feature_frequency = Counter(all_features)
    common_features = feature_frequency.most_common(5)
    
    print(f"=== FEATURE ANALYSIS ===")
    print(f"Most common features across transitions:")
    for feature_idx, count in common_features:
        print(f"Feature {feature_idx}: appears {count}/{len(activation_data)} times")
    
    return feature_analysis, common_features
```

## Phase 4: End-to-End Mini Pipeline (1 day)

### 4.1 Complete Mini Run
```python
def run_complete_mini_pipeline():
    """Run the entire pipeline on a tiny scale"""
    
    print("=== STARTING MINI PIPELINE TEST ===\n")
    
    # Step 1: Load model and SAE
    print("1. Loading model and SAE...")
    model, tokenizer = load_model()
    sae = find_compatible_sae()
    
    if sae is None:
        print("❌ No compatible SAE found!")
        return False
    print("✅ Model and SAE loaded successfully")
    
    # Step 2: Test basic steering
    print("\n2. Testing basic steering...")
    steering_results = mini_steering_test(model, tokenizer, n_samples=5)
    print("✅ Steering test completed")
    
    # Step 3: Generate mini transitions
    print("\n3. Generating transition samples...")
    transition_samples = generate_mini_transitions(model, tokenizer, n_samples=10)
    print(f"✅ Generated {len(transition_samples)} transition samples")
    
    # Step 4: Label transitions
    print("\n4. Labeling transition points...")
    labeled_samples = manual_label_mini_dataset(transition_samples)
    print(f"✅ Labeled {len(labeled_samples)} samples")
    
    # Step 5: Extract activations  
    print("\n5. Extracting activations...")
    activation_data = extract_mini_activations(model, labeled_samples)
    print(f"✅ Extracted activations from {len(activation_data)} samples")
    
    # Step 6: Test SAE
    print("\n6. Testing SAE reconstruction...")
    sae_results = test_sae_on_mini_data(sae, activation_data)
    
    if sae_results and sae_results['reconstruction_quality'] == 'good':
        print("✅ SAE reconstruction quality is good!")
    else:
        print("⚠️  SAE reconstruction quality is poor - may need custom training")
    
    # Step 7: Feature analysis
    print("\n7. Analyzing features...")
    feature_analysis, common_features = analyze_mini_features(sae, activation_data)
    print("✅ Feature analysis completed")
    
    # Summary
    print(f"\n=== MINI PIPELINE RESULTS ===")
    success_criteria = [
        ("Steering works", len(steering_results) > 0),
        ("Transitions generated", len(transition_samples) >= 5),
        ("Activations extracted", len(activation_data) >= 3),
        ("SAE reconstruction good", sae_results and sae_results['cosine_similarity'] > 0.85),
        ("Common features found", len(common_features) > 0)
    ]
    
    for criterion, passed in success_criteria:
        status = "✅" if passed else "❌"
        print(f"{status} {criterion}")
    
    overall_success = all(passed for _, passed in success_criteria)
    
    if overall_success:
        print("\n🎉 MINI PIPELINE SUCCESSFUL - Ready for full scale!")
    else:
        print("\n🔧 Issues found - fix before scaling up")
    
    return overall_success, {
        'steering_results': steering_results,
        'sae_results': sae_results,
        'feature_analysis': feature_analysis,
        'common_features': common_features
    }

# Run the test
success, results = run_complete_mini_pipeline()
```

## Phase 5: Scaling Decision Points

### 5.1 Go/No-Go Criteria
```python
def evaluate_scaling_readiness(mini_results):
    """Decide if ready to scale up based on mini pipeline results"""
    
    scaling_checklist = {
        'sae_reconstruction': mini_results['sae_results']['cosine_similarity'] > 0.9,
        'feature_coherence': len(mini_results['common_features']) >= 2,
        'steering_effectiveness': check_steering_quality(mini_results['steering_results']),
        'pipeline_completeness': all_steps_successful(mini_results)
    }
    
    recommendation = "SCALE UP" if all(scaling_checklist.values()) else "FIX ISSUES FIRST"
    
    print(f"\n=== SCALING RECOMMENDATION: {recommendation} ===")
    for check, passed in scaling_checklist.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    if recommendation == "FIX ISSUES FIRST":
        print("\nRecommended fixes:")
        if not scaling_checklist['sae_reconstruction']:
            print("- Train custom SAE or find better pre-trained SAE")
        if not scaling_checklist['feature_coherence']:
            print("- Improve transition quality or labeling")
        if not scaling_checklist['steering_effectiveness']:
            print("- Develop better steering vectors")
    
    return recommendation == "SCALE UP"
```


### Quick Iteration Strategy
1. **Day 1**: Model setup + basic steering test
2. **Day 2**: Transition generation + manual labeling 
3. **Day 3**: Activation extraction + SAE testing
4. **Day 4**: Feature analysis + interpretation
5. **Day 5**: End-to-end pipeline validation

This approach lets you validate every component of your full pipeline with <1% of the computational resources, giving you confidence before committing to the full-scale experiment
