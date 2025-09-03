[10 tools called]

Yes! TransformerLens has exactly what you're looking for - it's called **`run_with_cache()`** and it's the primary way to run a model on a string and see all the internal activations. This is similar to teacher forcing in that you're running the model forward and getting all the intermediate representations.

Here's how to do it:

## Basic Usage

```python
import transformer_lens
from transformer_lens import HookedTransformer

# Load a model
model = HookedTransformer.from_pretrained("gpt2-small")

# Run on a string and get ALL activations
text = "Hello world, how are you?"
logits, cache = model.run_with_cache(text)

print(f"Logits shape: {logits.shape}")  # [batch=1, seq_len, vocab_size]
print(f"Cache type: {type(cache)}")      # ActivationCache object
```

## What You Get in the Cache

The `cache` contains **every single activation** from the model:

### Attention Patterns
```python
# Get attention patterns for layer 0, all heads
attention_patterns = cache["pattern", 0, "attn"]  # Shape: [batch, head, seq_pos, seq_pos]

# Visualize attention (requires circuitsvis)
import circuitsvis as cv
tokens = model.to_str_tokens(text)
cv.attention.attention_patterns(tokens=tokens, attention=attention_patterns)
```

### Neuron Activations
```python
# MLP layer activations
mlp_pre = cache["pre", 5]      # Before activation function in layer 5
mlp_post = cache["post", 5]    # After activation function in layer 5

# Residual stream
resid_pre = cache["resid_pre", 3]   # Residual stream before layer 3
resid_post = cache["resid_post", 3] # Residual stream after layer 3
```

### Attention Head Outputs
```python
# Individual attention head outputs
head_output = cache["result", 2, "attn"]  # Layer 2, all heads: [batch, seq, head, d_model]
single_head = cache["result", 2, "attn"][:, :, 7]  # Just head 7 from layer 2
```

## Advanced Features

### Selective Caching (Memory Efficient)
```python
# Only cache specific activations to save memory
hook_names = ["blocks.0.attn.hook_pattern", "blocks.5.mlp.hook_pre"]
logits, cache = model.run_with_cache(text, names_filter=hook_names)
```

### Stop Early
```python
# Stop forward pass after layer 3 (saves compute)
logits, cache = model.run_with_cache(text, stop_at_layer=4)
```

### Remove Batch Dimension
```python
# Remove the batch dimension if you only have one sequence
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
print(f"Attention shape: {cache['pattern', 0, 'attn'].shape}")  # [head, seq, seq]
```

## Complete Example

```python
import transformer_lens
import torch

# Setup
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")

# Your text
text = "The cat sat on the mat and thought about"

# Run and cache everything
logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# Explore what activated:

# 1. Final predictions
predicted_tokens = model.to_str_tokens(logits.argmax(dim=-1))
print("Next token predictions:", predicted_tokens[-5:])  # Last 5 tokens

# 2. Attention patterns
for layer in [0, 5, 11]:  # First, middle, last layers
    attn = cache["pattern", layer, "attn"]
    print(f"Layer {layer} attention shape: {attn.shape}")
    # Most attended-to positions for each head
    avg_attn = attn.mean(dim=0)  # Average over batch (already removed)
    print(f"Average attention pattern for layer {layer}: {avg_attn}")

# 3. Neuron activations
for layer in [0, 5, 11]:
    mlp_acts = cache["post", layer]
    print(f"Layer {layer} MLP activations shape: {mlp_acts.shape}")
    print(f"Most activated neurons: {mlp_acts.abs().mean(dim=0).topk(5)}")

# 4. Residual stream evolution
for layer in [0, 5, 11]:
    resid = cache["resid_post", layer]
    print(f"Residual stream norm at layer {layer}: {resid.norm(dim=-1).mean()}")
```

## Key Methods for Analysis

Once you have the cache, you can:

- **`cache.decompose_resid()`** - Break down residual stream contributions
- **`cache.logit_attrs()`** - See which components contribute to predictions  
- **`cache.compute_head_results()`** - Get individual head outputs
- **`cache.stack_head_results()`** - Stack all head outputs

This gives you complete visibility into what the model "activates" when processing your text, similar to teacher forcing but with much more detailed introspection capabilities!