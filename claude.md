# Project Notes

## NNsight Library Usage

### Layer Output Structure
- `.input` and `.output` now return a tuple of length two:
  - First index `[0]`: positional arguments (the actual tensor data)
  - Second index `[1]`: keyword arguments
- This change was implemented to fix issues with accessing layer outputs
- Source: "@arunasank Appreciate the issue! Should be all set now. .input now returns a tuple of length two where the first index is the positional arguments and the second is the key word arguments"

### Correct Usage Pattern
```python
with model.trace(prompt) as tracer:
    # Access hidden states from layer output
    hidden_states = model.model.layers[layer_idx].output[0]  # [0] gets positional args
    
    # Then index normally for token positions
    activation = hidden_states[:, -1, :].save()  # last token
```

### Common Mistakes to Avoid
- Don't use `layer.output` directly without `[0]` indexing
- The output is NOT just a tensor, it's a tuple that needs to be unpacked