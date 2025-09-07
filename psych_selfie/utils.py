# Utility functions for SelfIE Psychology Experiments

def process_layers_to_interpret(layers_spec, model_num_layers=None):
    """
    Process layer specifications to support 'all', ranges, and mixed formats.
    
    Args:
        layers_spec: Can be:
            - 'all': All layers from 0 to model_num_layers-1
            - list of ints: [-1, -2, 0, 5] (negative indices supported)
            - tuple for range: (start, end) or (start, end, step)
            - string range: "0:5" or "0:10:2" (start:end or start:end:step)
            - mixed list: ['all', (0, 5), [-1, -2]]
        model_num_layers: Total number of layers in the model (required for 'all')
    
    Returns:
        List of layer indices (positive integers)
    
    Examples:
        >>> process_layers_to_interpret('all', 32)
        [0, 1, 2, ..., 31]
        
        >>> process_layers_to_interpret([-1, -2, -3], 32)
        [29, 30, 31]
        
        >>> process_layers_to_interpret((0, 5), 32)
        [0, 1, 2, 3, 4]
        
        >>> process_layers_to_interpret('0:5', 32)
        [0, 1, 2, 3, 4]
        
        >>> process_layers_to_interpret('0:10:2', 32)
        [0, 2, 4, 6, 8]
        
        >>> process_layers_to_interpret([(-1, 32), '0:3'], 32)
        [0, 1, 2, 31]
    """
    
    # Try to get model info automatically if not provided
    if model_num_layers is None:
        try:
            # Try to import and get model info from SelfIE patcher
            import sys
            if 'selfie_patcher' in sys.modules or 'selfie_patcher' in globals():
                selfie_patcher = sys.modules.get('selfie_patcher') or globals().get('selfie_patcher')
                if hasattr(selfie_patcher, 'model') and selfie_patcher.model is not None:
                    model_info = selfie_patcher.model.config
                    if hasattr(model_info, 'num_hidden_layers'):
                        model_num_layers = model_info.num_hidden_layers
                    elif hasattr(model_info, 'n_layers'):
                        model_num_layers = model_info.n_layers
                    elif hasattr(model_info, 'num_layers'):
                        model_num_layers = model_info.num_layers
        except:
            pass
    
    # Default fallback
    if model_num_layers is None:
        model_num_layers = 32
        print(f"⚠️  Using default layer count: {model_num_layers}")
    
    def normalize_layer_index(idx, total_layers):
        """Convert negative indices to positive"""
        if idx < 0:
            return total_layers + idx
        return idx
    
    def process_single_spec(spec):
        """Process a single layer specification"""
        if spec == 'all':
            return list(range(model_num_layers))
        elif isinstance(spec, str) and ':' in spec:
            # Handle string ranges like "0:5" or "0:10:2"
            parts = spec.split(':')
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                return list(range(start, end))
            elif len(parts) == 3:
                start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
                return list(range(start, end, step))
        elif isinstance(spec, tuple) and len(spec) >= 2:
            # Handle tuple ranges like (0, 5) or (0, 10, 2)
            if len(spec) == 2:
                start, end = spec
                return list(range(start, end))
            elif len(spec) == 3:
                start, end, step = spec
                return list(range(start, end, step))
        elif isinstance(spec, list):
            # Handle list of indices
            return [normalize_layer_index(idx, model_num_layers) for idx in spec]
        elif isinstance(spec, int):
            # Single integer
            return [normalize_layer_index(spec, model_num_layers)]
        
        return []
    
    # Process the input specification
    if isinstance(layers_spec, list) and len(layers_spec) > 0 and not all(isinstance(x, int) for x in layers_spec):
        # Mixed list of specifications
        result_layers = []
        for spec in layers_spec:
            result_layers.extend(process_single_spec(spec))
    else:
        # Single specification
        result_layers = process_single_spec(layers_spec)
    
    # Remove duplicates and sort
    result_layers = sorted(list(set(result_layers)))
    
    # Validate layer indices
    valid_layers = [layer for layer in result_layers if 0 <= layer < model_num_layers]
    if len(valid_layers) != len(result_layers):
        invalid_layers = [layer for layer in result_layers if layer not in valid_layers]
        print(f"⚠️  Removed invalid layer indices: {invalid_layers} (model has {model_num_layers} layers)")
    
    return valid_layers

def test_layer_processing():
    """Test function for layer processing"""
    print("🧪 Testing layer selection function:")
    print(f"  'all' (32 layers): {process_layers_to_interpret('all', 32)[:5]}... (showing first 5)")
    print(f"  [-1, -2, -3]: {process_layers_to_interpret([-1, -2, -3], 32)}")
    print(f"  (0, 5): {process_layers_to_interpret((0, 5), 32)}")
    print(f"  '0:5': {process_layers_to_interpret('0:5', 32)}")
    print(f"  '0:10:2': {process_layers_to_interpret('0:10:2', 32)}")
    print(f"  [(-1, 32), '0:3']: {process_layers_to_interpret([(-1, 32), '0:3'], 32)}")
    print("✅ Layer selection function ready!")

if __name__ == "__main__":
    test_layer_processing()