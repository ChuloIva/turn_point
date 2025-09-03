"""
Configuration file for Gemma SelfIE Integration
"""

# Model Configuration
MODEL_CONFIG = {
    "model_name": "google/gemma-2-2b-it",
    "torch_dtype": "float16",  # Use float16 for GPU, float32 for CPU
    "device_map": "auto"
}

# SelfIE Configuration
SELFIE_CONFIG = {
    "default_interpretation_template": (
        "<bos>", 0, 0, 0, 0, 0, 
        "\n\nThis neural activation pattern represents:"
    ),
    "batch_size": 2,
    "max_new_tokens": 50,
    "default_position": 'mean'  # use average
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "default_layer": 17,
    "max_patterns_per_run": 5,
    "supported_layers": [10, 15, 17, 20, 25],
    "positions": {
        "last": -1,
        "second_last": -2,
        "middle": "mean",
        "first": 0
    }
}

# Cognitive Pattern Mapping
PATTERN_TEXT_FIELDS = [
    "bad_good_narratives_match.original_thought_pattern",
    "states.depressed", 
    "depressed_text",
    "text",
    "content"
]

# Output Configuration
OUTPUT_CONFIG = {
    "save_json": True,
    "save_markdown": True,
    "save_csv": False,
    "include_activations": False  # Don't save raw activations by default
}

# Interpretation Templates for Different Use Cases
INTERPRETATION_TEMPLATES = {
    "cognitive_pattern": (
        "<bos>", 0, 0, 0, 0, 0, 
        "\n\nThis neural activation represents a cognitive pattern that:"
    ),
    "emotional_state": (
        "<bos>", 0, 0, 0, 0, 0, 
        "\n\nThis activation pattern corresponds to an emotional state of:"
    ),
    "general_concept": (
        "<bos>", 0, 0, 0, 0, 0, 
        "\n\nThis neural representation encodes the concept:"
    ),
    "decision_making": (
        "<bos>", 0, 0, 0, 0, 0, 
        "\n\nThis activation pattern reflects a decision-making process involving:"
    )
}