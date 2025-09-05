# Interpretation Templates for Llama Format
INTERPRETATION_TEMPLATES = {
    "cognitive_pattern": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant that analyzes neural activation patterns.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", 0, 0, 0,
        "Analyze this neural activation and describe what cognitive pattern it represents:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "emotional_state": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant that analyzes neural activation patterns.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", 0, 0, 0, 0, 0,
        "Analyze this activation pattern and describe what emotional state it corresponds to:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "general_concept": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant that analyzes neural representations.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", 0, 0, 0, 0, 0,
        "Analyze this neural representation and describe what concept it encodes:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "decision_making": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant that analyzes neural activation patterns.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", 0, 0, 0, 0, 0,
        "Analyze this activation pattern and describe what decision-making process it reflects:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "cognitive_pattern_gemma": (
        "<start_of_turn>user\n", 0, 0, 0, 0, 0,
        "Analyze this neural activation and describe what cognitive pattern it represents:<end_of_turn>\n<start_of_turn>model\n"
    ),
    "emotional_state_gemma": (
        "<start_of_turn>user\n", 0, 0, 0, 0, 0,
        "Analyze this activation pattern and describe what emotional state it corresponds to:<end_of_turn>\n<start_of_turn>model\n"
    ),
    "general_concept_gemma": (
        "<start_of_turn>user\n", 0, 0, 0, 0, 0,
        "Analyze this neural representation and describe what concept it encodes:<end_of_turn>\n<start_of_turn>model\n"
    ),
    "decision_making_gemma": (
        "<start_of_turn>user\n", 0, 0, 0, 0, 0,
        "Analyze this activation pattern and describe what decision-making process it reflects:<end_of_turn>\n<start_of_turn>model\n"
    )
}