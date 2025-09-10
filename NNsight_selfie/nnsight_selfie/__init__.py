from .model_agnostic_selfie import ModelAgnosticSelfie
from .interpretation_prompt import InterpretationPrompt
from .utils import interpret_vectors, get_model_layers
from .device_utils import get_optimal_device, print_device_info, DeviceManager

# Repeng-based steering vector utilities
from .repeng import (
    RepengDatasetGenerator, DatasetEntry, create_quick_dataset,
    RepengActivationExtractor, extract_repeng_activations,
    RepengSteeringVectorGenerator, SteeringVector, create_steering_vector,
    RepengMultiLayerInjector, inject_multi_layer,
    RepengInterpretationAnalyzer, SteeringInterpretationResult,
    create_interpretation_comparison_plot, quick_steering_interpretation_analysis
)

__version__ = "0.1.0"
__all__ = [
    "ModelAgnosticSelfie", 
    "InterpretationPrompt", 
    "interpret_vectors", 
    "get_model_layers",
    "get_optimal_device",
    "print_device_info", 
    "DeviceManager",
    # Repeng utilities
    "RepengDatasetGenerator",
    "DatasetEntry", 
    "create_quick_dataset",
    "RepengActivationExtractor",
    "extract_repeng_activations",
    "RepengSteeringVectorGenerator",
    "SteeringVector",
    "create_steering_vector",
    "RepengMultiLayerInjector",
    "inject_multi_layer",
    # Combined steering + interpretation utilities
    "RepengInterpretationAnalyzer",
    "SteeringInterpretationResult",
    "create_interpretation_comparison_plot",
    "quick_steering_interpretation_analysis"
]