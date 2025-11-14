"""
Semantic path learning for activation space navigation.

This module provides tools for learning optimal paths between concepts in neural
activation space, going beyond simple linear or spherical interpolation.

Key features:
- Generate semantic landmarks between concepts (e.g., "sad" → "neutral" → "happy")
- Multiple path representations (landmarks, parametric curves, tangent fields)
- Three generalization methods for applying learned paths to new concept pairs
- Multi-pair learning to extract universal transformation patterns

Compatible with ModelAgnosticSelfie for extraction, manipulation, and interpretation.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
import numpy as np
from dataclasses import dataclass, field
import warnings

try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Some curve fitting features will be limited.")

from .vector_operations import interpolate_vectors


# ============================================================================
# HELPER FUNCTIONS: Layer Parsing
# ============================================================================

def parse_layer_spec(layers: Union[int, str], total_layers: int) -> List[int]:
    """
    Parse layer specification into list of layer indices.

    Args:
        layers: Layer specification - can be:
            - int: Single layer index (e.g., 7)
            - 'all': All layers
            - 'start:end': Range of layers (e.g., '3:13' for layers 3-12)
            - 'start:end:step': Range with step (e.g., '0:20:2' for every other layer)
        total_layers: Total number of layers in the model

    Returns:
        List of layer indices

    Examples:
        >>> parse_layer_spec(7, 32)
        [7]
        >>> parse_layer_spec('all', 32)
        [0, 1, 2, ..., 31]
        >>> parse_layer_spec('3:13', 32)
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> parse_layer_spec('0:20:2', 32)
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    """
    # Handle integer (single layer)
    if isinstance(layers, int):
        if layers < 0 or layers >= total_layers:
            raise ValueError(f"Layer index {layers} out of range [0, {total_layers-1}]")
        return [layers]

    # Handle string specifications
    if not isinstance(layers, str):
        raise TypeError(f"layers must be int or str, got {type(layers)}")

    layers_str = layers.strip()

    # Handle 'all'
    if layers_str.lower() == 'all':
        return list(range(total_layers))

    # Handle range specification (e.g., '3:13' or '0:20:2')
    if ':' in layers_str:
        parts = layers_str.split(':')

        if len(parts) == 2:
            # Format: 'start:end'
            start, end = parts
            start_idx = int(start) if start else 0
            end_idx = int(end) if end else total_layers
            step = 1
        elif len(parts) == 3:
            # Format: 'start:end:step'
            start, end, step_str = parts
            start_idx = int(start) if start else 0
            end_idx = int(end) if end else total_layers
            step = int(step_str) if step_str else 1
        else:
            raise ValueError(f"Invalid range format: '{layers_str}'. Use 'start:end' or 'start:end:step'")

        # Validate range
        if start_idx < 0 or start_idx >= total_layers:
            raise ValueError(f"Start index {start_idx} out of range [0, {total_layers-1}]")
        if end_idx < 0 or end_idx > total_layers:
            raise ValueError(f"End index {end_idx} out of range [0, {total_layers}]")
        if start_idx >= end_idx:
            raise ValueError(f"Start index {start_idx} must be less than end index {end_idx}")
        if step <= 0:
            raise ValueError(f"Step {step} must be positive")

        return list(range(start_idx, end_idx, step))

    # If we get here, try to parse as single integer
    try:
        layer_idx = int(layers_str)
        if layer_idx < 0 or layer_idx >= total_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {total_layers-1}]")
        return [layer_idx]
    except ValueError:
        raise ValueError(f"Invalid layer specification: '{layers_str}'. Use int, 'all', or 'start:end[:step]'")


# ============================================================================
# HELPER FUNCTIONS: Keyword Generation
# ============================================================================

def generate_keywords_with_model(
    selfie,
    start_concept: str,
    end_concept: str,
    num_steps: int = 7,
    temperature: float = 0.3,
    max_retries: int = 3
) -> List[str]:
    """
    Use the model itself to generate semantic gradient keywords.

    This is more principled than template-based generation because the model
    knows its own semantic space. It generates actual intermediate concepts
    rather than just adding modifiers like "slightly" or "very".

    Args:
        selfie: ModelAgnosticSelfie instance with loaded model
        start_concept: Starting concept (e.g., "sad")
        end_concept: Ending concept (e.g., "happy")
        num_steps: Total number of steps including start and end
        temperature: Sampling temperature (lower = more deterministic)
        max_retries: Number of retries if parsing fails

    Returns:
        List of keyword strings representing semantic gradient

    Examples:
        >>> keywords = generate_keywords_with_model(selfie, "sad", "happy", 7)
        >>> # Might return: ["sad", "melancholic", "somber", "neutral", "content", "cheerful", "happy"]
    """
    # Craft prompt with few-shot examples
    prompt = f"""Generate a semantic gradient of exactly {num_steps} concepts between "{start_concept}" and "{end_concept}".

The gradient should smoothly transition from the starting concept to the ending concept, with each step representing a meaningful intermediate state.

Examples:
- sad to happy (7 steps): sad, melancholic, somber, neutral, content, pleased, happy
- angry to calm (7 steps): angry, irritated, tense, neutral, relaxed, peaceful, calm
- anxious to relaxed (5 steps): anxious, worried, neutral, comfortable, relaxed
- cold to hot (5 steps): cold, cool, lukewarm, warm, hot

Now generate for: {start_concept} to {end_concept} ({num_steps} steps)

Format your response as a comma-separated list of exactly {num_steps} words, starting with "{start_concept}" and ending with "{end_concept}".

Answer:"""

    # Load a fresh transformers model temporarily for generation
    # This avoids conflicts with NNsight's meta device wrapping
    print(f"🔄 Loading temporary model for keyword generation...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import gc

        # Get model name from selfie
        model_name = selfie.model.tokenizer.name_or_path
        device = selfie.device

        # Ensure device is a torch.device object
        if isinstance(device, str):
            device = torch.device(device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model - use device_map="auto" or manual device placement
        print(f"   Loading model to {device}...")

        device_type = device.type if hasattr(device, 'type') else str(device)

        if device_type == "cuda":
            # For CUDA, use device_map with device index
            device_index = device.index if hasattr(device, 'index') and device.index is not None else 0
            temp_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{device_index}"
            )
        elif device_type == "mps":
            # For MPS, load to CPU first then move (device_map doesn't support MPS well)
            temp_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            temp_model = temp_model.to(device)
        else:
            # For CPU or other devices
            temp_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
            )
            temp_model = temp_model.to(device)

        print(f"✅ Temporary model loaded on {device}")

        # Try generation with retries
        for attempt in range(max_retries):
            try:
                # Tokenize input
                tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

                # Generate
                with torch.no_grad():
                    output = temp_model.generate(
                        tokens,
                        max_new_tokens=100,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Decode response
                response = tokenizer.decode(output[0], skip_special_tokens=True)

                # Extract the answer part (after "Answer:")
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()

                # Parse the response
                keywords = parse_keyword_list(response, num_steps, start_concept, end_concept)

                if keywords and len(keywords) == num_steps:
                    print(f"✅ Model generated keywords: {keywords}")

                    # IMPORTANT: Clean up immediately
                    del temp_model
                    del tokenizer
                    del tokens
                    del output
                    gc.collect()
                    if device_type == "cuda":
                        torch.cuda.empty_cache()
                    elif device_type == "mps":
                        torch.mps.empty_cache()

                    print(f"🧹 Temporary model unloaded and memory cleared")
                    return keywords
                else:
                    print(f"⚠️ Attempt {attempt + 1}: Got {len(keywords) if keywords else 0} keywords, expected {num_steps}")

            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} generation failed: {str(e)}")
                continue

        # If all attempts failed, clean up and fallback
        print(f"⚠️ Model generation failed after {max_retries} attempts")

    except Exception as e:
        print(f"⚠️ Failed to load temporary model: {str(e)}")

    finally:
        # Ensure cleanup happens even if there's an error
        try:
            if 'temp_model' in locals():
                del temp_model
            if 'tokenizer' in locals():
                del tokenizer
            if 'gc' in dir():
                gc.collect()
            # Use device_type if available, otherwise try to get it from device
            if 'device_type' in locals():
                dt = device_type
            elif 'device' in locals():
                dt = device.type if hasattr(device, 'type') else str(device)
            else:
                dt = None

            if dt == "cuda":
                torch.cuda.empty_cache()
            elif dt == "mps":
                torch.mps.empty_cache()
            print(f"🧹 Cleanup completed")
        except:
            pass

    # Fallback to template-based generation
    print(f"⚠️ Falling back to template-based keyword generation")
    return generate_intermediate_keywords_template(start_concept, end_concept, num_steps, "emotion")


def parse_keyword_list(
    response: str,
    expected_count: int,
    start_concept: str,
    end_concept: str
) -> Optional[List[str]]:
    """
    Parse model's response to extract keyword list.

    Args:
        response: Raw model response
        expected_count: Expected number of keywords
        start_concept: Expected first keyword
        end_concept: Expected last keyword

    Returns:
        List of keywords, or None if parsing failed
    """
    # Clean up response
    response = response.strip()

    # Try to extract comma-separated list
    # Look for patterns like: "word, word, word" or "word,word,word"

    # Remove common prefixes
    for prefix in ["Answer:", "Gradient:", "List:", "Here is:", "Here's:"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

    # Remove quotes if present
    response = response.strip('"\'')

    # Split by comma
    keywords = [k.strip().strip('"\'.,;:') for k in response.split(',')]

    # Filter empty strings
    keywords = [k for k in keywords if k]

    # Validate
    if len(keywords) != expected_count:
        # Try to extract first N words
        keywords = keywords[:expected_count]

    # Ensure start and end match
    if len(keywords) >= 2:
        keywords[0] = start_concept
        keywords[-1] = end_concept

    # Basic validation
    if len(keywords) == expected_count and all(len(k) > 0 for k in keywords):
        return keywords

    return None


def generate_intermediate_keywords_template(
    start_concept: str,
    end_concept: str,
    num_steps: int = 7,
    template: str = "emotion"
) -> List[str]:
    """
    Template-based keyword generation (original implementation).

    This is a fallback when model-based generation fails.

    Args:
        start_concept: Starting concept
        end_concept: Ending concept
        num_steps: Number of steps
        template: Template type ("emotion", "intensity", "custom")

    Returns:
        List of template-generated keywords
    """
    keywords = []

    if template == "emotion":
        # Emotional transition template with intensity modifiers
        if num_steps == 3:
            keywords = [start_concept, "neutral", end_concept]
        elif num_steps == 5:
            keywords = [
                start_concept,
                f"slightly {start_concept}",
                "neutral",
                f"slightly {end_concept}",
                end_concept
            ]
        elif num_steps == 7:
            keywords = [
                start_concept,
                f"very {start_concept}",
                f"slightly {start_concept}",
                "neutral",
                f"slightly {end_concept}",
                f"moderately {end_concept}",
                end_concept
            ]
        else:
            # For arbitrary num_steps, distribute intensity modifiers
            keywords.append(start_concept)

            # Negative side
            neg_steps = (num_steps - 3) // 2
            for i in range(neg_steps, 0, -1):
                if i > neg_steps // 2:
                    keywords.append(f"very {start_concept}")
                else:
                    keywords.append(f"slightly {start_concept}")

            # Neutral point
            keywords.append("neutral")

            # Positive side
            pos_steps = (num_steps - 3) - neg_steps
            for i in range(pos_steps):
                if i < pos_steps // 2:
                    keywords.append(f"slightly {end_concept}")
                else:
                    keywords.append(f"moderately {end_concept}")

            keywords.append(end_concept)

    elif template == "intensity":
        # Intensity-based transitions (e.g., cold → hot)
        intensities = ["very weak", "weak", "slightly weak", "neutral",
                      "slightly strong", "strong", "very strong"]

        # Map num_steps to intensity levels
        step_indices = np.linspace(0, len(intensities) - 1, num_steps).astype(int)
        keywords = [start_concept if i == 0 else
                   end_concept if i == num_steps - 1 else
                   intensities[step_indices[i]]
                   for i in range(num_steps)]

    else:
        # Simple linear keyword generation
        keywords = [start_concept]
        for i in range(1, num_steps - 1):
            alpha = i / (num_steps - 1)
            if alpha < 0.5:
                keywords.append(f"somewhat {start_concept}")
            elif alpha == 0.5:
                keywords.append("neutral")
            else:
                keywords.append(f"somewhat {end_concept}")
        keywords.append(end_concept)

    return keywords


def generate_intermediate_keywords(
    start_concept: str,
    end_concept: str,
    num_steps: int = 7,
    template: str = "emotion",
    selfie = None,
    temperature: float = 0.3
) -> List[str]:
    """
    Generate intermediate semantic keywords between two concepts.

    Args:
        start_concept: Starting concept (e.g., "sad")
        end_concept: Ending concept (e.g., "happy")
        num_steps: Total number of steps including start and end (minimum 3)
        template: Keyword generation strategy
            - "model_generated": Use the LLM to generate semantic gradients (RECOMMENDED)
            - "emotion": For emotional transitions (negative → positive)
            - "intensity": For intensity gradients (e.g., "cold" → "hot")
            - "custom": Use custom interpolation logic
        selfie: ModelAgnosticSelfie instance (required for "model_generated" template)
        temperature: Temperature for model generation (only used with "model_generated")

    Returns:
        List of keyword strings representing semantic gradient

    Examples:
        >>> # Model-generated (best quality)
        >>> keywords = generate_intermediate_keywords(
        ...     "sad", "happy", num_steps=7,
        ...     template="model_generated", selfie=selfie
        ... )
        >>> # Might return: ["sad", "melancholic", "somber", "neutral", "content", "cheerful", "happy"]

        >>> # Template-based (fallback)
        >>> generate_intermediate_keywords("sad", "happy", num_steps=5, template="emotion")
        ["sad", "slightly sad", "neutral", "slightly happy", "happy"]
    """
    assert num_steps >= 3, "Need at least 3 steps (start, middle, end)"

    # Use model-based generation if requested
    if template == "model_generated":
        if selfie is None:
            warnings.warn("model_generated template requires selfie parameter. Falling back to emotion template.")
            return generate_intermediate_keywords_template(start_concept, end_concept, num_steps, "emotion")

        return generate_keywords_with_model(selfie, start_concept, end_concept, num_steps, temperature)

    # Otherwise use template-based generation
    return generate_intermediate_keywords_template(start_concept, end_concept, num_steps, template)




def extract_landmark_vectors(
    selfie,
    keywords: List[str],
    layer: Union[int, str],
    use_chat_template: bool = True,
    prompt_template: str = "think about the {word}"
) -> Union[List[torch.Tensor], Dict[int, List[torch.Tensor]]]:
    """
    Extract activation vectors for a list of landmark keywords.

    Args:
        selfie: ModelAgnosticSelfie instance
        keywords: List of concept keywords to extract
        layer: Layer specification - can be:
            - int: Single layer index (e.g., 7)
            - 'all': All layers
            - 'start:end': Range of layers (e.g., '3:13')
            - 'start:end:step': Range with step (e.g., '0:20:2')
        use_chat_template: Whether to use chat template formatting
        prompt_template: Template for concept extraction (must contain {word})

    Returns:
        - If single layer (int): List of activation tensors, one per keyword
        - If multi-layer (str): Dict mapping layer_idx -> List of vectors

    Examples:
        >>> # Single layer
        >>> keywords = ["sad", "neutral", "happy"]
        >>> vectors = extract_landmark_vectors(selfie, keywords, layer=15)
        >>> len(vectors)
        3

        >>> # Multi-layer
        >>> vectors_dict = extract_landmark_vectors(selfie, keywords, layer='3:13')
        >>> vectors_dict[7]  # Vectors for layer 7
        [tensor(...), tensor(...), tensor(...)]
    """
    # Parse layer specification
    total_layers = len(selfie.layer_paths)
    layer_indices = parse_layer_spec(layer, total_layers)

    # Single layer case - maintain backward compatibility
    if len(layer_indices) == 1:
        single_layer = layer_indices[0]
        activations = selfie.get_concept_activations(
            concepts=keywords,
            layer_indices=[single_layer],
            use_chat_template=use_chat_template,
            prompt_template=prompt_template
        )
        # Extract vectors in order
        vectors = [activations[keyword][single_layer] for keyword in keywords]
        return vectors

    # Multi-layer case
    activations = selfie.get_concept_activations(
        concepts=keywords,
        layer_indices=layer_indices,
        use_chat_template=use_chat_template,
        prompt_template=prompt_template
    )

    # Reorganize: Dict[layer_idx -> List[vectors]]
    result = {}
    for layer_idx in layer_indices:
        result[layer_idx] = [activations[keyword][layer_idx] for keyword in keywords]

    return result


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def compute_alignment_transform(
    source_vec_a: torch.Tensor,
    source_vec_b: torch.Tensor,
    target_vec_a: torch.Tensor,
    target_vec_b: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute geometric transformation to align source pair with target pair.

    This finds a rotation and scaling that maps (source_a, source_b) to (target_a, target_b).

    Args:
        source_vec_a: First vector of source pair
        source_vec_b: Second vector of source pair
        target_vec_a: First vector of target pair
        target_vec_b: Second vector of target pair

    Returns:
        Dictionary containing:
        - 'rotation_matrix': Rotation matrix
        - 'scale_factor': Scaling factor
        - 'translation': Translation vector

    Example:
        >>> # Align (sad, happy) to (angry, calm)
        >>> transform = compute_alignment_transform(sad_vec, happy_vec, angry_vec, calm_vec)
    """
    # Flatten all vectors
    src_a = source_vec_a.flatten()
    src_b = source_vec_b.flatten()
    tgt_a = target_vec_a.flatten()
    tgt_b = target_vec_b.flatten()

    # Compute direction vectors
    src_direction = F.normalize(src_b - src_a, dim=0)
    tgt_direction = F.normalize(tgt_b - tgt_a, dim=0)

    # Compute rotation to align directions (using Householder reflection)
    # For high-dimensional spaces, we use the reflection that maps src_direction to tgt_direction
    v = tgt_direction - src_direction
    v_norm = torch.norm(v)

    if v_norm > 1e-6:
        v = v / v_norm
        # Householder matrix: I - 2vv^T
        # For efficiency, we don't explicitly form the matrix
        rotation_matrix = torch.eye(len(src_a), device=src_a.device, dtype=src_a.dtype) - 2 * torch.outer(v, v)
    else:
        # Vectors already aligned
        rotation_matrix = torch.eye(len(src_a), device=src_a.device, dtype=src_a.dtype)

    # Compute scale factor
    src_magnitude = torch.norm(src_b - src_a)
    tgt_magnitude = torch.norm(tgt_b - tgt_a)
    scale_factor = tgt_magnitude / (src_magnitude + 1e-8)

    # Compute translation (align starting points after rotation and scaling)
    rotated_scaled_a = rotation_matrix @ src_a * scale_factor
    translation = tgt_a - rotated_scaled_a

    return {
        'rotation_matrix': rotation_matrix,
        'scale_factor': scale_factor,
        'translation': translation,
        'source_center': (src_a + src_b) / 2,
        'target_center': (tgt_a + tgt_b) / 2
    }


def apply_alignment_transform(
    vector: torch.Tensor,
    transform: Dict[str, torch.Tensor],
    original_shape: Optional[Tuple] = None
) -> torch.Tensor:
    """
    Apply geometric alignment transformation to a vector.

    Args:
        vector: Vector to transform
        transform: Transform dict from compute_alignment_transform
        original_shape: Optional shape to reshape result to

    Returns:
        Transformed vector
    """
    vec_flat = vector.flatten()

    # Apply: rotate -> scale -> translate
    transformed = transform['rotation_matrix'] @ vec_flat
    transformed = transformed * transform['scale_factor']
    transformed = transformed + transform['translation']

    if original_shape is not None:
        transformed = transformed.reshape(original_shape)
    else:
        transformed = transformed.reshape(vector.shape)

    return transformed


def compute_relative_position(
    vector: torch.Tensor,
    anchor_a: torch.Tensor,
    anchor_b: torch.Tensor
) -> Dict[str, float]:
    """
    Compute relative position of vector with respect to anchor pair.

    Encodes where 'vector' sits relative to the (anchor_a, anchor_b) axis.

    Args:
        vector: Vector to encode
        anchor_a: First anchor vector
        anchor_b: Second anchor vector

    Returns:
        Dictionary with:
        - 'alpha': Position along anchor_a → anchor_b axis (0.0 to 1.0)
        - 'perpendicular_distance': Distance from axis
        - 'perpendicular_direction': Unit vector pointing away from axis
    """
    vec = vector.flatten()
    a = anchor_a.flatten()
    b = anchor_b.flatten()

    # Project vector onto a→b direction
    axis = b - a
    axis_normalized = F.normalize(axis, dim=0)

    # Component along axis
    vec_from_a = vec - a
    projection_length = torch.dot(vec_from_a, axis_normalized)
    axis_length = torch.norm(axis)

    alpha = (projection_length / axis_length).item()

    # Perpendicular component
    projection = a + alpha * axis
    perpendicular = vec - projection
    perp_distance = torch.norm(perpendicular).item()

    if perp_distance > 1e-6:
        perp_direction = F.normalize(perpendicular, dim=0)
    else:
        perp_direction = torch.zeros_like(vec)

    return {
        'alpha': alpha,
        'perpendicular_distance': perp_distance,
        'perpendicular_direction': perp_direction
    }


def reconstruct_from_relative_position(
    relative_pos: Dict[str, float],
    new_anchor_a: torch.Tensor,
    new_anchor_b: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct vector from relative position encoding using new anchors.

    Args:
        relative_pos: Relative position dict from compute_relative_position
        new_anchor_a: New first anchor
        new_anchor_b: New second anchor

    Returns:
        Reconstructed vector
    """
    a = new_anchor_a.flatten()
    b = new_anchor_b.flatten()

    # Reconstruct position along axis
    axis = b - a
    point_on_axis = a + relative_pos['alpha'] * axis

    # Add perpendicular component
    if isinstance(relative_pos['perpendicular_direction'], torch.Tensor):
        perp_component = relative_pos['perpendicular_distance'] * relative_pos['perpendicular_direction']
        reconstructed = point_on_axis + perp_component
    else:
        reconstructed = point_on_axis

    return reconstructed.reshape(new_anchor_a.shape)


# ============================================================================
# PATH REPRESENTATION CLASSES
# ============================================================================

@dataclass
class LandmarkPath:
    """
    Piecewise path defined by landmark vectors.

    Stores explicit landmark vectors and interpolates between them using
    piecewise spherical linear interpolation (slerp).

    Attributes:
        landmarks: List of activation vectors defining the path
        alphas: Position of each landmark along path (0.0 to 1.0)
        metadata: Additional info (layer, concepts, etc.)
    """
    landmarks: List[torch.Tensor]
    alphas: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.landmarks) == len(self.alphas), \
            f"Mismatch: {len(self.landmarks)} landmarks but {len(self.alphas)} alphas"
        assert len(self.landmarks) >= 2, "Need at least 2 landmarks"

        # Ensure alphas are sorted
        assert all(self.alphas[i] <= self.alphas[i+1] for i in range(len(self.alphas)-1)), \
            "Alphas must be in ascending order"
        assert self.alphas[0] == 0.0 and self.alphas[-1] == 1.0, \
            "Alphas must start at 0.0 and end at 1.0"

    def interpolate(self, alpha: float, method: str = "spherical") -> torch.Tensor:
        """
        Get vector at position alpha along learned path.

        Args:
            alpha: Position along path (0.0 to 1.0)
            method: Interpolation method ("spherical" or "linear")

        Returns:
            Interpolated activation vector
        """
        # Find surrounding landmarks
        for i in range(len(self.alphas) - 1):
            if self.alphas[i] <= alpha <= self.alphas[i + 1]:
                # Interpolate between landmarks[i] and landmarks[i+1]
                local_alpha = (alpha - self.alphas[i]) / (self.alphas[i + 1] - self.alphas[i])
                return interpolate_vectors(
                    self.landmarks[i],
                    self.landmarks[i + 1],
                    alpha=local_alpha,
                    method=method
                )

        # Edge cases
        if alpha <= 0.0:
            return self.landmarks[0]
        elif alpha >= 1.0:
            return self.landmarks[-1]
        else:
            raise ValueError(f"Alpha {alpha} out of range")

    def apply_geometric_alignment(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply learned path to new concept pair using geometric alignment.

        Method: Compute rotation/scaling to align original endpoints with new endpoints,
        then transform all landmarks and interpolate.

        Args:
            new_vec_a: New start vector
            new_vec_b: New end vector
            alpha: Position along path (0.0 to 1.0)

        Returns:
            Transformed vector at position alpha
        """
        # Compute alignment transformation
        transform = compute_alignment_transform(
            self.landmarks[0],
            self.landmarks[-1],
            new_vec_a,
            new_vec_b
        )

        # Transform all landmarks
        transformed_landmarks = [
            apply_alignment_transform(lm, transform)
            for lm in self.landmarks
        ]

        # Create temporary path with transformed landmarks
        temp_path = LandmarkPath(
            landmarks=transformed_landmarks,
            alphas=self.alphas.copy()
        )

        # Interpolate in transformed space
        return temp_path.interpolate(alpha)

    def apply_relative_encoding(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply learned path using relative position encoding.

        Method: Encode each landmark's position relative to original endpoints,
        then reconstruct using new endpoints.

        Args:
            new_vec_a: New start vector
            new_vec_b: New end vector
            alpha: Position along path (0.0 to 1.0)

        Returns:
            Reconstructed vector at position alpha
        """
        # Find target landmark at alpha
        target_landmark = self.interpolate(alpha)

        # Compute relative position in original space
        relative_pos = compute_relative_position(
            target_landmark,
            self.landmarks[0],
            self.landmarks[-1]
        )

        # Reconstruct in new space
        return reconstruct_from_relative_position(
            relative_pos,
            new_vec_a,
            new_vec_b
        )

    def apply_direction_magnitude(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply learned path using direction field + magnitude decomposition.

        Method: Extract direction and magnitude patterns from original path,
        apply to new concept pair.

        Args:
            new_vec_a: New start vector
            new_vec_b: New end vector
            alpha: Position along path (0.0 to 1.0)

        Returns:
            Vector following learned direction field
        """
        # Compute tangent direction at alpha in original space
        epsilon = 0.01
        vec_at_alpha = self.interpolate(alpha)
        vec_before = self.interpolate(max(0.0, alpha - epsilon))
        vec_after = self.interpolate(min(1.0, alpha + epsilon))

        # Tangent direction
        tangent = (vec_after.flatten() - vec_before.flatten()) / (2 * epsilon)
        tangent_normalized = F.normalize(tangent, dim=0)

        # Original axis direction
        orig_axis = self.landmarks[-1].flatten() - self.landmarks[0].flatten()
        orig_axis_normalized = F.normalize(orig_axis, dim=0)

        # New axis direction
        new_axis = new_vec_b.flatten() - new_vec_a.flatten()
        new_axis_normalized = F.normalize(new_axis, dim=0)

        # Compute how much tangent deviates from axis
        tangent_axis_component = torch.dot(tangent_normalized, orig_axis_normalized)
        tangent_perp = tangent_normalized - tangent_axis_component * orig_axis_normalized
        tangent_perp_normalized = F.normalize(tangent_perp, dim=0) if torch.norm(tangent_perp) > 1e-6 else torch.zeros_like(tangent_perp)

        # Reconstruct tangent in new space
        new_tangent = tangent_axis_component * new_axis_normalized + torch.norm(tangent_perp) * tangent_perp_normalized

        # Start from new_vec_a and follow direction
        new_axis_length = torch.norm(new_axis)
        orig_axis_length = torch.norm(orig_axis)
        scale = new_axis_length / (orig_axis_length + 1e-8)

        # Simple linear blend with directional adjustment
        base_interpolation = (1 - alpha) * new_vec_a.flatten() + alpha * new_vec_b.flatten()

        # Add learned directional deviation
        displacement_magnitude = torch.norm(vec_at_alpha.flatten() - ((1-alpha) * self.landmarks[0].flatten() + alpha * self.landmarks[-1].flatten()))
        result = base_interpolation + scale * displacement_magnitude * new_tangent

        return result.reshape(new_vec_a.shape)


@dataclass
class ParametricCurvePath:
    """
    Path represented as parametric curve fitted through landmarks.

    Uses Bezier curves, splines, or polynomial fitting to create smooth
    continuous path through landmark vectors.

    Attributes:
        curve_type: Type of curve ("bezier", "spline", or "polynomial")
        control_points: Control points defining the curve
        landmarks: Original landmark vectors (for reference)
        parameters: Additional curve parameters
        metadata: Extra information
    """
    curve_type: Literal["bezier", "spline", "polynomial"]
    control_points: List[torch.Tensor]
    landmarks: List[torch.Tensor]
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit_from_landmarks(
        cls,
        landmarks: List[torch.Tensor],
        alphas: List[float],
        curve_type: str = "bezier",
        **kwargs
    ) -> 'ParametricCurvePath':
        """
        Fit parametric curve through landmark vectors.

        Args:
            landmarks: List of vectors to fit
            alphas: Position of each landmark (0.0 to 1.0)
            curve_type: "bezier", "spline", or "polynomial"
            **kwargs: Additional parameters for curve fitting

        Returns:
            ParametricCurvePath instance
        """
        if curve_type == "bezier":
            # Cubic Bezier with 4 control points
            # P0 = start, P3 = end, P1 and P2 are computed
            p0 = landmarks[0]
            p3 = landmarks[-1]

            # Estimate control points from landmarks
            if len(landmarks) >= 4:
                # Use actual landmarks as approximate control points
                p1 = landmarks[len(landmarks) // 3]
                p2 = landmarks[2 * len(landmarks) // 3]
            else:
                # Generate control points
                p1 = landmarks[0] + (landmarks[-1] - landmarks[0]) * 0.33
                p2 = landmarks[0] + (landmarks[-1] - landmarks[0]) * 0.67

            control_points = [p0, p1, p2, p3]
            parameters = {'degree': 3}

        elif curve_type == "spline" and SCIPY_AVAILABLE:
            # Will implement spline fitting in interpolate method
            control_points = landmarks
            parameters = {
                'smoothing': kwargs.get('smoothing', 0.0),
                'degree': kwargs.get('degree', 3)
            }

        elif curve_type == "polynomial":
            # Polynomial fit (degree = num_landmarks - 1)
            control_points = landmarks
            parameters = {'degree': len(landmarks) - 1}

        else:
            # Fallback: use landmarks as control points
            control_points = landmarks
            parameters = {}

        return cls(
            curve_type=curve_type,
            control_points=control_points,
            landmarks=landmarks,
            parameters=parameters,
            metadata={'alphas': alphas}
        )

    def interpolate(self, alpha: float) -> torch.Tensor:
        """
        Evaluate curve at position alpha.

        Args:
            alpha: Position along curve (0.0 to 1.0)

        Returns:
            Vector at position alpha
        """
        t = alpha

        if self.curve_type == "bezier":
            # Cubic Bezier formula
            p0, p1, p2, p3 = self.control_points
            result = (
                (1 - t)**3 * p0 +
                3 * (1 - t)**2 * t * p1 +
                3 * (1 - t) * t**2 * p2 +
                t**3 * p3
            )
            return result

        elif self.curve_type == "spline":
            # Piecewise spherical interpolation through landmarks
            # (True spline fitting would require scipy)
            alphas = self.metadata.get('alphas', np.linspace(0, 1, len(self.landmarks)))

            # Find surrounding landmarks
            for i in range(len(alphas) - 1):
                if alphas[i] <= alpha <= alphas[i + 1]:
                    local_alpha = (alpha - alphas[i]) / (alphas[i + 1] - alphas[i])
                    return interpolate_vectors(
                        self.landmarks[i],
                        self.landmarks[i + 1],
                        alpha=local_alpha,
                        method="spherical"
                    )

            return self.landmarks[-1] if alpha >= 1.0 else self.landmarks[0]

        elif self.curve_type == "polynomial":
            # Lagrange polynomial interpolation
            alphas = self.metadata.get('alphas', np.linspace(0, 1, len(self.landmarks)))

            result = torch.zeros_like(self.landmarks[0])
            for i, (landmark, alpha_i) in enumerate(zip(self.landmarks, alphas)):
                # Lagrange basis polynomial
                basis = 1.0
                for j, alpha_j in enumerate(alphas):
                    if i != j:
                        basis *= (alpha - alpha_j) / (alpha_i - alpha_j + 1e-8)
                result += basis * landmark

            return result

        else:
            # Fallback: linear interpolation
            return interpolate_vectors(
                self.control_points[0],
                self.control_points[-1],
                alpha=alpha,
                method="linear"
            )

    def apply_geometric_alignment(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Apply via geometric alignment (same as LandmarkPath)."""
        transform = compute_alignment_transform(
            self.landmarks[0],
            self.landmarks[-1],
            new_vec_a,
            new_vec_b
        )

        vec_at_alpha = self.interpolate(alpha)
        return apply_alignment_transform(vec_at_alpha, transform)

    def apply_relative_encoding(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Apply via relative position encoding."""
        vec_at_alpha = self.interpolate(alpha)

        relative_pos = compute_relative_position(
            vec_at_alpha,
            self.landmarks[0],
            self.landmarks[-1]
        )

        return reconstruct_from_relative_position(
            relative_pos,
            new_vec_a,
            new_vec_b
        )

    def apply_direction_magnitude(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Apply via direction field decomposition."""
        # Compute tangent at alpha
        epsilon = 0.01
        vec_before = self.interpolate(max(0.0, alpha - epsilon))
        vec_after = self.interpolate(min(1.0, alpha + epsilon))

        tangent = (vec_after.flatten() - vec_before.flatten()) / (2 * epsilon)

        # Scale tangent to new space
        orig_axis = self.landmarks[-1].flatten() - self.landmarks[0].flatten()
        new_axis = new_vec_b.flatten() - new_vec_a.flatten()
        scale = torch.norm(new_axis) / (torch.norm(orig_axis) + 1e-8)

        # Apply scaled tangent from linearly interpolated position
        base = (1 - alpha) * new_vec_a.flatten() + alpha * new_vec_b.flatten()
        result = base + scale * tangent * 0.5  # Damping factor

        return result.reshape(new_vec_a.shape)


@dataclass
class TangentVectorFieldPath:
    """
    Path represented as tangent vector field + curvature.

    Instead of storing landmarks, stores the direction to move at each
    position along the path, plus curvature information.

    Attributes:
        tangent_vectors: Direction of flow at sample points
        positions: Where each tangent applies (alpha values)
        curvatures: How much path bends at each point
        anchor_start: Original start vector (for reference)
        anchor_end: Original end vector (for reference)
        metadata: Extra information
    """
    tangent_vectors: List[torch.Tensor]
    positions: List[float]
    curvatures: List[float]
    anchor_start: torch.Tensor
    anchor_end: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit_from_landmarks(
        cls,
        landmarks: List[torch.Tensor],
        alphas: List[float]
    ) -> 'TangentVectorFieldPath':
        """
        Compute tangent vector field from landmarks.

        Args:
            landmarks: List of vectors defining path
            alphas: Position of each landmark

        Returns:
            TangentVectorFieldPath instance
        """
        tangents = []
        curvatures = []

        # Compute tangent at each landmark
        for i in range(len(landmarks)):
            if i == 0:
                # Forward difference
                tangent = (landmarks[i + 1].flatten() - landmarks[i].flatten()) / (alphas[i + 1] - alphas[i])
            elif i == len(landmarks) - 1:
                # Backward difference
                tangent = (landmarks[i].flatten() - landmarks[i - 1].flatten()) / (alphas[i] - alphas[i - 1])
            else:
                # Central difference
                tangent = (landmarks[i + 1].flatten() - landmarks[i - 1].flatten()) / (alphas[i + 1] - alphas[i - 1])

            tangents.append(F.normalize(tangent, dim=0))

            # Compute curvature (change in tangent direction)
            if i > 0 and i < len(landmarks) - 1:
                prev_tangent = (landmarks[i].flatten() - landmarks[i - 1].flatten()) / (alphas[i] - alphas[i - 1])
                next_tangent = (landmarks[i + 1].flatten() - landmarks[i].flatten()) / (alphas[i + 1] - alphas[i])
                prev_tangent = F.normalize(prev_tangent, dim=0)
                next_tangent = F.normalize(next_tangent, dim=0)

                # Curvature as angle between consecutive tangents
                curvature = torch.acos(torch.clamp(torch.dot(prev_tangent, next_tangent), -1.0, 1.0)).item()
            else:
                curvature = 0.0

            curvatures.append(curvature)

        return cls(
            tangent_vectors=tangents,
            positions=alphas,
            curvatures=curvatures,
            anchor_start=landmarks[0],
            anchor_end=landmarks[-1],
            metadata={'num_landmarks': len(landmarks)}
        )

    def interpolate_tangent(self, alpha: float) -> Tuple[torch.Tensor, float]:
        """
        Get tangent direction and curvature at position alpha.

        Args:
            alpha: Position along path (0.0 to 1.0)

        Returns:
            Tuple of (tangent_vector, curvature)
        """
        # Find surrounding positions
        for i in range(len(self.positions) - 1):
            if self.positions[i] <= alpha <= self.positions[i + 1]:
                # Interpolate tangent (spherical interpolation)
                local_alpha = (alpha - self.positions[i]) / (self.positions[i + 1] - self.positions[i])

                tangent = interpolate_vectors(
                    self.tangent_vectors[i].unsqueeze(0),
                    self.tangent_vectors[i + 1].unsqueeze(0),
                    alpha=local_alpha,
                    method="spherical"
                ).squeeze(0)

                # Interpolate curvature (linear)
                curvature = (1 - local_alpha) * self.curvatures[i] + local_alpha * self.curvatures[i + 1]

                return tangent, curvature

        # Edge cases
        if alpha <= 0.0:
            return self.tangent_vectors[0], self.curvatures[0]
        else:
            return self.tangent_vectors[-1], self.curvatures[-1]

    def interpolate(self, alpha: float, num_steps: int = 100) -> torch.Tensor:
        """
        Reconstruct vector at position alpha by integrating tangent field.

        Args:
            alpha: Position along path (0.0 to 1.0)
            num_steps: Number of integration steps (higher = more accurate)

        Returns:
            Vector at position alpha
        """
        if alpha <= 0.0:
            return self.anchor_start
        elif alpha >= 1.0:
            return self.anchor_end

        # Euler integration along tangent field
        current_pos = self.anchor_start.flatten().clone()

        # Compute step size
        step_size = alpha / num_steps

        for i in range(num_steps):
            current_alpha = (i + 0.5) * step_size  # Midpoint
            tangent, _ = self.interpolate_tangent(current_alpha)

            # Take step in tangent direction
            # Scale by the total axis length to get correct magnitude
            axis_length = torch.norm(self.anchor_end.flatten() - self.anchor_start.flatten())
            current_pos = current_pos + step_size * axis_length * tangent

        return current_pos.reshape(self.anchor_start.shape)

    def apply_geometric_alignment(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply via geometric alignment of tangent field.

        Transforms tangent vectors to align with new concept pair.
        """
        # Get tangent in original space
        tangent, _ = self.interpolate_tangent(alpha)

        # Compute alignment transform
        transform = compute_alignment_transform(
            self.anchor_start,
            self.anchor_end,
            new_vec_a,
            new_vec_b
        )

        # Transform tangent
        transformed_tangent = transform['rotation_matrix'] @ tangent
        transformed_tangent = F.normalize(transformed_tangent, dim=0)

        # Integrate tangent to get position (simplified Euler integration)
        axis_length = torch.norm(new_vec_b.flatten() - new_vec_a.flatten())
        step_size = alpha * axis_length

        result = new_vec_a.flatten() + step_size * transformed_tangent

        return result.reshape(new_vec_a.shape)

    def apply_relative_encoding(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply via relative tangent encoding.

        Encodes tangent relative to main axis, then reconstructs.
        """
        tangent, _ = self.interpolate_tangent(alpha)

        # Original axis
        orig_axis = self.anchor_end.flatten() - self.anchor_start.flatten()
        orig_axis_norm = F.normalize(orig_axis, dim=0)

        # Decompose tangent into parallel and perpendicular components
        parallel_component = torch.dot(tangent, orig_axis_norm)
        perpendicular = tangent - parallel_component * orig_axis_norm
        perp_magnitude = torch.norm(perpendicular)

        # New axis
        new_axis = new_vec_b.flatten() - new_vec_a.flatten()
        new_axis_norm = F.normalize(new_axis, dim=0)

        # Reconstruct tangent
        if perp_magnitude > 1e-6:
            # Find perpendicular direction in new space (simplified)
            perp_normalized = F.normalize(perpendicular, dim=0)
            new_tangent = parallel_component * new_axis_norm + perp_magnitude * perp_normalized
        else:
            new_tangent = parallel_component * new_axis_norm

        # Integrate
        new_axis_length = torch.norm(new_axis)
        step_size = alpha * new_axis_length

        result = new_vec_a.flatten() + step_size * F.normalize(new_tangent, dim=0)

        return result.reshape(new_vec_a.shape)

    def apply_direction_magnitude(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply via pure direction field transfer.

        This is the most natural application for tangent field representation.
        """
        tangent, curvature = self.interpolate_tangent(alpha)

        # Scale tangent to new space
        orig_axis_length = torch.norm(self.anchor_end.flatten() - self.anchor_start.flatten())
        new_axis_length = torch.norm(new_vec_b.flatten() - new_vec_a.flatten())
        scale = new_axis_length / (orig_axis_length + 1e-8)

        # Apply curvature-weighted integration
        base_position = (1 - alpha) * new_vec_a.flatten() + alpha * new_vec_b.flatten()
        displacement = scale * torch.norm(tangent) * F.normalize(tangent, dim=0)

        # Add curvature effect (more curvature = more deviation from straight line)
        curvature_factor = np.sin(curvature) * 0.5  # Damping

        result = base_position + curvature_factor * displacement

        return result.reshape(new_vec_a.shape)


# ============================================================================
# MULTI-LAYER PATH WRAPPERS
# ============================================================================

@dataclass
class MultiLayerLandmarkPath:
    """
    Wrapper for LandmarkPath that handles multiple layers.

    Stores independent LandmarkPath objects for each layer and provides
    unified interface for multi-layer operations.

    Attributes:
        layer_paths: Dictionary mapping layer indices to LandmarkPath objects
        layer_indices: Sorted list of layer indices
        metadata: Shared metadata (concepts, keywords, etc.)

    Examples:
        >>> # Extract multi-layer path
        >>> ml_path = learn_semantic_path(selfie, "sad", "happy", layer='3:13')
        >>> # Interpolate across all layers
        >>> layer_vecs = ml_path.interpolate(0.5)  # Dict[int, torch.Tensor]
        >>> # Apply to new concept pair
        >>> new_vecs = ml_path.apply_geometric_alignment(sad_vecs, happy_vecs, 0.5)
    """
    layer_paths: Dict[int, LandmarkPath]
    layer_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and sort layer indices."""
        if not self.layer_paths:
            raise ValueError("layer_paths cannot be empty")
        # Ensure layer_indices is sorted
        self.layer_indices = sorted(self.layer_paths.keys())

    def interpolate(
        self,
        alpha: float,
        method: str = "spherical"
    ) -> Dict[int, torch.Tensor]:
        """
        Interpolate at given alpha across all layers.

        Args:
            alpha: Position along path (0.0 to 1.0)
            method: Interpolation method ("spherical" or "linear")

        Returns:
            Dictionary mapping layer index to interpolated vector
        """
        return {
            layer_idx: path.interpolate(alpha, method)
            for layer_idx, path in self.layer_paths.items()
        }

    def apply_geometric_alignment(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """
        Apply geometric alignment independently per layer.

        Args:
            new_vecs_a: Start vectors per layer
            new_vecs_b: End vectors per layer
            alpha: Position along path (0.0 to 1.0)

        Returns:
            Dictionary mapping layer index to transformed vector
        """
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_geometric_alignment(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results

    def apply_relative_encoding(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Apply relative encoding independently per layer."""
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_relative_encoding(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results

    def apply_direction_magnitude(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Apply direction+magnitude independently per layer."""
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_direction_magnitude(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results


@dataclass
class MultiLayerParametricCurvePath:
    """
    Wrapper for ParametricCurvePath that handles multiple layers.

    Stores independent ParametricCurvePath objects for each layer.
    """
    layer_paths: Dict[int, ParametricCurvePath]
    layer_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and sort layer indices."""
        if not self.layer_paths:
            raise ValueError("layer_paths cannot be empty")
        self.layer_indices = sorted(self.layer_paths.keys())

    def interpolate(
        self,
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Interpolate at given alpha across all layers."""
        return {
            layer_idx: path.interpolate(alpha)
            for layer_idx, path in self.layer_paths.items()
        }

    def apply_geometric_alignment(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Apply geometric alignment independently per layer."""
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_geometric_alignment(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results

    def apply_relative_encoding(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Apply relative encoding independently per layer."""
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_relative_encoding(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results

    def apply_direction_magnitude(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Apply direction+magnitude independently per layer."""
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_direction_magnitude(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results


@dataclass
class MultiLayerTangentVectorFieldPath:
    """
    Wrapper for TangentVectorFieldPath that handles multiple layers.

    Stores independent TangentVectorFieldPath objects for each layer.
    """
    layer_paths: Dict[int, TangentVectorFieldPath]
    layer_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and sort layer indices."""
        if not self.layer_paths:
            raise ValueError("layer_paths cannot be empty")
        self.layer_indices = sorted(self.layer_paths.keys())

    def interpolate(
        self,
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Interpolate at given alpha across all layers."""
        return {
            layer_idx: path.interpolate(alpha)
            for layer_idx, path in self.layer_paths.items()
        }

    def apply_geometric_alignment(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Apply geometric alignment independently per layer."""
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_geometric_alignment(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results

    def apply_relative_encoding(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Apply relative encoding independently per layer."""
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_relative_encoding(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results

    def apply_direction_magnitude(
        self,
        new_vecs_a: Dict[int, torch.Tensor],
        new_vecs_b: Dict[int, torch.Tensor],
        alpha: float
    ) -> Dict[int, torch.Tensor]:
        """Apply direction+magnitude independently per layer."""
        results = {}
        for layer_idx, path in self.layer_paths.items():
            if layer_idx not in new_vecs_a or layer_idx not in new_vecs_b:
                raise ValueError(f"Missing vectors for layer {layer_idx}")
            results[layer_idx] = path.apply_direction_magnitude(
                new_vecs_a[layer_idx],
                new_vecs_b[layer_idx],
                alpha
            )
        return results


# ============================================================================
# MULTI-PAIR LEARNING
# ============================================================================

class SemanticPathAggregator:
    """
    Learn universal transformation patterns from multiple concept pairs.

    Aggregates multiple learned paths to extract common semantic structure,
    enabling application to entirely new concept pairs.

    Examples:
        >>> aggregator = SemanticPathAggregator()
        >>> aggregator.add_path(sad_happy_path, ("sad", "happy"))
        >>> aggregator.add_path(angry_calm_path, ("angry", "calm"))
        >>> aggregator.fit()
        >>> universal_path = aggregator.get_universal_path()
    """

    def __init__(self):
        self.paths = []
        self.concept_pairs = []
        self.universal_representation = None

    def add_path(
        self,
        path: Union[LandmarkPath, ParametricCurvePath, TangentVectorFieldPath],
        concept_pair: Tuple[str, str]
    ):
        """
        Add a learned path to the aggregator.

        Args:
            path: Learned path object
            concept_pair: Tuple of (start_concept, end_concept)
        """
        self.paths.append(path)
        self.concept_pairs.append(concept_pair)

    def fit(self, method: str = "direction_statistics"):
        """
        Learn universal transformation pattern from all paths.

        Args:
            method: Aggregation method
                - "direction_statistics": Average tangent directions
                - "curvature_transfer": Average curvature patterns
                - "relative_geometry": Average relative positions
        """
        if method == "direction_statistics":
            self._fit_direction_statistics()
        elif method == "curvature_transfer":
            self._fit_curvature_transfer()
        else:
            self._fit_relative_geometry()

    def _fit_direction_statistics(self):
        """Learn average tangent field from all paths."""
        # Sample paths at common alpha values
        alphas = np.linspace(0, 1, 20)

        # Check if paths are multi-layer by testing first path
        if not self.paths:
            raise ValueError("No paths to aggregate")

        test_vec = self.paths[0].interpolate(0.0)
        is_multilayer = isinstance(test_vec, dict)

        if is_multilayer:
            raise ValueError(
                "Multi-layer paths detected. Please use single-layer paths for aggregation. "
                "Set USE_SINGLE_LAYER=True in script 03_learn_pattern_paths.py to create single-layer paths."
            )

        avg_tangents = []
        avg_curvatures = []

        for alpha in alphas:
            tangents_at_alpha = []
            curvatures_at_alpha = []

            for path in self.paths:
                # Convert all paths to tangent representation
                if isinstance(path, TangentVectorFieldPath):
                    tangent, curvature = path.interpolate_tangent(alpha)
                else:
                    # Approximate tangent by finite difference
                    epsilon = 0.01
                    vec_before = path.interpolate(max(0.0, alpha - epsilon))
                    vec_after = path.interpolate(min(1.0, alpha + epsilon))
                    tangent = (vec_after.flatten() - vec_before.flatten()) / (2 * epsilon)
                    tangent = F.normalize(tangent, dim=0)
                    curvature = 0.0

                tangents_at_alpha.append(tangent)
                curvatures_at_alpha.append(curvature)

            # Average tangents (spherical mean)
            avg_tangent = torch.stack(tangents_at_alpha).mean(dim=0)
            avg_tangent = F.normalize(avg_tangent, dim=0)
            avg_tangents.append(avg_tangent)

            # Average curvature
            avg_curvature = np.mean(curvatures_at_alpha)
            avg_curvatures.append(avg_curvature)

        # Store universal representation
        self.universal_representation = {
            'type': 'tangent_field',
            'tangents': avg_tangents,
            'curvatures': avg_curvatures,
            'positions': alphas.tolist()
        }

    def _fit_curvature_transfer(self):
        """Learn average curvature pattern."""
        # Similar to direction_statistics but focuses on curvature
        self._fit_direction_statistics()
        self.universal_representation['type'] = 'curvature_transfer'

    def _fit_relative_geometry(self):
        """Learn average relative positions."""
        alphas = np.linspace(0, 1, 20)

        # Check if paths are multi-layer by testing first path
        if not self.paths:
            raise ValueError("No paths to aggregate")

        test_vec = self.paths[0].interpolate(0.0)
        is_multilayer = isinstance(test_vec, dict)

        if is_multilayer:
            raise ValueError(
                "Multi-layer paths detected. Please use single-layer paths for aggregation. "
                "Set USE_SINGLE_LAYER=True in script 03_learn_pattern_paths.py to create single-layer paths."
            )

        avg_relative_positions = []

        for alpha in alphas:
            relative_positions = []

            for path in self.paths:
                if isinstance(path, LandmarkPath):
                    start, end = path.landmarks[0], path.landmarks[-1]
                elif isinstance(path, ParametricCurvePath):
                    start, end = path.landmarks[0], path.landmarks[-1]
                else:  # TangentVectorFieldPath
                    start, end = path.anchor_start, path.anchor_end

                vec_at_alpha = path.interpolate(alpha)
                rel_pos = compute_relative_position(vec_at_alpha, start, end)
                relative_positions.append(rel_pos)

            # Average relative positions
            avg_alpha = np.mean([rp['alpha'] for rp in relative_positions])
            avg_perp_dist = np.mean([rp['perpendicular_distance'] for rp in relative_positions])

            avg_relative_positions.append({
                'alpha': avg_alpha,
                'perpendicular_distance': avg_perp_dist
            })

        self.universal_representation = {
            'type': 'relative_geometry',
            'relative_positions': avg_relative_positions,
            'positions': alphas.tolist()
        }

    def apply_universal(
        self,
        new_vec_a: torch.Tensor,
        new_vec_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply learned universal pattern to new concept pair.

        Args:
            new_vec_a: Start vector of new pair
            new_vec_b: End vector of new pair
            alpha: Position along path (0.0 to 1.0)

        Returns:
            Vector following universal pattern
        """
        if self.universal_representation is None:
            raise ValueError("Must call fit() before apply_universal()")

        rep_type = self.universal_representation['type']

        if rep_type in ['tangent_field', 'curvature_transfer']:
            # Find tangent at alpha
            positions = self.universal_representation['positions']
            tangents = self.universal_representation['tangents']

            # Interpolate to find tangent at alpha
            for i in range(len(positions) - 1):
                if positions[i] <= alpha <= positions[i + 1]:
                    local_alpha = (alpha - positions[i]) / (positions[i + 1] - positions[i])
                    tangent = (1 - local_alpha) * tangents[i] + local_alpha * tangents[i + 1]
                    tangent = F.normalize(tangent, dim=0)
                    break
            else:
                tangent = tangents[-1] if alpha >= 1.0 else tangents[0]

            # Apply tangent in new space
            new_axis = new_vec_b.flatten() - new_vec_a.flatten()
            axis_length = torch.norm(new_axis)

            base = (1 - alpha) * new_vec_a.flatten() + alpha * new_vec_b.flatten()
            result = base + 0.3 * axis_length * tangent  # Scale tangent by axis length

            return result.reshape(new_vec_a.shape)

        elif rep_type == 'relative_geometry':
            # Find relative position at alpha
            positions = self.universal_representation['positions']
            rel_positions = self.universal_representation['relative_positions']

            for i in range(len(positions) - 1):
                if positions[i] <= alpha <= positions[i + 1]:
                    local_alpha = (alpha - positions[i]) / (positions[i + 1] - positions[i])

                    interp_alpha = ((1 - local_alpha) * rel_positions[i]['alpha'] +
                                   local_alpha * rel_positions[i + 1]['alpha'])
                    interp_perp = ((1 - local_alpha) * rel_positions[i]['perpendicular_distance'] +
                                  local_alpha * rel_positions[i + 1]['perpendicular_distance'])

                    # Reconstruct using average relative position
                    axis = new_vec_b.flatten() - new_vec_a.flatten()
                    point_on_axis = new_vec_a.flatten() + interp_alpha * axis

                    # Add perpendicular component (simplified)
                    result = point_on_axis + interp_perp * torch.randn_like(point_on_axis) * 0.1

                    return result.reshape(new_vec_a.shape)

            # Edge case
            return (1 - alpha) * new_vec_a + alpha * new_vec_b

        else:
            raise ValueError(f"Unknown representation type: {rep_type}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def learn_semantic_path(
    selfie,
    start_concept: str,
    end_concept: str,
    layer: Union[int, str],
    num_steps: int = 7,
    template: str = "emotion",
    path_type: str = "landmark",
    use_chat_template: bool = True,
    **kwargs
) -> Union[LandmarkPath, ParametricCurvePath, TangentVectorFieldPath,
           MultiLayerLandmarkPath, MultiLayerParametricCurvePath, MultiLayerTangentVectorFieldPath]:
    """
    Learn semantic path between two concepts (high-level convenience function).

    Args:
        selfie: ModelAgnosticSelfie instance
        start_concept: Starting concept (e.g., "sad")
        end_concept: Ending concept (e.g., "happy")
        layer: Layer specification - can be:
            - int: Single layer index (e.g., 7) - returns single-layer path
            - 'all': All layers - returns multi-layer path
            - 'start:end': Range of layers (e.g., '3:13') - returns multi-layer path
            - 'start:end:step': Range with step (e.g., '0:20:2') - returns multi-layer path
        num_steps: Number of intermediate steps
        template: Keyword generation template
            - "model_generated": Use LLM to generate keywords (RECOMMENDED)
            - "emotion": Template-based emotional transitions
            - "intensity": Template-based intensity gradients
            - "custom": Simple template interpolation
        path_type: Type of path representation ("landmark", "parametric", "tangent")
        use_chat_template: Whether to use chat template for extraction
        **kwargs: Additional parameters for path fitting (and temperature for model_generated)

    Returns:
        Learned path object - single-layer or multi-layer depending on layer parameter

    Examples:
        >>> # Single layer (backward compatible)
        >>> path = learn_semantic_path(
        ...     selfie,
        ...     "sad", "happy",
        ...     layer=15,
        ...     num_steps=7,
        ...     template="model_generated",
        ...     path_type="landmark"
        ... )

        >>> # Multi-layer (all layers)
        >>> ml_path = learn_semantic_path(
        ...     selfie,
        ...     "sad", "happy",
        ...     layer='all',
        ...     num_steps=7,
        ...     template="model_generated",
        ...     path_type="landmark"
        ... )

        >>> # Multi-layer (range)
        >>> ml_path = learn_semantic_path(
        ...     selfie,
        ...     "sad", "happy",
        ...     layer='3:13',
        ...     num_steps=7,
        ...     template="emotion",
        ...     path_type="parametric"
        ... )
    """
    # Generate intermediate keywords
    temperature = kwargs.get('temperature', 0.3)
    keywords = generate_intermediate_keywords(
        start_concept,
        end_concept,
        num_steps=num_steps,
        template=template,
        selfie=selfie,  # Pass selfie for model_generated template
        temperature=temperature
    )

    # Extract landmark vectors (returns List or Dict depending on layer spec)
    landmarks = extract_landmark_vectors(
        selfie,
        keywords,
        layer=layer,
        use_chat_template=use_chat_template
    )

    # Alphas (evenly distributed)
    alphas = np.linspace(0.0, 1.0, num_steps).tolist()

    # Check if single-layer or multi-layer based on return type
    is_multilayer = isinstance(landmarks, dict)

    if not is_multilayer:
        # Single-layer path (backward compatible)
        if path_type == "landmark":
            path = LandmarkPath(
                landmarks=landmarks,
                alphas=alphas,
                metadata={
                    'concepts': (start_concept, end_concept),
                    'keywords': keywords,
                    'layer': layer
                }
            )

        elif path_type == "parametric":
            path = ParametricCurvePath.fit_from_landmarks(
                landmarks=landmarks,
                alphas=alphas,
                curve_type=kwargs.get('curve_type', 'bezier'),
                **kwargs
            )
            path.metadata['concepts'] = (start_concept, end_concept)
            path.metadata['keywords'] = keywords
            path.metadata['layer'] = layer

        elif path_type == "tangent":
            path = TangentVectorFieldPath.fit_from_landmarks(
                landmarks=landmarks,
                alphas=alphas,
                **kwargs
            )
            path.metadata['concepts'] = (start_concept, end_concept)
            path.metadata['keywords'] = keywords
            path.metadata['layer'] = layer

        else:
            raise ValueError(f"Unknown path_type: {path_type}. Choose 'landmark', 'parametric', or 'tangent'")

    else:
        # Multi-layer path
        layer_paths = {}
        layer_indices = sorted(landmarks.keys())

        for layer_idx in layer_indices:
            layer_landmarks = landmarks[layer_idx]

            if path_type == "landmark":
                layer_path = LandmarkPath(
                    landmarks=layer_landmarks,
                    alphas=alphas,
                    metadata={
                        'concepts': (start_concept, end_concept),
                        'keywords': keywords,
                        'layer': layer_idx
                    }
                )

            elif path_type == "parametric":
                layer_path = ParametricCurvePath.fit_from_landmarks(
                    landmarks=layer_landmarks,
                    alphas=alphas,
                    curve_type=kwargs.get('curve_type', 'bezier'),
                    **kwargs
                )
                layer_path.metadata['concepts'] = (start_concept, end_concept)
                layer_path.metadata['keywords'] = keywords
                layer_path.metadata['layer'] = layer_idx

            elif path_type == "tangent":
                layer_path = TangentVectorFieldPath.fit_from_landmarks(
                    landmarks=layer_landmarks,
                    alphas=alphas,
                    **kwargs
                )
                layer_path.metadata['concepts'] = (start_concept, end_concept)
                layer_path.metadata['keywords'] = keywords
                layer_path.metadata['layer'] = layer_idx

            else:
                raise ValueError(f"Unknown path_type: {path_type}. Choose 'landmark', 'parametric', or 'tangent'")

            layer_paths[layer_idx] = layer_path

        # Create multi-layer wrapper
        if path_type == "landmark":
            path = MultiLayerLandmarkPath(
                layer_paths=layer_paths,
                layer_indices=layer_indices,
                metadata={
                    'concepts': (start_concept, end_concept),
                    'keywords': keywords,
                    'layer_spec': layer
                }
            )

        elif path_type == "parametric":
            path = MultiLayerParametricCurvePath(
                layer_paths=layer_paths,
                layer_indices=layer_indices,
                metadata={
                    'concepts': (start_concept, end_concept),
                    'keywords': keywords,
                    'layer_spec': layer
                }
            )

        elif path_type == "tangent":
            path = MultiLayerTangentVectorFieldPath(
                layer_paths=layer_paths,
                layer_indices=layer_indices,
                metadata={
                    'concepts': (start_concept, end_concept),
                    'keywords': keywords,
                    'layer_spec': layer
                }
            )

    return path


def interpret_multilayer_path(
    selfie,
    multilayer_path: Union[MultiLayerLandmarkPath, MultiLayerParametricCurvePath, MultiLayerTangentVectorFieldPath],
    alpha: float,
    prompt,
    max_new_tokens: int = 25,
    injection_positions: Union[int, List[int]] = -1,
    injection_strength: float = 1.0,
    injection_mode: str = 'addition'
) -> str:
    """
    Interpret a multi-layer path by injecting vectors from all layers simultaneously.

    Uses nnsight to inject activations into multiple layers at once during generation.

    Args:
        selfie: ModelAgnosticSelfie instance
        multilayer_path: Multi-layer path object
        alpha: Position along path (0.0 to 1.0)
        prompt: InterpretationPrompt object
        max_new_tokens: Maximum tokens to generate
        injection_positions: Token position(s) to inject at (-1 for last token)
        injection_strength: Strength of intervention (default 1.0)
        injection_mode: 'addition' or 'normalized'

    Returns:
        Generated interpretation text

    Examples:
        >>> ml_path = learn_semantic_path(selfie, "sad", "happy", layer='3:13')
        >>> interpretation = interpret_multilayer_path(
        ...     selfie, ml_path, alpha=0.5, prompt=concept_prompt
        ... )
    """
    # Get vectors at alpha for all layers
    layer_vectors = multilayer_path.interpolate(alpha)

    # Import utility function
    from .utils import get_layer_by_path

    # Get prompt text
    formatted_prompt = prompt.get_prompt()

    # Convert single position to list
    if isinstance(injection_positions, int):
        injection_positions = [injection_positions]

    # Ensure all activations are on correct device
    device = selfie.device
    for layer_idx in layer_vectors:
        layer_vectors[layer_idx] = layer_vectors[layer_idx].to(device)

    # Generate with multi-layer injection
    with selfie.model.generate(formatted_prompt, max_new_tokens=max_new_tokens) as tracer:
        # Inject into each layer
        for layer_idx, activation in layer_vectors.items():
            layer = get_layer_by_path(selfie.model, selfie.layer_paths[layer_idx])

            # Get original activations
            original_output = layer.output[0]

            # Get batch size, sequence length, hidden size
            batch_size, seq_len, hidden_size = original_output.shape

            # Expand activation for injection positions
            try:
                activation_expanded = activation.expand(batch_size, len(injection_positions), hidden_size)
            except Exception:
                # Fallback for devices with expand issues
                activation_expanded = activation.repeat(batch_size, len(injection_positions), 1)

            # Apply intervention at each position
            for i, pos in enumerate(injection_positions):
                if injection_mode == 'addition':
                    original_output[:, pos, :] = original_output[:, pos, :] + injection_strength * activation_expanded[:, i, :]
                elif injection_mode == 'normalized':
                    original_output[:, pos, :] = (
                        injection_strength * activation_expanded[:, i, :] +
                        (1 - injection_strength) * original_output[:, pos, :]
                    )

        output_ids = selfie.model.generator.output.save()

    # Decode output
    generated_text = selfie.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the new generated text (remove prompt)
    prompt_text = selfie.model.tokenizer.decode(
        selfie.model.tokenizer.encode(formatted_prompt, add_special_tokens=False),
        skip_special_tokens=True
    )

    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text):].strip()

    return generated_text
