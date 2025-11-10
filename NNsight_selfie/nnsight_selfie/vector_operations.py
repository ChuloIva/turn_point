"""
Vector interpolation and transformation operations for activation vectors.

This module provides advanced vector manipulation techniques including:
- Linear and spherical interpolation (lerp/slerp)
- Vector rotation in concept planes
- Smooth blending between activation vectors

Compatible with ModelAgnosticSelfie for extraction, manipulation, and interpretation.
"""

import torch
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
import math


def interpolate_vectors(
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
    alpha: float = 0.5,
    method: Literal["linear", "spherical"] = "linear"
) -> torch.Tensor:
    """
    Smoothly blend between two activation vectors.

    Args:
        vec_a: First vector (shape: [hidden_dim] or [batch, hidden_dim])
        vec_b: Second vector (same shape as vec_a)
        alpha: Interpolation factor (0.0 = all vec_a, 1.0 = all vec_b)
        method: "linear" for lerp, "spherical" for slerp

    Returns:
        Interpolated vector with same shape as inputs

    Examples:
        # Blend 30% toward "sad" from "happy"
        neutral = interpolate_vectors(happy_vec, sad_vec, alpha=0.3)

        # Spherical interpolation for smooth semantic transitions
        blend = interpolate_vectors(king_vec, queen_vec, alpha=0.5, method="spherical")
    """
    # Ensure vectors are the same shape
    assert vec_a.shape == vec_b.shape, f"Vector shapes must match: {vec_a.shape} vs {vec_b.shape}"

    if method == "linear":
        # Linear interpolation (lerp)
        return (1 - alpha) * vec_a + alpha * vec_b

    # method == "spherical"
    # Spherical linear interpolation (slerp)
    # Flatten to 1D if needed for dot product calculation
    original_shape = vec_a.shape
    vec_a_flat = vec_a.flatten()
    vec_b_flat = vec_b.flatten()

    # Normalize vectors
    vec_a_norm = F.normalize(vec_a_flat.unsqueeze(0), dim=1).squeeze(0)
    vec_b_norm = F.normalize(vec_b_flat.unsqueeze(0), dim=1).squeeze(0)

    # Compute angle between vectors
    dot = torch.dot(vec_a_norm, vec_b_norm)
    dot = torch.clamp(dot, -1.0, 1.0)  # Numerical stability
    omega = torch.acos(dot)

    # Handle parallel vectors (omega near 0)
    if torch.abs(omega) < 1e-6:
        # Vectors are parallel, use linear interpolation
        result = (1 - alpha) * vec_a_flat + alpha * vec_b_flat
    else:
        # Standard slerp formula
        sin_omega = torch.sin(omega)
        result = (torch.sin((1 - alpha) * omega) * vec_a_flat +
                 torch.sin(alpha * omega) * vec_b_flat) / sin_omega

    return result.reshape(original_shape)


def rotate_vector_in_plane(
    vector: torch.Tensor,
    reference_vec: torch.Tensor,
    angle: float
) -> torch.Tensor:
    """
    Rotate a vector in the plane defined by the vector and a reference vector.

    Uses Gram-Schmidt orthogonalization to create an orthonormal basis,
    then rotates within that 2D subspace.

    Args:
        vector: Vector to rotate (shape: [hidden_dim] or [batch, hidden_dim])
        reference_vec: Reference vector defining rotation plane (same shape as vector)
        angle: Rotation angle in radians (π/4 = 45 degrees)

    Returns:
        Rotated vector with same shape as inputs

    Examples:
        # Rotate "happy" concept 45° around "emotional" axis
        rotated = rotate_vector_in_plane(happy_vec, emotional_vec, torch.pi/4)

        # Rotate 180° to find opposite concept
        opposite = rotate_vector_in_plane(concept_vec, reference_vec, torch.pi)
    """
    # Ensure vectors are the same shape
    assert vector.shape == reference_vec.shape, f"Vector shapes must match: {vector.shape} vs {reference_vec.shape}"

    # Flatten to 1D if needed
    original_shape = vector.shape
    vector_flat = vector.flatten()
    reference_flat = reference_vec.flatten()

    # Gram-Schmidt orthogonalization to create orthonormal basis
    # u1 is normalized version of input vector
    u1 = F.normalize(vector_flat.unsqueeze(0), dim=1).squeeze(0)

    # u2 is component of reference_vec orthogonal to u1
    u2_unnorm = reference_flat - torch.dot(reference_flat, u1) * u1

    # Check if vectors are parallel
    if torch.norm(u2_unnorm) < 1e-6:
        raise ValueError("Vector and reference_vec are parallel - cannot define a rotation plane")

    u2 = F.normalize(u2_unnorm.unsqueeze(0), dim=1).squeeze(0)

    # Rotate in the (u1, u2) plane
    rotated = math.cos(angle) * u1 + math.sin(angle) * u2

    # Scale back to original magnitude
    original_magnitude = torch.norm(vector_flat)
    rotated = rotated * original_magnitude

    return rotated.reshape(original_shape)


def multi_vector_interpolation(
    vectors: list[torch.Tensor],
    weights: Optional[list[float]] = None,
    method: Literal["weighted_sum", "sequential_lerp", "centroid"] = "weighted_sum"
) -> torch.Tensor:
    """
    Interpolate between multiple vectors using various methods.

    Args:
        vectors: List of vectors to interpolate (each shape: [hidden_dim])
        weights: Optional weights for each vector (must sum to 1.0 for weighted_sum)
        method:
            - "weighted_sum": Weighted combination of all vectors
            - "sequential_lerp": Sequential pairwise linear interpolation
            - "centroid": Geometric center (equal weights)

    Returns:
        Interpolated vector (shape: [hidden_dim])

    Examples:
        # Blend multiple emotions with custom weights
        complex_emotion = multi_vector_interpolation(
            [happy_vec, sad_vec, angry_vec],
            weights=[0.5, 0.3, 0.2]
        )

        # Find centroid of multiple concepts
        average_concept = multi_vector_interpolation(
            [vec1, vec2, vec3],
            method="centroid"
        )
    """
    assert len(vectors) > 0, "Must provide at least one vector"

    # Verify all vectors have same shape
    reference_shape = vectors[0].shape
    for i, vec in enumerate(vectors):
        assert vec.shape == reference_shape, f"Vector {i} shape {vec.shape} doesn't match {reference_shape}"

    if method == "centroid":
        # Simple average of all vectors
        return torch.stack(vectors).mean(dim=0)

    elif method == "weighted_sum":
        # Weighted combination
        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)

        assert len(weights) == len(vectors), f"Need {len(vectors)} weights, got {len(weights)}"
        assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1.0, got {sum(weights)}"

        result = torch.zeros_like(vectors[0])
        for vec, weight in zip(vectors, weights):
            result += weight * vec
        return result

    # method == "sequential_lerp"
    # Sequential pairwise interpolation
    if weights is None:
        # Evenly distribute interpolation factors
        weights = [1.0 / (len(vectors) - 1)] * (len(vectors) - 1)
    else:
        assert len(weights) == len(vectors) - 1, \
            f"Sequential lerp needs {len(vectors)-1} weights, got {len(weights)}"

    result = vectors[0]
    for i, (next_vec, alpha) in enumerate(zip(vectors[1:], weights)):
        result = interpolate_vectors(result, next_vec, alpha=alpha, method="linear")
    return result


def get_interpolation_path(
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
    num_steps: int = 10,
    method: Literal["linear", "spherical"] = "linear"
) -> list[torch.Tensor]:
    """
    Generate a path of vectors interpolating from vec_a to vec_b.

    Useful for analyzing how concepts smoothly transition in activation space.

    Args:
        vec_a: Start vector
        vec_b: End vector
        num_steps: Number of intermediate steps (including endpoints)
        method: "linear" or "spherical" interpolation

    Returns:
        List of vectors forming interpolation path

    Examples:
        # Generate smooth transition from happy to sad
        path = get_interpolation_path(happy_vec, sad_vec, num_steps=10)

        # Interpret each step to see semantic transition
        for i, vec in enumerate(path):
            interp = selfie.interpret_vectors([vec], prompt, layer)
            print(f"Step {i}: {interp}")
    """
    assert num_steps >= 2, "Need at least 2 steps (start and end)"

    alphas = torch.linspace(0, 1, num_steps)
    path = []

    for alpha in alphas:
        interpolated = interpolate_vectors(vec_a, vec_b, alpha=alpha.item(), method=method)
        path.append(interpolated)

    return path


def project_and_rotate(
    vector: torch.Tensor,
    projection_vec: torch.Tensor,
    rotation_angle: float,
    keep_projection: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Project vector onto another vector, then rotate in the orthogonal plane.

    This combines projection with rotation for more controlled transformations.

    Args:
        vector: Input vector to transform (shape: [hidden_dim] or [batch, hidden_dim])
        projection_vec: Vector to project onto (same shape as vector)
        rotation_angle: Angle to rotate in orthogonal plane (radians)
        keep_projection: If True, add back the projection component

    Returns:
        Tuple of (transformed_vector, info_dict)
        info_dict contains: projection, orthogonal_component, rotated_component

    Examples:
        # Rotate around gender axis while preserving occupation info
        result, info = project_and_rotate(
            doctor_vec,
            gender_vec,
            angle=torch.pi/2,
            keep_projection=True
        )
    """
    # Ensure vectors are the same shape
    assert vector.shape == projection_vec.shape, f"Vector shapes must match: {vector.shape} vs {projection_vec.shape}"

    # Flatten to 1D if needed
    original_shape = vector.shape
    vector_flat = vector.flatten()
    projection_flat = projection_vec.flatten()

    # Normalize projection vector
    proj_norm = F.normalize(projection_flat.unsqueeze(0), dim=1).squeeze(0)

    # Get projection component
    projection_magnitude = torch.dot(vector_flat, proj_norm)
    projection_component = projection_magnitude * proj_norm

    # Get orthogonal component
    orthogonal_component = vector_flat - projection_component

    # Rotate orthogonal component
    if torch.norm(orthogonal_component) > 1e-6:
        # Create arbitrary orthogonal vector for rotation
        # Use cross product approach for 3D-like rotation in high-dim space
        rotation_basis = F.normalize(orthogonal_component.unsqueeze(0), dim=1).squeeze(0)

        # Find another orthogonal direction
        # Try standard basis vectors until we find a non-parallel one
        for i in range(min(10, len(vector_flat))):
            test_vec = torch.zeros_like(vector_flat)
            test_vec[i] = 1.0
            ortho_test = test_vec - torch.dot(test_vec, rotation_basis) * rotation_basis
            if torch.norm(ortho_test) > 1e-6:
                rotation_basis_2 = F.normalize(ortho_test.unsqueeze(0), dim=1).squeeze(0)
                break
        else:
            # Fallback: just scale the orthogonal component
            rotated_component = orthogonal_component * math.cos(rotation_angle)

        # Rotate in the plane
        original_magnitude = torch.norm(orthogonal_component)
        rotated_component = (math.cos(rotation_angle) * rotation_basis +
                           math.sin(rotation_angle) * rotation_basis_2) * original_magnitude
    else:
        rotated_component = orthogonal_component

    # Combine components
    if keep_projection:
        result = projection_component + rotated_component
    else:
        result = rotated_component

    info = {
        'projection': projection_component.reshape(original_shape),
        'orthogonal_component': orthogonal_component.reshape(original_shape),
        'rotated_component': rotated_component.reshape(original_shape),
        'projection_magnitude': projection_magnitude.item()
    }

    return result.reshape(original_shape), info