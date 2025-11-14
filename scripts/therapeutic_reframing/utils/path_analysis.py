"""
Utilities for analyzing geometric properties of semantic paths.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Union
from nnsight_selfie.semantic_path_learning import (
    MultiLayerLandmarkPath,
    MultiLayerParametricCurvePath,
    MultiLayerTangentVectorFieldPath
)


def compute_layer_curvature(path, layer_idx: int, alphas: List[float], epsilon: float = 0.02) -> List[float]:
    """
    Compute curvature at each alpha position for a specific layer.

    Args:
        path: Multi-layer path object
        layer_idx: Layer index to compute curvature for
        alphas: List of alpha values to sample
        epsilon: Step size for finite differences

    Returns:
        List of curvature values (radians) at each alpha
    """
    curvatures = []

    for alpha in alphas[1:-1]:  # Skip endpoints
        # Get layer-specific vectors
        v_before_dict = path.interpolate(max(0.0, alpha - epsilon))
        v_center_dict = path.interpolate(alpha)
        v_after_dict = path.interpolate(min(1.0, alpha + epsilon))

        v_before = v_before_dict[layer_idx].flatten()
        v_center = v_center_dict[layer_idx].flatten()
        v_after = v_after_dict[layer_idx].flatten()

        # Compute tangent vectors
        t1 = v_center - v_before
        t2 = v_after - v_center

        # Normalize
        t1 = torch.nn.functional.normalize(t1, dim=0)
        t2 = torch.nn.functional.normalize(t2, dim=0)

        # Angle between tangents (curvature)
        dot_product = torch.dot(t1, t2).item()
        dot_product = max(-1.0, min(1.0, dot_product))  # Clip for numerical stability
        angle = np.arccos(dot_product)

        curvatures.append(angle)

    # Pad endpoints with 0
    return [0.0] + curvatures + [0.0]


def compute_all_layer_curvatures(path, layer_indices: List[int],
                                 alphas: List[float]) -> Dict[int, List[float]]:
    """
    Compute curvature profiles for all layers in a path.

    Returns:
        Dict[layer_idx -> list of curvature values]
    """
    return {
        layer_idx: compute_layer_curvature(path, layer_idx, alphas)
        for layer_idx in layer_indices
    }


def compute_semantic_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute Euclidean distance between two vectors."""
    return torch.norm(vec_b - vec_a).item()


def compute_path_length(path, layer_idx: int, alphas: List[float]) -> float:
    """
    Compute total path length for a specific layer.

    Args:
        path: Multi-layer path object
        layer_idx: Layer index
        alphas: List of alpha values to sample path

    Returns:
        Total path length (sum of segment distances)
    """
    total_length = 0.0

    for i in range(len(alphas) - 1):
        v1_dict = path.interpolate(alphas[i])
        v2_dict = path.interpolate(alphas[i + 1])

        v1 = v1_dict[layer_idx]
        v2 = v2_dict[layer_idx]

        segment_length = torch.norm(v2 - v1).item()
        total_length += segment_length

    return total_length


def compute_geodesic_efficiency(semantic_distance: float, path_length: float) -> float:
    """
    Compute how "straight" a path is.

    Returns:
        Efficiency ratio (0-1), where 1 = perfectly straight
    """
    if path_length == 0:
        return 0.0
    return semantic_distance / path_length


def compute_layer_distance_metrics(path, layer_idx: int, alphas: List[float]) -> Dict[str, float]:
    """
    Compute all distance-related metrics for a specific layer.

    Returns:
        Dict with 'semantic_distance', 'path_length', 'geodesic_efficiency'
    """
    # Get start and end points
    start_dict = path.interpolate(0.0)
    end_dict = path.interpolate(1.0)

    start_vec = start_dict[layer_idx]
    end_vec = end_dict[layer_idx]

    semantic_dist = compute_semantic_distance(start_vec, end_vec)
    path_len = compute_path_length(path, layer_idx, alphas)
    efficiency = compute_geodesic_efficiency(semantic_dist, path_len)

    return {
        'semantic_distance': semantic_dist,
        'path_length': path_len,
        'geodesic_efficiency': efficiency
    }


def compute_landmark_accuracy(path, layer_idx: int, landmark_vec: torch.Tensor, alpha: float = 0.5) -> float:
    """
    Compute distance between path interpolation at alpha and actual landmark vector.

    Args:
        path: Multi-layer path
        layer_idx: Layer index
        landmark_vec: Actual landmark activation at alpha (e.g., transformed example)
        alpha: Position to check (default 0.5 for middle landmark)

    Returns:
        Distance between interpolated and actual landmark
    """
    interpolated_dict = path.interpolate(alpha)
    interpolated_vec = interpolated_dict[layer_idx]

    return compute_semantic_distance(interpolated_vec, landmark_vec)


def compute_curvature_stats(curvatures: List[float]) -> Dict[str, float]:
    """
    Compute statistical summary of curvature values.

    Returns:
        Dict with 'mean', 'max', 'variance', 'total_curvature'
    """
    arr = np.array(curvatures)

    return {
        'mean_curvature': float(np.mean(arr)),
        'max_curvature': float(np.max(arr)),
        'variance_curvature': float(np.var(arr)),
        'total_curvature': float(np.sum(arr))
    }


def compute_reframing_difficulty(
    semantic_distance: float,
    mean_curvature: float,
    max_curvature: float,
    geodesic_efficiency: float,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute composite reframing difficulty score.

    Args:
        semantic_distance: Normalized semantic distance
        mean_curvature: Normalized mean curvature
        max_curvature: Normalized max curvature
        geodesic_efficiency: Geodesic efficiency (0-1)
        weights: Optional custom weights (default: distance=0.4, mean=0.3, max=0.2, efficiency=0.1)

    Returns:
        Difficulty score (higher = harder to reframe)
    """
    if weights is None:
        weights = {
            'distance': 0.4,
            'mean_curvature': 0.3,
            'max_curvature': 0.2,
            'efficiency': 0.1
        }

    difficulty = (
        weights['distance'] * semantic_distance +
        weights['mean_curvature'] * mean_curvature +
        weights['max_curvature'] * max_curvature +
        weights['efficiency'] * (1.0 - geodesic_efficiency)
    )

    return difficulty


def compute_comprehensive_path_analysis(
    path,
    layer_indices: List[int],
    fine_alphas: List[float],
    landmark_vecs: Dict[int, torch.Tensor] = None
) -> Dict:
    """
    Compute all geometric properties for a path across all layers.

    Args:
        path: Multi-layer path object
        layer_indices: List of layer indices to analyze
        fine_alphas: Fine-grained alpha samples for curvature/length
        landmark_vecs: Optional dict of actual landmark vectors per layer (for accuracy)

    Returns:
        Comprehensive analysis dict with per-layer and aggregate metrics
    """
    analysis = {
        'per_layer': {},
        'aggregate': {}
    }

    # Per-layer analysis
    for layer_idx in layer_indices:
        # Curvature
        curvatures = compute_layer_curvature(path, layer_idx, fine_alphas)
        curv_stats = compute_curvature_stats(curvatures)

        # Distance metrics
        dist_metrics = compute_layer_distance_metrics(path, layer_idx, fine_alphas)

        # Landmark accuracy (if provided)
        landmark_acc = None
        if landmark_vecs is not None and layer_idx in landmark_vecs:
            landmark_acc = compute_landmark_accuracy(path, layer_idx, landmark_vecs[layer_idx])

        analysis['per_layer'][layer_idx] = {
            'curvatures': curvatures,
            'curvature_stats': curv_stats,
            'distance_metrics': dist_metrics,
            'landmark_accuracy': landmark_acc
        }

    # Aggregate across layers
    all_semantic_dists = [analysis['per_layer'][l]['distance_metrics']['semantic_distance']
                          for l in layer_indices]
    all_mean_curvs = [analysis['per_layer'][l]['curvature_stats']['mean_curvature']
                      for l in layer_indices]
    all_max_curvs = [analysis['per_layer'][l]['curvature_stats']['max_curvature']
                     for l in layer_indices]
    all_efficiencies = [analysis['per_layer'][l]['distance_metrics']['geodesic_efficiency']
                        for l in layer_indices]

    analysis['aggregate'] = {
        'mean_semantic_distance': float(np.mean(all_semantic_dists)),
        'mean_mean_curvature': float(np.mean(all_mean_curvs)),
        'mean_max_curvature': float(np.mean(all_max_curvs)),
        'mean_geodesic_efficiency': float(np.mean(all_efficiencies)),
        'layer_consistency': {
            'semantic_distance_std': float(np.std(all_semantic_dists)),
            'mean_curvature_std': float(np.std(all_mean_curvs)),
            'geodesic_efficiency_std': float(np.std(all_efficiencies))
        }
    }

    return analysis
