
# Example usage of cluster_analysis_utils.py
# This shows how to use the utility after running the UMAP analysis notebook

import sys
from pathlib import Path

# Add the project directory to path if needed
# sys.path.append('/path/to/your/project')

from cluster_analysis_utils import (
    analyze_clustering_from_notebook_results,
    analyze_3d_clustering_from_notebook_results,
    ClusterAnalyzer
)

# Set up paths
base_path = Path("/Users/ivanculo/Desktop/Projects/turn_point")
activations_dir = base_path / "activations"
metadata_path = base_path / "data" / "final" / "enriched_metadata.json"

# Example 1: Analyze standard clustering results
print("=" * 80)
print("🔍 ANALYZING STANDARD CLUSTERING RESULTS")
print("=" * 80)

# Assuming you have results_all_tokens_all_samples from the notebook
analysis = analyze_clustering_from_notebook_results(
    results_all_tokens_all_samples,
    activations_dir=activations_dir,
    metadata_path=metadata_path,
    top_k=3
)

print("Top clusters by activation magnitude:")
for state, clusters in analysis['top_clusters'].items():
    print(f"
{state} state:")
    for i, cluster in enumerate(clusters):
        print(f"  #{i+1}: Cluster {cluster['cluster_id']} - "
              f"Magnitude: {cluster['activation_magnitude']:.2f}, "
              f"Size: {cluster['size']}")

print(f"
Activation filter created with:")
for state, filter_info in analysis['activation_filter'].items():
    print(f"  {state}: {filter_info['activation_count']} activations")


# Example 2: Analyze 3D clustering results (UMAP-first approach)
# print("
# " + "=" * 80)
# print("🌌 ANALYZING 3D CLUSTERING RESULTS")
# print("=" * 80)

# Assuming you have results_umap_first from the notebook
# enhanced_analysis = analyze_3d_clustering_from_notebook_results(
#     results_umap_first,
#     activations_dir=activations_dir,
#     metadata_path=metadata_path,
#     top_k=2
# )

# print("Directionally extreme clusters:")
# for direction, clusters in enhanced_analysis['directional_extremes'].items():
#     print(f"
{direction}:")
#     for i, cluster in enumerate(clusters):
#         print(f"  #{i+1}: {cluster['state']} Cluster {cluster['cluster_id']} - "
#               f"Coord: {cluster['coordinate_value']:.2f}, "
#               f"Size: {cluster['size']}")

# print(f"
Directional filters created:")
# for direction, filter_info in enhanced_analysis['directional_filter'].items():
#     if filter_info['activation_count'] > 0:
#         print(f"  {direction}: {filter_info['activation_count']} activations "
#               f"from states: {filter_info['states']}")


# Example 3: Direct use of ClusterAnalyzer class
print("
" + "=" * 80) 
print("🔧 DIRECT CLUSTER ANALYZER USAGE")
print("=" * 80)

# analyzer = ClusterAnalyzer(activations_dir, metadata_path)

# # You can run individual analysis steps:
# pattern_name = 'Executive Fatigue & Avolition'

# # Step 1: Analyze cluster activations
# cluster_analysis = analyzer.analyze_cluster_activations(
#     results_all_tokens_all_samples['clustering_results'],
#     pattern_name,
#     layer=17
# )

# # Step 2: Find top clusters
# top_clusters = analyzer.find_highest_activating_clusters(
#     cluster_analysis, 
#     top_k=3
# )

# # Step 3: Extract activation IDs  
# activation_ids = analyzer.extract_activation_ids(
#     top_clusters,
#     pattern_name,
#     layer=17,
#     all_tokens=True,
#     sample_indices=None  # Uses all samples for the pattern
# )

# # Step 4: Create filter
# activation_filter = analyzer.create_activation_filter(activation_ids)

# # Use the filter to subset your data as needed
# print("Filter ready for use!")


# Example 4: Using filters to subset activation data
print("
" + "=" * 80)
print("🎯 USING FILTERS TO SUBSET DATA")
print("=" * 80)

# Example of how you might use the activation filters:
# def apply_cluster_filter(activations, activation_filter, state_name):
#     """
#     Apply a cluster-based filter to activation data.
#     
#     Args:
#         activations: Original activation tensor [samples, tokens, features]
#         activation_filter: Filter from create_activation_filter
#         state_name: Which state filter to apply ('Negative', 'Positive', 'Transition')
#     
#     Returns:
#         Filtered activation data
#     """
#     if state_name not in activation_filter:
#         return None
#     
#     filter_info = activation_filter[state_name]
#     sample_token_pairs = filter_info['sample_token_pairs']
#     
#     # Extract the specific activations
#     filtered_activations = []
#     for sample_id, token_pos in sample_token_pairs:
#         if token_pos == -1:  # Last token
#             activation = activations[sample_id, -1, :]
#         else:
#             activation = activations[sample_id, token_pos, :]
#         filtered_activations.append(activation)
#     
#     return torch.stack(filtered_activations) if filtered_activations else None

print("Complete example code generated!")
print("Uncomment the relevant sections to run with your actual data.")
