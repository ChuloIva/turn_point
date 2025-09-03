"""
Cluster Analysis Utilities

This module provides tools to analyze UMAP clustering results and extract 
the highest activating clusters from neural activation data. It traces clusters 
back to specific activation IDs for use as filters.

Key functionality:
- Analyze cluster activation magnitudes across different states
- Find highest activating clusters in 3D embedding space
- Trace cluster membership back to original activation indices
- Extract activation IDs for cluster-based filtering
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import hdbscan
from sklearn.preprocessing import StandardScaler


class ClusterAnalyzer:
    """
    Utility class for analyzing UMAP clustering results and extracting
    highest activating clusters for use as activation filters.
    """
    
    def __init__(self, 
                 activations_dir: Union[str, Path] = None,
                 metadata_path: Union[str, Path] = None,
                 device: str = 'auto'):
        """
        Initialize the cluster analyzer.
        
        Args:
            activations_dir: Path to directory containing activation files
            metadata_path: Path to enriched metadata JSON file
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        self.activations_dir = Path(activations_dir) if activations_dir else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        self.activations_data = {}
        self.metadata = None
        self.pattern_indices = {}
        
        # Load data if paths provided
        if self.activations_dir and self.metadata_path:
            self.load_data()
    
    def load_data(self):
        """Load activation data and metadata."""
        print("Loading activation data and metadata...")
        
        # Load activations
        self.activations_data['negative'] = torch.load(
            self.activations_dir / "activations_8ff00d963316212d.pt", 
            map_location=self.device
        )
        self.activations_data['positive'] = torch.load(
            self.activations_dir / "activations_e5ad16e9b3c33c9b.pt", 
            map_location=self.device
        )
        self.activations_data['transition'] = torch.load(
            self.activations_dir / "activations_332f24de2a3f82ff.pt", 
            map_location=self.device
        )
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Create pattern indices mapping
        self.pattern_indices = {}
        for i, entry in enumerate(self.metadata):
            pattern_name = entry['bad_good_narratives_match']['cognitive_pattern_name_from_bad_good']
            if pattern_name not in self.pattern_indices:
                self.pattern_indices[pattern_name] = []
            self.pattern_indices[pattern_name].append(i)
        
        print(f"Loaded data for {len(self.metadata)} examples")
        print(f"Found {len(self.pattern_indices)} cognitive patterns")
        print(f"Using device: {self.device}")
    
    def analyze_cluster_activations(self, 
                                  clustering_results: Dict,
                                  pattern_name: str,
                                  layer: int = 17) -> Dict:
        """
        Analyze the activation magnitudes of clusters across states.
        
        Args:
            clustering_results: Results from UMAP clustering analysis
            pattern_name: Name of the cognitive pattern analyzed
            layer: Layer number to analyze
            
        Returns:
            Dictionary with cluster activation statistics
        """
        print(f"\n🔍 Analyzing cluster activations for {pattern_name}")
        print(f"   Layer: {layer}")
        
        analysis_results = {}
        
        for state_name, cluster_info in clustering_results.items():
            print(f"\n📊 Analyzing {state_name} state clusters:")
            
            data = cluster_info['data']  # Original activation data
            cluster_labels = cluster_info['cluster_labels']
            
            # Analyze each cluster
            unique_labels = np.unique(cluster_labels)
            cluster_stats = {}
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                # Get data points for this cluster
                cluster_mask = cluster_labels == label
                cluster_data = data[cluster_mask]
                
                # Calculate statistics
                mean_activation = np.mean(cluster_data, axis=0)
                max_activation = np.max(cluster_data, axis=0)
                activation_magnitude = np.linalg.norm(mean_activation)
                max_magnitude = np.linalg.norm(max_activation)
                
                cluster_stats[int(label)] = {
                    'size': int(np.sum(cluster_mask)),
                    'mean_activation': mean_activation,
                    'max_activation': max_activation,
                    'activation_magnitude': float(activation_magnitude),
                    'max_magnitude': float(max_magnitude),
                    'cluster_indices': np.where(cluster_mask)[0].tolist()
                }
                
                print(f"   Cluster {label}: {int(np.sum(cluster_mask))} points, "
                      f"magnitude: {activation_magnitude:.2f}")
            
            analysis_results[state_name] = {
                'cluster_stats': cluster_stats,
                'total_points': len(cluster_labels),
                'n_clusters': cluster_info['n_clusters']
            }
        
        return analysis_results
    
    def find_highest_activating_clusters(self, 
                                       cluster_analysis: Dict,
                                       top_k: int = 3) -> Dict:
        """
        Find the highest activating clusters across all states.
        
        Args:
            cluster_analysis: Results from analyze_cluster_activations
            top_k: Number of top clusters to return per state
            
        Returns:
            Dictionary with top clusters per state
        """
        print(f"\n🎯 Finding top {top_k} highest activating clusters per state:")
        
        top_clusters = {}
        
        for state_name, analysis in cluster_analysis.items():
            cluster_stats = analysis['cluster_stats']
            
            if not cluster_stats:
                print(f"   {state_name}: No clusters found")
                top_clusters[state_name] = []
                continue
            
            # Sort clusters by activation magnitude
            sorted_clusters = sorted(
                cluster_stats.items(),
                key=lambda x: x[1]['activation_magnitude'],
                reverse=True
            )
            
            # Get top k clusters
            top_k_actual = min(top_k, len(sorted_clusters))
            top_clusters[state_name] = []
            
            print(f"   {state_name} state:")
            for i in range(top_k_actual):
                cluster_id, stats = sorted_clusters[i]
                top_clusters[state_name].append({
                    'cluster_id': cluster_id,
                    'size': stats['size'],
                    'activation_magnitude': stats['activation_magnitude'],
                    'cluster_indices': stats['cluster_indices']
                })
                
                print(f"     #{i+1}: Cluster {cluster_id} "
                      f"(size: {stats['size']}, magnitude: {stats['activation_magnitude']:.2f})")
        
        return top_clusters
    
    def analyze_clusters_in_3d_space(self,
                                   clustering_results: Dict,
                                   pattern_name: str) -> Dict:
        """
        Analyze clusters in 3D UMAP embedding space to find directional patterns.
        This is specifically for UMAP-first analysis results that include 3D embeddings.
        
        Args:
            clustering_results: Results from UMAP-first clustering analysis
            pattern_name: Name of the cognitive pattern
            
        Returns:
            Dictionary with 3D spatial analysis of clusters
        """
        print(f"\n🌌 Analyzing clusters in 3D UMAP space for {pattern_name}")
        
        spatial_analysis = {}
        
        for state_name, cluster_info in clustering_results.items():
            if 'embedding_3d' not in cluster_info:
                print(f"   ⚠️  {state_name}: No 3D embedding found, skipping spatial analysis")
                continue
            
            print(f"\n📊 {state_name} state 3D spatial analysis:")
            
            embedding_3d = cluster_info['embedding_3d']
            cluster_labels = cluster_info['cluster_labels']
            
            # Analyze each cluster's position in 3D space
            unique_labels = np.unique(cluster_labels)
            cluster_3d_stats = {}
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                # Get 3D coordinates for this cluster
                cluster_mask = cluster_labels == label
                cluster_coords = embedding_3d[cluster_mask]
                
                # Calculate 3D statistics
                centroid = np.mean(cluster_coords, axis=0)
                std_dev = np.std(cluster_coords, axis=0)
                min_coords = np.min(cluster_coords, axis=0)
                max_coords = np.max(cluster_coords, axis=0)
                
                # Calculate distances from origin and spread
                distances_from_origin = np.linalg.norm(cluster_coords, axis=1)
                mean_distance = np.mean(distances_from_origin)
                
                # Determine dominant direction
                abs_centroid = np.abs(centroid)
                dominant_axis = np.argmax(abs_centroid)
                dominant_direction = ['X', 'Y', 'Z'][dominant_axis]
                dominant_value = centroid[dominant_axis]
                
                cluster_3d_stats[int(label)] = {
                    'size': int(np.sum(cluster_mask)),
                    'centroid': centroid.tolist(),
                    'std_dev': std_dev.tolist(),
                    'min_coords': min_coords.tolist(),
                    'max_coords': max_coords.tolist(),
                    'mean_distance_from_origin': float(mean_distance),
                    'dominant_axis': dominant_axis,
                    'dominant_direction': dominant_direction,
                    'dominant_value': float(dominant_value),
                    'cluster_indices': np.where(cluster_mask)[0].tolist()
                }
                
                print(f"   Cluster {label}: centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
                      f"dominant={dominant_direction}({dominant_value:.2f})")
            
            spatial_analysis[state_name] = {
                'cluster_3d_stats': cluster_3d_stats,
                'embedding_shape': embedding_3d.shape,
                'total_clusters': len(cluster_3d_stats)
            }
        
        return spatial_analysis
    
    def find_extreme_directional_clusters(self,
                                        spatial_analysis: Dict,
                                        top_k: int = 2) -> Dict:
        """
        Find clusters that are most extreme in each 3D direction (X, Y, Z) 
        across all states.
        
        Args:
            spatial_analysis: Results from analyze_clusters_in_3d_space
            top_k: Number of most extreme clusters per direction
            
        Returns:
            Dictionary with most extreme clusters per direction
        """
        print(f"\n🎯 Finding top {top_k} most extreme clusters in each 3D direction:")
        
        # Collect all clusters across states with their 3D positions
        all_clusters = []
        for state_name, analysis in spatial_analysis.items():
            for cluster_id, stats in analysis['cluster_3d_stats'].items():
                cluster_entry = {
                    'state': state_name,
                    'cluster_id': cluster_id,
                    'centroid': np.array(stats['centroid']),
                    'size': stats['size'],
                    'cluster_indices': stats['cluster_indices']
                }
                all_clusters.append(cluster_entry)
        
        if not all_clusters:
            print("   No clusters found for directional analysis")
            return {}
        
        directional_extremes = {}
        directions = ['X', 'Y', 'Z']
        
        for axis_idx, direction in enumerate(directions):
            print(f"\n   📍 {direction}-axis analysis:")
            
            # Sort by coordinate value in this direction
            positive_extreme = sorted(all_clusters, 
                                    key=lambda x: x['centroid'][axis_idx], 
                                    reverse=True)[:top_k]
            negative_extreme = sorted(all_clusters, 
                                    key=lambda x: x['centroid'][axis_idx])[:top_k]
            
            directional_extremes[f'{direction}_positive'] = []
            directional_extremes[f'{direction}_negative'] = []
            
            print(f"     Most positive {direction}:")
            for i, cluster in enumerate(positive_extreme):
                coord_val = cluster['centroid'][axis_idx]
                directional_extremes[f'{direction}_positive'].append({
                    'state': cluster['state'],
                    'cluster_id': cluster['cluster_id'],
                    'coordinate_value': float(coord_val),
                    'size': cluster['size'],
                    'centroid': cluster['centroid'].tolist(),
                    'cluster_indices': cluster['cluster_indices']
                })
                print(f"       #{i+1}: {cluster['state']} Cluster {cluster['cluster_id']} "
                      f"(coord: {coord_val:.2f}, size: {cluster['size']})")
            
            print(f"     Most negative {direction}:")
            for i, cluster in enumerate(negative_extreme):
                coord_val = cluster['centroid'][axis_idx]
                directional_extremes[f'{direction}_negative'].append({
                    'state': cluster['state'],
                    'cluster_id': cluster['cluster_id'],
                    'coordinate_value': float(coord_val),
                    'size': cluster['size'],
                    'centroid': cluster['centroid'].tolist(),
                    'cluster_indices': cluster['cluster_indices']
                })
                print(f"       #{i+1}: {cluster['state']} Cluster {cluster['cluster_id']} "
                      f"(coord: {coord_val:.2f}, size: {cluster['size']})")
        
        return directional_extremes
    
    def extract_activation_ids(self,
                             top_clusters: Dict,
                             pattern_name: str,
                             layer: int = 17,
                             all_tokens: bool = True,
                             sample_indices: Optional[List[int]] = None) -> Dict:
        """
        Extract activation IDs that correspond to the highest activating clusters.
        
        Args:
            top_clusters: Results from find_highest_activating_clusters
            pattern_name: Name of the cognitive pattern
            layer: Layer number
            all_tokens: Whether analysis used all tokens or just last token
            sample_indices: Specific sample indices used (None for all samples)
            
        Returns:
            Dictionary mapping states to activation IDs for filtering
        """
        print(f"\n📋 Extracting activation IDs for cluster filtering:")
        print(f"   Pattern: {pattern_name}")
        print(f"   Layer: {layer}")
        print(f"   Token strategy: {'All tokens' if all_tokens else 'Last token only'}")
        
        if pattern_name not in self.pattern_indices:
            raise ValueError(f"Pattern '{pattern_name}' not found in metadata")
        
        # Get sample indices
        if sample_indices is None:
            sample_indices = self.pattern_indices[pattern_name]
        
        activation_ids = {}
        
        for state_name, clusters in top_clusters.items():
            if not clusters:
                activation_ids[state_name] = []
                continue
            
            state_activation_ids = []
            
            # Load the appropriate activation data
            state_key = f"{state_name.lower()}_layer_{layer}"
            if state_key in self.activations_data[state_name.lower()]:
                state_data = self.activations_data[state_name.lower()][state_key][sample_indices]
                
                print(f"\n   {state_name} state:")
                print(f"     Data shape: {state_data.shape}")
                
                for cluster_info in clusters:
                    cluster_indices = cluster_info['cluster_indices']
                    cluster_id = cluster_info['cluster_id']
                    
                    # Convert flat indices back to (sample, token) coordinates
                    if all_tokens:
                        # For all tokens: cluster_indices are flat indices across all tokens
                        n_samples = len(sample_indices)
                        tokens_per_sample = state_data.shape[1]
                        
                        activation_coords = []
                        for flat_idx in cluster_indices:
                            sample_idx = flat_idx // tokens_per_sample
                            token_idx = flat_idx % tokens_per_sample
                            
                            if sample_idx < n_samples:
                                original_sample_id = sample_indices[sample_idx]
                                activation_coords.append({
                                    'sample_id': original_sample_id,
                                    'token_position': token_idx,
                                    'flat_index': flat_idx,
                                    'activation_magnitude': cluster_info['activation_magnitude']
                                })
                    else:
                        # For last token only: cluster_indices are sample indices
                        activation_coords = []
                        for sample_idx in cluster_indices:
                            if sample_idx < len(sample_indices):
                                original_sample_id = sample_indices[sample_idx]
                                activation_coords.append({
                                    'sample_id': original_sample_id,
                                    'token_position': -1,  # Last token
                                    'flat_index': sample_idx,
                                    'activation_magnitude': cluster_info['activation_magnitude']
                                })
                    
                    state_activation_ids.extend(activation_coords)
                    
                    print(f"     Cluster {cluster_id}: {len(activation_coords)} activations")
            
            activation_ids[state_name] = state_activation_ids
            print(f"   Total {state_name} activations: {len(state_activation_ids)}")
        
        return activation_ids
    
    def create_activation_filter(self, activation_ids: Dict) -> Dict:
        """
        Create a filter structure that can be used to subset activations.
        
        Args:
            activation_ids: Results from extract_activation_ids
            
        Returns:
            Filter dictionary with sample and token masks
        """
        print(f"\n🔧 Creating activation filter structure:")
        
        filter_structure = {}
        
        for state_name, activations in activation_ids.items():
            if not activations:
                filter_structure[state_name] = {
                    'sample_ids': [],
                    'token_positions': [],
                    'sample_token_pairs': []
                }
                continue
            
            sample_ids = [act['sample_id'] for act in activations]
            token_positions = [act['token_position'] for act in activations]
            sample_token_pairs = [(act['sample_id'], act['token_position']) for act in activations]
            
            filter_structure[state_name] = {
                'sample_ids': list(set(sample_ids)),  # Unique sample IDs
                'token_positions': list(set(token_positions)),  # Unique token positions  
                'sample_token_pairs': sample_token_pairs,  # Exact (sample, token) pairs
                'activation_count': len(activations)
            }
            
            print(f"   {state_name}: {len(set(sample_ids))} unique samples, "
                  f"{len(activations)} total activations")
        
        return filter_structure
    
    def extract_directional_activation_ids(self,
                                         directional_extremes: Dict,
                                         pattern_name: str,
                                         layer: int = 17,
                                         all_tokens: bool = True,
                                         sample_indices: Optional[List[int]] = None) -> Dict:
        """
        Extract activation IDs for directionally extreme clusters.
        
        Args:
            directional_extremes: Results from find_extreme_directional_clusters
            pattern_name: Name of the cognitive pattern
            layer: Layer number
            all_tokens: Whether analysis used all tokens or just last token
            sample_indices: Specific sample indices used
            
        Returns:
            Dictionary mapping directions to activation IDs
        """
        print(f"\n🧭 Extracting activation IDs for directionally extreme clusters:")
        print(f"   Pattern: {pattern_name}")
        print(f"   Layer: {layer}")
        
        if sample_indices is None:
            sample_indices = self.pattern_indices[pattern_name]
        
        directional_activation_ids = {}
        
        for direction_key, clusters in directional_extremes.items():
            if not clusters:
                directional_activation_ids[direction_key] = []
                continue
            
            direction_activations = []
            
            print(f"\n   {direction_key}:")
            for cluster_info in clusters:
                state_name = cluster_info['state']
                cluster_indices = cluster_info['cluster_indices']
                
                # Load the appropriate activation data
                state_key = f"{state_name.lower()}_layer_{layer}"
                if state_key in self.activations_data[state_name.lower()]:
                    state_data = self.activations_data[state_name.lower()][state_key][sample_indices]
                    
                    # Convert flat indices to activation coordinates
                    if all_tokens:
                        n_samples = len(sample_indices)
                        tokens_per_sample = state_data.shape[1]
                        
                        for flat_idx in cluster_indices:
                            sample_idx = flat_idx // tokens_per_sample
                            token_idx = flat_idx % tokens_per_sample
                            
                            if sample_idx < n_samples:
                                original_sample_id = sample_indices[sample_idx]
                                direction_activations.append({
                                    'sample_id': original_sample_id,
                                    'token_position': token_idx,
                                    'flat_index': flat_idx,
                                    'state': state_name,
                                    'cluster_id': cluster_info['cluster_id'],
                                    'coordinate_value': cluster_info['coordinate_value']
                                })
                    else:
                        for sample_idx in cluster_indices:
                            if sample_idx < len(sample_indices):
                                original_sample_id = sample_indices[sample_idx]
                                direction_activations.append({
                                    'sample_id': original_sample_id,
                                    'token_position': -1,  # Last token
                                    'flat_index': sample_idx,
                                    'state': state_name,
                                    'cluster_id': cluster_info['cluster_id'],
                                    'coordinate_value': cluster_info['coordinate_value']
                                })
                    
                    print(f"     {state_name} Cluster {cluster_info['cluster_id']}: "
                          f"{len(cluster_indices)} activations")
            
            directional_activation_ids[direction_key] = direction_activations
            print(f"   Total {direction_key} activations: {len(direction_activations)}")
        
        return directional_activation_ids

    def analyze_umap_clustering_result(self,
                                     clustering_results: Dict,
                                     pattern_name: str,
                                     layer: int = 17,
                                     all_tokens: bool = True,
                                     sample_indices: Optional[List[int]] = None,
                                     top_k: int = 3) -> Dict:
        """
        Complete pipeline to analyze clustering results and extract activation filters.
        
        Args:
            clustering_results: UMAP clustering results dictionary
            pattern_name: Name of cognitive pattern
            layer: Layer number analyzed
            all_tokens: Whether all tokens were used
            sample_indices: Sample indices used in analysis
            top_k: Number of top clusters per state
            
        Returns:
            Complete analysis with activation filters
        """
        print(f"\n{'='*80}")
        print(f"🔬 COMPLETE CLUSTER ANALYSIS PIPELINE")
        print(f"{'='*80}")
        
        # Step 1: Analyze cluster activations
        cluster_analysis = self.analyze_cluster_activations(
            clustering_results, pattern_name, layer
        )
        
        # Step 2: Find highest activating clusters
        top_clusters = self.find_highest_activating_clusters(
            cluster_analysis, top_k
        )
        
        # Step 3: Extract activation IDs
        activation_ids = self.extract_activation_ids(
            top_clusters, pattern_name, layer, all_tokens, sample_indices
        )
        
        # Step 4: Create filter structure
        activation_filter = self.create_activation_filter(activation_ids)
        
        # Return complete analysis
        return {
            'cluster_analysis': cluster_analysis,
            'top_clusters': top_clusters,
            'activation_ids': activation_ids,
            'activation_filter': activation_filter,
            'analysis_config': {
                'pattern_name': pattern_name,
                'layer': layer,
                'all_tokens': all_tokens,
                'sample_indices': sample_indices,
                'top_k': top_k
            }
        }

    def analyze_umap_3d_clustering_result(self,
                                        clustering_results: Dict,
                                        pattern_name: str,
                                        layer: int = 17,
                                        all_tokens: bool = True,
                                        sample_indices: Optional[List[int]] = None,
                                        top_k: int = 3) -> Dict:
        """
        Enhanced pipeline for UMAP-first results that includes 3D spatial analysis.
        
        Args:
            clustering_results: UMAP-first clustering results with 3D embeddings
            pattern_name: Name of cognitive pattern
            layer: Layer number analyzed
            all_tokens: Whether all tokens were used
            sample_indices: Sample indices used in analysis
            top_k: Number of top clusters per direction/state
            
        Returns:
            Complete analysis including 3D directional analysis
        """
        print(f"\n{'='*80}")
        print(f"🌌 ENHANCED 3D CLUSTER ANALYSIS PIPELINE")
        print(f"{'='*80}")
        
        # Step 1: Standard cluster activation analysis
        cluster_analysis = self.analyze_cluster_activations(
            clustering_results, pattern_name, layer
        )
        
        # Step 2: Find highest activating clusters by magnitude
        top_clusters = self.find_highest_activating_clusters(
            cluster_analysis, top_k
        )
        
        # Step 3: 3D spatial analysis (only if 3D embeddings available)
        spatial_analysis = self.analyze_clusters_in_3d_space(
            clustering_results, pattern_name
        )
        
        # Step 4: Find directionally extreme clusters
        directional_extremes = self.find_extreme_directional_clusters(
            spatial_analysis, top_k
        )
        
        # Step 5: Extract activation IDs for magnitude-based clusters
        activation_ids = self.extract_activation_ids(
            top_clusters, pattern_name, layer, all_tokens, sample_indices
        )
        
        # Step 6: Extract activation IDs for directionally extreme clusters
        directional_activation_ids = self.extract_directional_activation_ids(
            directional_extremes, pattern_name, layer, all_tokens, sample_indices
        )
        
        # Step 7: Create filter structures
        activation_filter = self.create_activation_filter(activation_ids)
        directional_filter = self.create_directional_activation_filter(directional_activation_ids)
        
        # Return enhanced analysis
        return {
            'cluster_analysis': cluster_analysis,
            'top_clusters': top_clusters,
            'spatial_analysis': spatial_analysis,
            'directional_extremes': directional_extremes,
            'activation_ids': activation_ids,
            'directional_activation_ids': directional_activation_ids,
            'activation_filter': activation_filter,
            'directional_filter': directional_filter,
            'analysis_config': {
                'pattern_name': pattern_name,
                'layer': layer,
                'all_tokens': all_tokens,
                'sample_indices': sample_indices,
                'top_k': top_k,
                'analysis_type': '3d_enhanced'
            }
        }
    
    def create_directional_activation_filter(self, directional_activation_ids: Dict) -> Dict:
        """
        Create filter structures for directionally extreme clusters.
        
        Args:
            directional_activation_ids: Results from extract_directional_activation_ids
            
        Returns:
            Filter dictionary organized by direction
        """
        print(f"\n🧭 Creating directional activation filter structures:")
        
        directional_filters = {}
        
        for direction_key, activations in directional_activation_ids.items():
            if not activations:
                directional_filters[direction_key] = {
                    'sample_ids': [],
                    'token_positions': [],
                    'sample_token_pairs': [],
                    'states': []
                }
                continue
            
            sample_ids = [act['sample_id'] for act in activations]
            token_positions = [act['token_position'] for act in activations]
            states = [act['state'] for act in activations]
            sample_token_pairs = [(act['sample_id'], act['token_position']) for act in activations]
            
            directional_filters[direction_key] = {
                'sample_ids': list(set(sample_ids)),
                'token_positions': list(set(token_positions)),
                'sample_token_pairs': sample_token_pairs,
                'states': list(set(states)),
                'activation_count': len(activations)
            }
            
            print(f"   {direction_key}: {len(set(sample_ids))} samples, "
                  f"{len(activations)} activations, states: {list(set(states))}")
        
        return directional_filters


def analyze_clustering_from_notebook_results(results_dict: Dict,
                                           activations_dir: Union[str, Path],
                                           metadata_path: Union[str, Path],
                                           top_k: int = 3) -> Dict:
    """
    Convenience function to analyze standard clustering results from the notebook.
    
    Args:
        results_dict: Results dictionary from perform_single_pattern_analysis
        activations_dir: Path to activations directory
        metadata_path: Path to metadata JSON
        top_k: Number of top clusters per state
        
    Returns:
        Complete cluster analysis results
    """
    # Create analyzer
    analyzer = ClusterAnalyzer(activations_dir, metadata_path)
    
    # Extract configuration from results
    pattern_name = results_dict['pattern_name']
    config_str = results_dict['config_str']
    all_tokens = 'AllTokens' in config_str
    
    # Get sample indices (assume all samples for now)
    sample_indices = analyzer.pattern_indices[pattern_name]
    
    # Run complete analysis
    return analyzer.analyze_umap_clustering_result(
        results_dict['clustering_results'],
        pattern_name,
        layer=17,  # Fixed to layer 17 as in notebook
        all_tokens=all_tokens,
        sample_indices=sample_indices,
        top_k=top_k
    )


def analyze_3d_clustering_from_notebook_results(results_dict: Dict,
                                              activations_dir: Union[str, Path],
                                              metadata_path: Union[str, Path],
                                              top_k: int = 3) -> Dict:
    """
    Convenience function to analyze UMAP-first (3D) clustering results from the notebook.
    
    Args:
        results_dict: Results dictionary from perform_single_pattern_analysis_umap_first
        activations_dir: Path to activations directory
        metadata_path: Path to metadata JSON
        top_k: Number of top clusters per state/direction
        
    Returns:
        Enhanced cluster analysis results with 3D directional analysis
    """
    # Create analyzer
    analyzer = ClusterAnalyzer(activations_dir, metadata_path)
    
    # Extract configuration from results
    pattern_name = results_dict['pattern_name']
    config_str = results_dict['config_str']
    all_tokens = 'AllTokens' in config_str.replace('_UMAPFirst', '')
    
    # Get sample indices (assume all samples for now)
    sample_indices = analyzer.pattern_indices[pattern_name]
    
    # Run enhanced 3D analysis
    return analyzer.analyze_umap_3d_clustering_result(
        results_dict['clustering_results'],
        pattern_name,
        layer=17,  # Fixed to layer 17 as in notebook
        all_tokens=all_tokens,
        sample_indices=sample_indices,
        top_k=top_k
    )


def create_example_usage_script():
    """
    Creates a complete example script showing how to use the utilities.
    
    Returns:
        String containing example code
    """
    example_code = '''
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
# analysis = analyze_clustering_from_notebook_results(
#     results_all_tokens_all_samples,
#     activations_dir=activations_dir,
#     metadata_path=metadata_path,
#     top_k=3
# )

# print("Top clusters by activation magnitude:")
# for state, clusters in analysis['top_clusters'].items():
#     print(f"\n{state} state:")
#     for i, cluster in enumerate(clusters):
#         print(f"  #{i+1}: Cluster {cluster['cluster_id']} - "
#               f"Magnitude: {cluster['activation_magnitude']:.2f}, "
#               f"Size: {cluster['size']}")

# print(f"\nActivation filter created with:")
# for state, filter_info in analysis['activation_filter'].items():
#     print(f"  {state}: {filter_info['activation_count']} activations")


# Example 2: Analyze 3D clustering results (UMAP-first approach)
print("\n" + "=" * 80)
print("🌌 ANALYZING 3D CLUSTERING RESULTS")
print("=" * 80)

# Assuming you have results_umap_first from the notebook
# enhanced_analysis = analyze_3d_clustering_from_notebook_results(
#     results_umap_first,
#     activations_dir=activations_dir,
#     metadata_path=metadata_path,
#     top_k=2
# )

# print("Directionally extreme clusters:")
# for direction, clusters in enhanced_analysis['directional_extremes'].items():
#     print(f"\n{direction}:")
#     for i, cluster in enumerate(clusters):
#         print(f"  #{i+1}: {cluster['state']} Cluster {cluster['cluster_id']} - "
#               f"Coord: {cluster['coordinate_value']:.2f}, "
#               f"Size: {cluster['size']}")

# print(f"\nDirectional filters created:")
# for direction, filter_info in enhanced_analysis['directional_filter'].items():
#     if filter_info['activation_count'] > 0:
#         print(f"  {direction}: {filter_info['activation_count']} activations "
#               f"from states: {filter_info['states']}")


# Example 3: Direct use of ClusterAnalyzer class
print("\n" + "=" * 80) 
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
print("\n" + "=" * 80)
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
'''
    
    return example_code


# Example usage function
def demo_cluster_analysis():
    """
    Demonstration of how to use the cluster analysis utilities.
    """
    print("🚀 Cluster Analysis Utilities Demo")
    print("="*50)
    
    example_script = create_example_usage_script()
    print("\n" + example_script)
    
    print("\n" + "="*50)
    print("💡 Quick Start Guide:")
    print("="*50)
    print("1. Run your UMAP analysis notebook to get clustering results")
    print("2. Import this utility: from cluster_analysis_utils import analyze_clustering_from_notebook_results")
    print("3. Call analyze_clustering_from_notebook_results() with your results")
    print("4. Use the returned activation filters to subset your data")
    print("5. For 3D analysis, use analyze_3d_clustering_from_notebook_results() instead")
    print("\nSee docstrings in each function for detailed parameter descriptions.")
    
def save_example_script(filename: str = "example_cluster_analysis.py"):
    """Save the example usage script to a file."""
    example_script = create_example_usage_script()
    
    with open(filename, 'w') as f:
        f.write(example_script)
    
    print(f"Example script saved to {filename}")
    return filename


if __name__ == "__main__":
    demo_cluster_analysis()