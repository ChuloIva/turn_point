#!/usr/bin/env python3
"""
Comprehensive transformation experiments on cognitive pattern activations.

This script implements all the transformation and validation approaches:
- Transition direction vectors
- Reconstruction experiments  
- Dot product analysis
- Multi-sample averaging
- Layer-wise analysis
- PCA analysis
"""

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from datetime import datetime
import os

class CognitiveTransformationAnalyzer:
    def __init__(self, base_path: str = "/Users/ivanculo/Desktop/Projects/turn_point", use_mps: bool = True):
        self.base_path = Path(base_path)
        self.activations_dir = self.base_path / "activations"
        self.data_dir = self.base_path / "data"
        self.output_dir = self.base_path / "transformation_outputs"
        
        # Set device (Mac MPS support)
        if use_mps and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load activation files and metadata"""
        print("Loading activation files and metadata...")
        
        # Load activation files
        self.negative_activations = torch.load(
            self.activations_dir / "activations_8ff00d963316212d.pt", 
            map_location=self.device
        )
        self.positive_activations = torch.load(
            self.activations_dir / "activations_e5ad16e9b3c33c9b.pt", 
            map_location=self.device
        )
        self.transition_activations = torch.load(
            self.activations_dir / "activations_332f24de2a3f82ff.pt", 
            map_location=self.device
        )
        
        # Load metadata
        with open(self.data_dir / "final" / "enriched_metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Create mapping from pattern names to indices
        self.pattern_indices = {}
        for i, entry in enumerate(self.metadata):
            pattern_name = entry['bad_good_narratives_match']['cognitive_pattern_name_from_bad_good']
            if pattern_name not in self.pattern_indices:
                self.pattern_indices[pattern_name] = []
            self.pattern_indices[pattern_name].append(i)
            
        print(f"Loaded data for {len(self.pattern_indices)} cognitive patterns")
        print(f"Total samples: {len(self.metadata)}")
        
        # Extract layer activations for easier access
        self.layers = [17, 21]
        self.activation_data = {}
        
        for layer in self.layers:
            self.activation_data[layer] = {
                'negative': self.negative_activations[f'negative_layer_{layer}'],
                'positive': self.positive_activations[f'positive_layer_{layer}'],
                'transition': self.transition_activations[f'transition_layer_{layer}']
            }
    
    def get_all_token_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Extract all token activations flattened"""
        # activations shape: [batch, seq_len, hidden_dim]
        # Flatten to [batch * seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = activations.shape
        return activations.reshape(-1, hidden_dim)  # [batch * seq_len, hidden_dim]
    
    def compute_direction_vectors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute direction vectors for each cognitive pattern and layer.
        
        Returns:
            Dictionary with structure:
            {
                pattern_name: {
                    layer: {
                        'recovery_direction': positive - negative,
                        'therapeutic_direction': transition - negative, 
                        'positive_transition_direction': positive - transition,
                        'canonical_negative': averaged negative,
                        'canonical_positive': averaged positive,
                        'canonical_transition': averaged transition
                    }
                }
            }
        """
        print("Computing direction vectors...")
        
        direction_vectors = {}
        
        for pattern_name, indices in self.pattern_indices.items():
            direction_vectors[pattern_name] = {}
            
            for layer in self.layers:
                # Get ALL token activations for this pattern and layer
                neg_acts = self.get_all_token_activations(
                    self.activation_data[layer]['negative'][indices]
                )
                pos_acts = self.get_all_token_activations(
                    self.activation_data[layer]['positive'][indices]
                )
                trans_acts = self.get_all_token_activations(
                    self.activation_data[layer]['transition'][indices]
                )
                
                # Compute canonical representations (averages)
                canonical_negative = neg_acts.mean(dim=0)
                canonical_positive = pos_acts.mean(dim=0) 
                canonical_transition = trans_acts.mean(dim=0)
                
                # Compute direction vectors
                recovery_direction = canonical_positive - canonical_negative
                therapeutic_direction = canonical_transition - canonical_negative
                positive_transition_direction = canonical_positive - canonical_transition
                
                direction_vectors[pattern_name][layer] = {
                    'recovery_direction': recovery_direction,
                    'therapeutic_direction': therapeutic_direction,
                    'positive_transition_direction': positive_transition_direction,
                    'canonical_negative': canonical_negative,
                    'canonical_positive': canonical_positive,
                    'canonical_transition': canonical_transition,
                    'individual_negative': neg_acts,
                    'individual_positive': pos_acts,
                    'individual_transition': trans_acts
                }
        
        return direction_vectors
    
    def reconstruction_experiments(self, direction_vectors: Dict) -> Dict[str, Dict]:
        """
        Perform reconstruction experiments:
        - negative + recovery_direction should ≈ positive
        - negative + therapeutic_direction should ≈ transition
        - transition + positive_transition_direction should ≈ positive
        """
        print("Running reconstruction experiments...")
        
        reconstruction_results = {}
        
        for pattern_name in self.pattern_indices.keys():
            reconstruction_results[pattern_name] = {}
            
            for layer in self.layers:
                vectors = direction_vectors[pattern_name][layer]
                
                # Reconstruct positive from negative
                reconstructed_positive = vectors['canonical_negative'] + vectors['recovery_direction']
                
                # Reconstruct transition from negative  
                reconstructed_transition = vectors['canonical_negative'] + vectors['therapeutic_direction']
                
                # Reconstruct positive from transition
                reconstructed_positive_from_trans = vectors['canonical_transition'] + vectors['positive_transition_direction']
                
                # Compute reconstruction accuracies (cosine similarities)
                pos_similarity = cosine_similarity(
                    reconstructed_positive.unsqueeze(0).numpy(),
                    vectors['canonical_positive'].unsqueeze(0).numpy()
                )[0, 0]
                
                trans_similarity = cosine_similarity(
                    reconstructed_transition.unsqueeze(0).numpy(),
                    vectors['canonical_transition'].unsqueeze(0).numpy()
                )[0, 0]
                
                pos_from_trans_similarity = cosine_similarity(
                    reconstructed_positive_from_trans.unsqueeze(0).numpy(),
                    vectors['canonical_positive'].unsqueeze(0).numpy()
                )[0, 0]
                
                reconstruction_results[pattern_name][layer] = {
                    'negative_to_positive_similarity': float(pos_similarity),
                    'negative_to_transition_similarity': float(trans_similarity),
                    'transition_to_positive_similarity': float(pos_from_trans_similarity),
                    'reconstructed_positive': reconstructed_positive,
                    'reconstructed_transition': reconstructed_transition,
                    'reconstructed_positive_from_transition': reconstructed_positive_from_trans
                }
        
        return reconstruction_results
    
    def dot_product_analysis(self, direction_vectors: Dict) -> Dict[str, Dict]:
        """
        Perform dot product analysis:
        - Check if transition · recovery_direction > 0 (aligned with recovery)
        - Measure alignment of individual samples with direction vectors
        """
        print("Running dot product analysis...")
        
        dot_product_results = {}
        
        for pattern_name in self.pattern_indices.keys():
            dot_product_results[pattern_name] = {}
            
            for layer in self.layers:
                vectors = direction_vectors[pattern_name][layer]
                
                # Transition alignment with recovery direction
                transition_recovery_alignment = torch.dot(
                    vectors['canonical_transition'], 
                    vectors['recovery_direction']
                ).item()
                
                # Individual sample alignments
                neg_recovery_alignments = torch.matmul(
                    vectors['individual_negative'], 
                    vectors['recovery_direction']
                ).numpy()
                
                pos_recovery_alignments = torch.matmul(
                    vectors['individual_positive'], 
                    vectors['recovery_direction'] 
                ).numpy()
                
                trans_recovery_alignments = torch.matmul(
                    vectors['individual_transition'],
                    vectors['recovery_direction']
                ).numpy()
                
                # Therapeutic direction alignments
                trans_therapeutic_alignments = torch.matmul(
                    vectors['individual_transition'],
                    vectors['therapeutic_direction']
                ).numpy()
                
                dot_product_results[pattern_name][layer] = {
                    'transition_recovery_alignment': transition_recovery_alignment,
                    'negative_recovery_alignments': neg_recovery_alignments.tolist(),
                    'positive_recovery_alignments': pos_recovery_alignments.tolist(),
                    'transition_recovery_alignments': trans_recovery_alignments.tolist(),
                    'transition_therapeutic_alignments': trans_therapeutic_alignments.tolist(),
                    'mean_negative_recovery_alignment': float(neg_recovery_alignments.mean()),
                    'mean_positive_recovery_alignment': float(pos_recovery_alignments.mean()),
                    'mean_transition_recovery_alignment': float(trans_recovery_alignments.mean()),
                    'mean_transition_therapeutic_alignment': float(trans_therapeutic_alignments.mean())
                }
        
        return dot_product_results
    
    def interpolation_analysis(self, direction_vectors: Dict) -> Dict[str, Dict]:
        """
        Create interpolation paths: negative + t*(positive - negative) for t ∈ [0,1]
        and measure how they relate to actual transition samples
        """
        print("Running interpolation analysis...")
        
        interpolation_results = {}
        t_values = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
        
        for pattern_name in self.pattern_indices.keys():
            interpolation_results[pattern_name] = {}
            
            for layer in self.layers:
                vectors = direction_vectors[pattern_name][layer]
                
                # Create interpolation path
                interpolated_points = []
                similarities_to_transition = []
                
                for t in t_values:
                    interpolated = vectors['canonical_negative'] + t * vectors['recovery_direction']
                    interpolated_points.append(interpolated)
                    
                    # Compute similarity to canonical transition
                    sim = cosine_similarity(
                        interpolated.unsqueeze(0).numpy(),
                        vectors['canonical_transition'].unsqueeze(0).numpy()
                    )[0, 0]
                    similarities_to_transition.append(float(sim))
                
                # Find t value with highest similarity to transition
                best_t_idx = np.argmax(similarities_to_transition)
                best_t = t_values[best_t_idx]
                
                interpolation_results[pattern_name][layer] = {
                    't_values': t_values.tolist(),
                    'similarities_to_transition': similarities_to_transition,
                    'best_t': float(best_t),
                    'max_similarity': float(similarities_to_transition[best_t_idx]),
                    'interpolated_points': [p.numpy().tolist() for p in interpolated_points]
                }
        
        return interpolation_results
    
    def pca_analysis_per_pattern(self, direction_vectors: Dict) -> Dict[str, Dict]:
        """
        Run PCA on individual cognitive patterns using all token activations
        """
        print("Running PCA analysis per cognitive pattern...")
        
        pca_results = {}
        
        for pattern_name, indices in self.pattern_indices.items():
            print(f"Analyzing pattern: {pattern_name}")
            pca_results[pattern_name] = {}
            
            for layer in self.layers:
                # Get ALL token activations for this pattern and layer
                # Take every 4th token to reduce memory usage
                neg_data = self.activation_data[layer]['negative'][indices][:, ::4, :]
                pos_data = self.activation_data[layer]['positive'][indices][:, ::4, :]
                trans_data = self.activation_data[layer]['transition'][indices][:, ::4, :]
                
                neg_acts = self.get_all_token_activations(neg_data).cpu().numpy()
                pos_acts = self.get_all_token_activations(pos_data).cpu().numpy()
                trans_acts = self.get_all_token_activations(trans_data).cpu().numpy()
                
                # Combine all activations for this pattern
                all_activations = np.vstack([neg_acts, pos_acts, trans_acts])
                
                # Create labels for each activation
                labels = (['negative'] * len(neg_acts) + 
                         ['positive'] * len(pos_acts) + 
                         ['transition'] * len(trans_acts))
                
                # Standardize features
                scaler = StandardScaler()
                all_activations_scaled = scaler.fit_transform(all_activations)
                
                # Run PCA
                n_components = min(50, all_activations_scaled.shape[1])
                pca = PCA(n_components=n_components)
                pca_transformed = pca.fit_transform(all_activations_scaled)
                
                pca_results[pattern_name][layer] = {
                    'pca_components': pca.components_[:10].tolist(),
                    'explained_variance_ratio': pca.explained_variance_ratio_[:10].tolist(),
                    'pca_transformed': pca_transformed.tolist(),
                    'labels': labels,
                    'total_explained_variance': float(pca.explained_variance_ratio_[:10].sum()),
                    'n_samples': len(all_activations),
                    'n_negative': len(neg_acts),
                    'n_positive': len(pos_acts),
                    'n_transition': len(trans_acts)
                }
        
        return pca_results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file"""
        output_path = self.output_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert any torch tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        
        results_serializable = convert_tensors(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    def create_summary_report(self, direction_vectors: Dict, reconstruction_results: Dict, 
                            dot_product_results: Dict, interpolation_results: Dict, 
                            pca_results: Dict) -> Dict[str, Any]:
        """Create a comprehensive summary report"""
        
        summary = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_patterns': len(self.pattern_indices),
                'total_samples': len(self.metadata),
                'samples_per_pattern': 40,
                'layers_analyzed': self.layers,
                'cognitive_patterns': list(self.pattern_indices.keys())
            },
            'key_findings': {},
            'pattern_summaries': {}
        }
        
        # Overall findings across patterns
        for layer in self.layers:
            layer_findings = {
                'reconstruction_accuracies': {},
                'alignment_scores': {},
                'interpolation_insights': {},
                'pca_insights': {
                    'per_pattern_analysis': 'enabled',
                    'patterns_analyzed': len(self.pattern_indices)
                }
            }
            
            # Collect reconstruction accuracies
            recon_scores = []
            for pattern_name in self.pattern_indices.keys():
                scores = reconstruction_results[pattern_name][layer]
                recon_scores.extend([
                    scores['negative_to_positive_similarity'],
                    scores['negative_to_transition_similarity'], 
                    scores['transition_to_positive_similarity']
                ])
            
            layer_findings['reconstruction_accuracies'] = {
                'mean': float(np.mean(recon_scores)),
                'std': float(np.std(recon_scores)),
                'min': float(np.min(recon_scores)),
                'max': float(np.max(recon_scores))
            }
            
            # Collect alignment scores
            alignment_scores = []
            for pattern_name in self.pattern_indices.keys():
                scores = dot_product_results[pattern_name][layer]
                alignment_scores.extend([
                    scores['mean_negative_recovery_alignment'],
                    scores['mean_positive_recovery_alignment'],
                    scores['mean_transition_recovery_alignment']
                ])
            
            layer_findings['alignment_scores'] = {
                'mean': float(np.mean(alignment_scores)),
                'std': float(np.std(alignment_scores)),
                'recovery_alignment_positive': np.mean([dot_product_results[p][layer]['mean_positive_recovery_alignment'] for p in self.pattern_indices.keys()])
            }
            
            # Interpolation insights
            best_t_values = [interpolation_results[p][layer]['best_t'] for p in self.pattern_indices.keys()]
            layer_findings['interpolation_insights'] = {
                'mean_best_t': float(np.mean(best_t_values)),
                'std_best_t': float(np.std(best_t_values)),
                'interpretation': 'Transitions occur early in recovery path' if np.mean(best_t_values) < 0.5 else 'Transitions occur late in recovery path'
            }
            
            summary['key_findings'][f'layer_{layer}'] = layer_findings
        
        # Individual pattern summaries
        for pattern_name in self.pattern_indices.keys():
            pattern_summary = {}
            
            for layer in self.layers:
                pattern_summary[f'layer_{layer}'] = {
                    'reconstruction_quality': {
                        'neg_to_pos': reconstruction_results[pattern_name][layer]['negative_to_positive_similarity'],
                        'neg_to_trans': reconstruction_results[pattern_name][layer]['negative_to_transition_similarity'],
                        'trans_to_pos': reconstruction_results[pattern_name][layer]['transition_to_positive_similarity']
                    },
                    'recovery_alignment': dot_product_results[pattern_name][layer]['mean_positive_recovery_alignment'],
                    'transition_position': interpolation_results[pattern_name][layer]['best_t'],
                    'max_transition_similarity': interpolation_results[pattern_name][layer]['max_similarity']
                }
            
            summary['pattern_summaries'][pattern_name] = pattern_summary
        
        return summary
    
    def run_all_experiments(self):
        """Run all transformation experiments"""
        print("Starting comprehensive cognitive transformation analysis...")
        print("="*60)
        
        # 1. Compute direction vectors
        direction_vectors = self.compute_direction_vectors()
        self.save_results(direction_vectors, 'direction_vectors')
        
        # 2. Reconstruction experiments
        reconstruction_results = self.reconstruction_experiments(direction_vectors)
        self.save_results(reconstruction_results, 'reconstruction_results')
        
        # 3. Dot product analysis
        dot_product_results = self.dot_product_analysis(direction_vectors)
        self.save_results(dot_product_results, 'dot_product_analysis')
        
        # 4. Interpolation analysis
        interpolation_results = self.interpolation_analysis(direction_vectors)
        self.save_results(interpolation_results, 'interpolation_analysis')
        
        # 5. PCA analysis per pattern
        pca_results = self.pca_analysis_per_pattern(direction_vectors)
        self.save_results(pca_results, 'pca_analysis_per_pattern')
        
        # 6. Create summary report
        summary_report = self.create_summary_report(
            direction_vectors, reconstruction_results, dot_product_results, 
            interpolation_results, pca_results
        )
        self.save_results(summary_report, 'summary_report')
        
        print("="*60)
        print("All experiments completed successfully!")
        print(f"Results saved in: {self.output_dir}")
        
        return {
            'direction_vectors': direction_vectors,
            'reconstruction_results': reconstruction_results,
            'dot_product_results': dot_product_results,
            'interpolation_results': interpolation_results,
            'pca_results': pca_results,
            'summary_report': summary_report
        }

def main():
    analyzer = CognitiveTransformationAnalyzer()
    results = analyzer.run_all_experiments()
    
    # Print key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS SUMMARY")
    print("="*60)
    
    summary = results['summary_report']
    
    for layer in [17, 21]:
        layer_findings = summary['key_findings'][f'layer_{layer}']
        print(f"\nLayer {layer}:")
        print(f"  Mean Reconstruction Accuracy: {layer_findings['reconstruction_accuracies']['mean']:.3f}")
        print(f"  Mean Alignment Score: {layer_findings['alignment_scores']['mean']:.3f}")
        print(f"  Mean Transition Position (t): {layer_findings['interpolation_insights']['mean_best_t']:.3f}")
        print(f"  PCA Patterns Analyzed: {layer_findings['pca_insights']['patterns_analyzed']}")
    
    print(f"\nTotal cognitive patterns analyzed: {summary['experiment_metadata']['total_patterns']}")
    print(f"Total samples processed: {summary['experiment_metadata']['total_samples']}")

if __name__ == "__main__":
    main()