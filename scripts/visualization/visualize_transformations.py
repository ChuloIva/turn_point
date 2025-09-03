#!/usr/bin/env python3
"""
Visualization script for cognitive transformation analysis results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class TransformationVisualizer:
    def __init__(self, output_dir: str = "/Users/ivanculo/Desktop/Projects/turn_point/transformation_outputs"):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Load latest results
        self.load_latest_results()
        
    def load_latest_results(self):
        """Load the most recent analysis results"""
        print("Loading latest analysis results...")
        
        # Find latest files
        result_files = {
            'summary': list(self.output_dir.glob("summary_report_*.json"))[-1],
            'reconstruction': list(self.output_dir.glob("reconstruction_results_*.json"))[-1],
            'dot_product': list(self.output_dir.glob("dot_product_analysis_*.json"))[-1],
            'interpolation': list(self.output_dir.glob("interpolation_analysis_*.json"))[-1],
            'pca': list(self.output_dir.glob("pca_analysis_*.json"))[-1]
        }
        
        self.results = {}
        for key, file_path in result_files.items():
            with open(file_path, 'r') as f:
                self.results[key] = json.load(f)
        
        print("Results loaded successfully!")
    
    def plot_reconstruction_accuracy(self):
        """Plot reconstruction accuracy across patterns and layers"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reconstruction Accuracy Analysis', fontsize=16)
        
        # Prepare data
        patterns = list(self.results['reconstruction'].keys())
        layers = [17, 21]
        
        # Plot 1: Heatmap of reconstruction accuracies
        recon_data = []
        for pattern in patterns:
            row = []
            for layer in layers:
                scores = self.results['reconstruction'][pattern][str(layer)]
                mean_score = np.mean([
                    scores['negative_to_positive_similarity'],
                    scores['negative_to_transition_similarity'],
                    scores['transition_to_positive_similarity']
                ])
                row.append(mean_score)
            recon_data.append(row)
        
        sns.heatmap(recon_data, 
                   xticklabels=[f'Layer {l}' for l in layers],
                   yticklabels=[p[:20] + '...' if len(p) > 20 else p for p in patterns],
                   annot=True, fmt='.3f', cmap='viridis',
                   ax=axes[0,0])
        axes[0,0].set_title('Mean Reconstruction Accuracy by Pattern')
        
        # Plot 2: Reconstruction type comparison
        recon_types = ['negative_to_positive_similarity', 'negative_to_transition_similarity', 'transition_to_positive_similarity']
        type_labels = ['Neg→Pos', 'Neg→Trans', 'Trans→Pos']
        
        layer_17_scores = {rtype: [] for rtype in recon_types}
        layer_21_scores = {rtype: [] for rtype in recon_types}
        
        for pattern in patterns:
            for rtype in recon_types:
                layer_17_scores[rtype].append(self.results['reconstruction'][pattern]['17'][rtype])
                layer_21_scores[rtype].append(self.results['reconstruction'][pattern]['21'][rtype])
        
        x = np.arange(len(type_labels))
        width = 0.35
        
        layer_17_means = [np.mean(layer_17_scores[rtype]) for rtype in recon_types]
        layer_21_means = [np.mean(layer_21_scores[rtype]) for rtype in recon_types]
        
        axes[0,1].bar(x - width/2, layer_17_means, width, label='Layer 17', alpha=0.8)
        axes[0,1].bar(x + width/2, layer_21_means, width, label='Layer 21', alpha=0.8)
        axes[0,1].set_xlabel('Reconstruction Type')
        axes[0,1].set_ylabel('Mean Cosine Similarity')
        axes[0,1].set_title('Reconstruction Accuracy by Type')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(type_labels)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution of reconstruction scores
        all_scores_17 = []
        all_scores_21 = []
        
        for pattern in patterns:
            scores_17 = list(self.results['reconstruction'][pattern]['17'].values())[:3]
            scores_21 = list(self.results['reconstruction'][pattern]['21'].values())[:3]
            all_scores_17.extend(scores_17)
            all_scores_21.extend(scores_21)
        
        axes[1,0].hist(all_scores_17, bins=20, alpha=0.7, label='Layer 17', density=True)
        axes[1,0].hist(all_scores_21, bins=20, alpha=0.7, label='Layer 21', density=True)
        axes[1,0].set_xlabel('Cosine Similarity')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Distribution of Reconstruction Scores')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Pattern-specific reconstruction quality
        pattern_means = []
        pattern_names = []
        
        for pattern in patterns:
            scores_17 = list(self.results['reconstruction'][pattern]['17'].values())[:3]
            scores_21 = list(self.results['reconstruction'][pattern]['21'].values())[:3]
            mean_score = np.mean(scores_17 + scores_21)
            pattern_means.append(mean_score)
            pattern_names.append(pattern[:15] + '...' if len(pattern) > 15 else pattern)
        
        y_pos = np.arange(len(pattern_names))
        axes[1,1].barh(y_pos, pattern_means)
        axes[1,1].set_yticks(y_pos)
        axes[1,1].set_yticklabels(pattern_names)
        axes[1,1].set_xlabel('Mean Reconstruction Score')
        axes[1,1].set_title('Overall Reconstruction Quality by Pattern')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'reconstruction_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_interpolation_analysis(self):
        """Plot interpolation analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Interpolation Analysis: Transition Positioning', fontsize=16)
        
        patterns = list(self.results['interpolation'].keys())
        
        # Plot 1: Best t values (transition position in interpolation)
        t_values_17 = [self.results['interpolation'][p]['17']['best_t'] for p in patterns]
        t_values_21 = [self.results['interpolation'][p]['21']['best_t'] for p in patterns]
        
        x = np.arange(len(patterns))
        width = 0.35
        
        axes[0,0].bar(x - width/2, t_values_17, width, label='Layer 17', alpha=0.8)
        axes[0,0].bar(x + width/2, t_values_21, width, label='Layer 21', alpha=0.8)
        axes[0,0].set_xlabel('Cognitive Pattern')
        axes[0,0].set_ylabel('Best t value')
        axes[0,0].set_title('Optimal Transition Position (t) by Pattern')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in patterns], 
                                 rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Midpoint')
        
        # Plot 2: Max similarity to transition
        max_sim_17 = [self.results['interpolation'][p]['17']['max_similarity'] for p in patterns]
        max_sim_21 = [self.results['interpolation'][p]['21']['max_similarity'] for p in patterns]
        
        axes[0,1].bar(x - width/2, max_sim_17, width, label='Layer 17', alpha=0.8)
        axes[0,1].bar(x + width/2, max_sim_21, width, label='Layer 21', alpha=0.8)
        axes[0,1].set_xlabel('Cognitive Pattern')
        axes[0,1].set_ylabel('Max Similarity to Transition')
        axes[0,1].set_title('Maximum Similarity to Actual Transitions')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in patterns], 
                                 rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution of t values
        all_t_17 = [self.results['interpolation'][p]['17']['best_t'] for p in patterns]
        all_t_21 = [self.results['interpolation'][p]['21']['best_t'] for p in patterns]
        
        axes[1,0].hist(all_t_17, bins=10, alpha=0.7, label='Layer 17', density=True)
        axes[1,0].hist(all_t_21, bins=10, alpha=0.7, label='Layer 21', density=True)
        axes[1,0].set_xlabel('Best t value')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Distribution of Optimal t Values')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Midpoint')
        
        # Plot 4: Example interpolation curve for first pattern
        first_pattern = patterns[0]
        t_vals = self.results['interpolation'][first_pattern]['17']['t_values']
        similarities = self.results['interpolation'][first_pattern]['17']['similarities_to_transition']
        best_t = self.results['interpolation'][first_pattern]['17']['best_t']
        
        axes[1,1].plot(t_vals, similarities, 'b-o', label='Layer 17', markersize=4)
        
        t_vals_21 = self.results['interpolation'][first_pattern]['21']['t_values']
        similarities_21 = self.results['interpolation'][first_pattern]['21']['similarities_to_transition']
        axes[1,1].plot(t_vals_21, similarities_21, 'r-s', label='Layer 21', markersize=4)
        
        axes[1,1].axvline(x=best_t, color='green', linestyle='--', alpha=0.7, label=f'Best t={best_t:.2f}')
        axes[1,1].set_xlabel('t (Interpolation Parameter)')
        axes[1,1].set_ylabel('Similarity to Actual Transition')
        axes[1,1].set_title(f'Interpolation Curve: {first_pattern[:20]}...')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'interpolation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_alignment_analysis(self):
        """Plot dot product alignment analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Direction Vector Alignment Analysis', fontsize=16)
        
        patterns = list(self.results['dot_product'].keys())
        
        # Plot 1: Recovery alignment by state type
        neg_alignments_17 = [self.results['dot_product'][p]['17']['mean_negative_recovery_alignment'] for p in patterns]
        pos_alignments_17 = [self.results['dot_product'][p]['17']['mean_positive_recovery_alignment'] for p in patterns]
        trans_alignments_17 = [self.results['dot_product'][p]['17']['mean_transition_recovery_alignment'] for p in patterns]
        
        x = np.arange(len(patterns))
        width = 0.25
        
        axes[0,0].bar(x - width, neg_alignments_17, width, label='Negative', alpha=0.8)
        axes[0,0].bar(x, pos_alignments_17, width, label='Positive', alpha=0.8)
        axes[0,0].bar(x + width, trans_alignments_17, width, label='Transition', alpha=0.8)
        
        axes[0,0].set_xlabel('Cognitive Pattern')
        axes[0,0].set_ylabel('Mean Recovery Alignment')
        axes[0,0].set_title('Recovery Direction Alignment (Layer 17)')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([p[:8] + '...' if len(p) > 8 else p for p in patterns], 
                                 rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Layer comparison
        pos_alignments_21 = [self.results['dot_product'][p]['21']['mean_positive_recovery_alignment'] for p in patterns]
        
        axes[0,1].bar(x - width/2, pos_alignments_17, width, label='Layer 17', alpha=0.8)
        axes[0,1].bar(x + width/2, pos_alignments_21, width, label='Layer 21', alpha=0.8)
        axes[0,1].set_xlabel('Cognitive Pattern')
        axes[0,1].set_ylabel('Mean Positive Recovery Alignment')
        axes[0,1].set_title('Positive State Recovery Alignment by Layer')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([p[:8] + '...' if len(p) > 8 else p for p in patterns], 
                                 rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 3: Therapeutic alignment
        therapeutic_alignments_17 = [self.results['dot_product'][p]['17']['mean_transition_therapeutic_alignment'] for p in patterns]
        therapeutic_alignments_21 = [self.results['dot_product'][p]['21']['mean_transition_therapeutic_alignment'] for p in patterns]
        
        axes[1,0].bar(x - width/2, therapeutic_alignments_17, width, label='Layer 17', alpha=0.8)
        axes[1,0].bar(x + width/2, therapeutic_alignments_21, width, label='Layer 21', alpha=0.8)
        axes[1,0].set_xlabel('Cognitive Pattern')
        axes[1,0].set_ylabel('Mean Therapeutic Alignment')
        axes[1,0].set_title('Transition-Therapeutic Direction Alignment')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels([p[:8] + '...' if len(p) > 8 else p for p in patterns], 
                                 rotation=45, ha='right')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 4: Alignment scatter plot
        recovery_align = [self.results['dot_product'][p]['17']['mean_positive_recovery_alignment'] for p in patterns]
        therapeutic_align = [self.results['dot_product'][p]['17']['mean_transition_therapeutic_alignment'] for p in patterns]
        
        scatter = axes[1,1].scatter(recovery_align, therapeutic_align, 
                                   c=range(len(patterns)), cmap='tab20', s=100, alpha=0.7)
        axes[1,1].set_xlabel('Recovery Alignment')
        axes[1,1].set_ylabel('Therapeutic Alignment')
        axes[1,1].set_title('Recovery vs Therapeutic Alignment (Layer 17)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add pattern labels to scatter plot
        for i, pattern in enumerate(patterns):
            axes[1,1].annotate(pattern[:10], (recovery_align[i], therapeutic_align[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'alignment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        
        # Extract key metrics from summary
        summary = self.results['summary']
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Cognitive Pattern Transformation Analysis - Summary Dashboard', fontsize=20)
        
        # 1. Reconstruction accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        layers = [17, 21]
        recon_means = [summary['key_findings'][f'layer_{l}']['reconstruction_accuracies']['mean'] for l in layers]
        ax1.bar(layers, recon_means, color=['skyblue', 'lightcoral'])
        ax1.set_title('Mean Reconstruction Accuracy')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Cosine Similarity')
        ax1.grid(True, alpha=0.3)
        
        # 2. Alignment scores comparison
        ax2 = fig.add_subplot(gs[0, 1])
        align_means = [summary['key_findings'][f'layer_{l}']['alignment_scores']['mean'] for l in layers]
        ax2.bar(layers, align_means, color=['lightgreen', 'orange'])
        ax2.set_title('Mean Alignment Scores')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Dot Product')
        ax2.grid(True, alpha=0.3)
        
        # 3. Interpolation insights
        ax3 = fig.add_subplot(gs[0, 2])
        t_means = [summary['key_findings'][f'layer_{l}']['interpolation_insights']['mean_best_t'] for l in layers]
        ax3.bar(layers, t_means, color=['plum', 'gold'])
        ax3.set_title('Mean Transition Position (t)')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('t value')
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
        
        # 4. PCA explained variance
        ax4 = fig.add_subplot(gs[0, 3])
        pca_vars = [summary['key_findings'][f'layer_{l}']['pca_insights']['explained_variance'] for l in layers]
        ax4.bar(layers, pca_vars, color=['lightcyan', 'mistyrose'])
        ax4.set_title('PCA Explained Variance')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Explained Variance Ratio')
        ax4.grid(True, alpha=0.3)
        
        # 5. Pattern-specific reconstruction quality heatmap
        ax5 = fig.add_subplot(gs[1, :2])
        patterns = list(summary['pattern_summaries'].keys())
        recon_matrix = []
        
        for pattern in patterns:
            row = []
            for layer in [17, 21]:
                scores = summary['pattern_summaries'][pattern][f'layer_{layer}']['reconstruction_quality']
                mean_score = np.mean([scores['neg_to_pos'], scores['neg_to_trans'], scores['trans_to_pos']])
                row.append(mean_score)
            recon_matrix.append(row)
        
        im = ax5.imshow(recon_matrix, cmap='viridis', aspect='auto')
        ax5.set_xticks([0, 1])
        ax5.set_xticklabels(['Layer 17', 'Layer 21'])
        ax5.set_yticks(range(len(patterns)))
        ax5.set_yticklabels([p[:25] + '...' if len(p) > 25 else p for p in patterns])
        ax5.set_title('Reconstruction Quality by Pattern and Layer')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
        cbar.set_label('Mean Reconstruction Score')
        
        # 6. Transition position analysis
        ax6 = fig.add_subplot(gs[1, 2:])
        
        t_values_all = []
        pattern_labels = []
        for pattern in patterns:
            for layer in [17, 21]:
                t_val = summary['pattern_summaries'][pattern][f'layer_{layer}']['transition_position']
                t_values_all.append(t_val)
                pattern_labels.append(f"{pattern[:15]}... L{layer}")
        
        y_pos = np.arange(len(t_values_all))
        bars = ax6.barh(y_pos, t_values_all, color=['skyblue' if 'L17' in label else 'lightcoral' for label in pattern_labels])
        ax6.set_yticks(y_pos[::2])  # Show every other label to avoid crowding
        ax6.set_yticklabels([pattern_labels[i] for i in range(0, len(pattern_labels), 2)])
        ax6.set_xlabel('Transition Position (t)')
        ax6.set_title('Transition Position in Recovery Path')
        ax6.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        ax6.grid(True, alpha=0.3)
        
        # 7. Key statistics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        stats_data = [
            ['Metric', 'Layer 17', 'Layer 21'],
            ['Mean Reconstruction Accuracy', f"{recon_means[0]:.3f}", f"{recon_means[1]:.3f}"],
            ['Mean Alignment Score', f"{align_means[0]:.3f}", f"{align_means[1]:.3f}"],
            ['Mean Transition Position', f"{t_means[0]:.3f}", f"{t_means[1]:.3f}"],
            ['PCA Explained Variance', f"{pca_vars[0]:.3f}", f"{pca_vars[1]:.3f}"],
            ['Total Patterns Analyzed', str(summary['experiment_metadata']['total_patterns']), str(summary['experiment_metadata']['total_patterns'])],
            ['Total Samples', str(summary['experiment_metadata']['total_samples']), str(summary['experiment_metadata']['total_samples'])]
        ]
        
        table = ax7.table(cellText=stats_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax7.set_title('Summary Statistics', fontsize=14, pad=20)
        
        plt.savefig(self.viz_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Generating comprehensive visualizations...")
        
        print("1. Creating reconstruction accuracy plots...")
        self.plot_reconstruction_accuracy()
        
        print("2. Creating interpolation analysis plots...")
        self.plot_interpolation_analysis()
        
        print("3. Creating alignment analysis plots...")
        self.plot_alignment_analysis()
        
        print("4. Creating summary dashboard...")
        self.create_summary_dashboard()
        
        print(f"\nAll visualizations saved to: {self.viz_dir}")
        print("Visualization files created:")
        for viz_file in self.viz_dir.glob("*.png"):
            print(f"  - {viz_file.name}")

def main():
    visualizer = TransformationVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()