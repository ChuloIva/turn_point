#!/usr/bin/env python3
"""
Practical SAE Analysis Script for Turn Point Project

This script demonstrates how to use the SAE analysis workflow with your
pre-computed activations from the cognitive patterns dataset.
"""

import sys
import torch
from pathlib import Path
import json
from sae_analysis import SAEActivationAnalyzer, SAEAnalysisConfig

def main():
    """Run SAE analysis on pre-computed activations"""
    
    print("=" * 60)
    print("SAE Analysis for Cognitive Patterns Project")
    print("=" * 60)
    
    # Configuration for your specific use case
    config = SAEAnalysisConfig(
        device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        sae_release="gpt2-small-res-jb",  # Start with GPT-2 small
        sae_id="blocks.7.hook_resid_pre",  # Layer 7 residual stream
        top_k_features=15,
        analysis_output_dir="sae_analysis_results"
    )
    
    print(f"Using device: {config.device}")
    print(f"SAE Configuration: {config.sae_release}/{config.sae_id}")
    
    # Initialize analyzer
    analyzer = SAEActivationAnalyzer(config)
    
    try:
        # Step 1: Discover available SAEs
        print("\n" + "─" * 40)
        print("Step 1: Discovering Available SAEs")
        print("─" * 40)
        
        available_saes = analyzer.discover_available_saes()
        
        # Find SAEs that might work with your model
        print("\nSearching for compatible SAEs...")
        gpt2_saes = analyzer.find_matching_saes("gpt2", "resid")
        print("\nAvailable GPT-2 SAEs:")
        print(gpt2_saes.to_string(index=False))
        
        # Step 2: Load SAE
        print("\n" + "─" * 40)
        print("Step 2: Loading SAE")
        print("─" * 40)
        
        sae = analyzer.load_sae()
        
        # Step 3: Load your activations
        print("\n" + "─" * 40)
        print("Step 3: Loading Pre-computed Activations")
        print("─" * 40)
        
        # Get available activation files
        activation_files = list(Path("activations").glob("*.pt"))
        if not activation_files:
            print("No activation files found in ./activations/")
            return
            
        print(f"Found {len(activation_files)} activation files:")
        for i, file_path in enumerate(activation_files):
            print(f"  {i+1}. {file_path.name}")
        
        # Use the first activation file as example
        activation_path = activation_files[0]
        print(f"\nUsing: {activation_path}")
        
        activations = analyzer.load_activations(str(activation_path))
        
        # Step 4: Process activations through SAE
        print("\n" + "─" * 40)
        print("Step 4: Processing Through SAE")
        print("─" * 40)
        
        # Check if activations are compatible with SAE
        print(f"Activation shape: {activations.shape}")
        print(f"SAE expects d_in: {sae.cfg.d_in}")
        
        # Handle dimension mismatch
        if len(activations.shape) >= 2:
            actual_dim = activations.shape[-1]
            if actual_dim != sae.cfg.d_in:
                print(f"\nWarning: Dimension mismatch!")
                print(f"  Activations: {actual_dim}, SAE expects: {sae.cfg.d_in}")
                
                if actual_dim > sae.cfg.d_in:
                    print(f"  Truncating to first {sae.cfg.d_in} dimensions")
                    activations = activations[..., :sae.cfg.d_in]
                else:
                    print(f"  Padding to {sae.cfg.d_in} dimensions")
                    padding = torch.zeros(*activations.shape[:-1], sae.cfg.d_in - actual_dim)
                    activations = torch.cat([activations, padding], dim=-1)
        
        # Process through SAE
        results = analyzer.process_activations(activations)
        
        # Step 5: Find top features
        print("\n" + "─" * 40)
        print("Step 5: Analyzing Top Features")
        print("─" * 40)
        
        values, indices = analyzer.find_top_features(results['feature_activations'])
        
        # Step 6: Neuronpedia Integration
        print("\n" + "─" * 40)
        print("Step 6: Neuronpedia Integration")
        print("─" * 40)
        
        # Generate dashboard URLs for top features
        print(f"\nNeuronpedia Dashboard URLs for top {config.top_k_features} features:")
        for i, (val, idx) in enumerate(zip(values[:5], indices[:5])):  # Show top 5
            url = analyzer.get_neuronpedia_dashboard_url(int(idx))
            print(f"  {i+1}. Feature {idx} (activation: {val:.4f})")
            print(f"     Dashboard: {url}")
        
        # Try to download feature explanations
        print("\nAttempting to download feature explanations...")
        try:
            explanations = analyzer.download_feature_explanations("gpt2-small", "7-res-jb")
            if explanations is not None:
                print(f"Downloaded {len(explanations)} feature explanations")
                
                # Search for interesting patterns
                print("\nSearching for features related to cognitive patterns:")
                
                search_terms = ["thought", "emotion", "pattern", "cognitive", "mental", "mind"]
                for term in search_terms:
                    matches = analyzer.search_features_by_description(term)
                    if matches is not None and len(matches) > 0:
                        print(f"\n  Features related to '{term}': {len(matches)}")
                        if len(matches) > 0:
                            print(f"    Example: Feature {matches.iloc[0]['feature']} - {matches.iloc[0]['description'][:100]}...")
            
        except Exception as e:
            print(f"Could not download feature explanations: {e}")
        
        # Step 7: Generate visualizations
        print("\n" + "─" * 40)
        print("Step 7: Generating Visualizations")
        print("─" * 40)
        
        fig = analyzer.visualize_feature_activations(results['feature_activations'], 
                                                   title="Cognitive Pattern Activation Analysis")
        
        # Save visualization
        viz_path = Path(config.analysis_output_dir) / "activation_analysis.html"
        fig.write_html(viz_path)
        print(f"Visualization saved to: {viz_path}")
        
        # Step 8: Save results and generate report
        print("\n" + "─" * 40)
        print("Step 8: Saving Results")
        print("─" * 40)
        
        # Save analysis results
        results_path = analyzer.save_analysis_results(results, 
                                                     f"cognitive_analysis_{activation_path.stem}.pt")
        
        # Generate comprehensive report
        report_path = analyzer.generate_analysis_report(results, (values, indices),
                                                       f"cognitive_analysis_report_{activation_path.stem}.md")
        
        # Create a summary of the analysis
        summary = {
            "activation_file": str(activation_path),
            "sae_config": {
                "release": config.sae_release,
                "sae_id": config.sae_id,
                "d_in": sae.cfg.d_in,
                "d_sae": sae.cfg.d_sae
            },
            "activation_stats": {
                "original_shape": list(activations.shape),
                "avg_sparsity": float(results['l0_norm'].mean()),
                "avg_reconstruction_error": float(results['reconstruction_error'].mean())
            },
            "top_features": [
                {"feature_idx": int(idx), "activation_value": float(val)}
                for val, idx in zip(values, indices)
            ]
        }
        
        summary_path = Path(config.analysis_output_dir) / f"analysis_summary_{activation_path.stem}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis summary saved to: {summary_path}")
        
        # Step 9: Advanced Analysis Example
        print("\n" + "─" * 40)
        print("Step 9: Advanced Analysis Examples")
        print("─" * 40)
        
        # Feature ablation example
        print("Demonstrating feature ablation...")
        top_3_features = [int(idx) for idx in indices[:3]]
        ablated_recon = analyzer.perform_feature_ablation(results['feature_activations'], 
                                                         top_3_features)
        
        original_recon_error = torch.nn.functional.mse_loss(activations, results['reconstructions'])
        ablated_recon_error = torch.nn.functional.mse_loss(activations, ablated_recon)
        
        print(f"  Original reconstruction error: {original_recon_error:.6f}")
        print(f"  After ablating top 3 features: {ablated_recon_error:.6f}")
        print(f"  Error increase: {(ablated_recon_error - original_recon_error):.6f}")
        
        # Steering example
        print("\nDemonstrating feature steering...")
        steering_vectors = [(int(indices[0]), 1.5), (int(indices[1]), -0.8)]
        steered_activations = analyzer.apply_steering(activations, steering_vectors)
        steered_results = analyzer.process_activations(steered_activations)
        
        print(f"  Original avg L0: {results['l0_norm'].mean():.2f}")
        print(f"  After steering avg L0: {steered_results['l0_norm'].mean():.2f}")
        
        print("\n" + "=" * 60)
        print("SAE Analysis Complete!")
        print("=" * 60)
        print(f"\nResults saved in: {config.analysis_output_dir}/")
        print("\nKey files:")
        print(f"  - Analysis results: {results_path.name}")
        print(f"  - Report: {report_path.name}")
        print(f"  - Summary: {summary_path.name}")
        print(f"  - Visualization: {viz_path.name}")
        
        print(f"\nNext steps:")
        print(f"  1. Open the HTML visualization in your browser")
        print(f"  2. Visit the Neuronpedia URLs to explore feature interpretations")
        print(f"  3. Use the summary JSON to understand which features are most active")
        print(f"  4. Run analysis on other activation files to compare patterns")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print(f"Stack trace:")
        import traceback
        traceback.print_exc()
        
        # Provide troubleshooting guidance
        print(f"\nTroubleshooting tips:")
        print(f"  1. Install dependencies: pip install sae-lens transformer-lens")
        print(f"  2. Check activation file format and dimensions")
        print(f"  3. Try a different SAE configuration")
        print(f"  4. Ensure sufficient memory for large activation files")

def analyze_all_activations():
    """Run analysis on all activation files"""
    print("Analyzing all activation files...")
    
    activation_files = list(Path("activations").glob("*.pt"))
    
    for i, activation_file in enumerate(activation_files):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(activation_files)}: {activation_file.name}")
        print(f"{'='*60}")
        
        # You could modify main() to accept a specific file
        # For now, just note which file we would process
        print(f"Would analyze: {activation_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        analyze_all_activations()
    else:
        main()