#!/usr/bin/env python3
"""
Cognitive Pattern Analysis Main Script
Orchestrates the entire pipeline for analyzing cognitive patterns via model activations.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np

# Import our modules
from model_loader import ModelLoader
from activation_capture import ActivationCapturer
from data.data_loader import DataLoader
from analysis.pca_analysis import PCAAnalyzer
from analysis.sae_interface import SAEInterface
from analysis.interpretation import SelfieInterpreter, ActivationArithmetic
from utils.device_detection import get_device_manager, detect_and_print_devices


class CognitivePatternAnalyzer:
    """Main orchestrator for cognitive pattern analysis."""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
        # Initialize device manager
        self.device_manager = get_device_manager()
        logging.info("Device detection results:")
        self.device_manager.print_device_info()
        
        # Initialize components
        self.data_loader = None
        self.activation_capturer = None
        self.pca_analyzer = None
        self.sae_interface = None
        self.selfie_interpreter = None
        self.activation_arithmetic = None
        
        # Storage
        self.activations = {}
        self.results = {}
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config['logging']['level'])
        log_file = self.config['logging'].get('log_file')
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        
    def setup_directories(self) -> None:
        """Create necessary directories."""
        dirs_to_create = [
            self.config['capture'].get('activation_save_path', './activations/'),
            self.config['output'].get('results_path', './results/'),
            self.config['output'].get('figures_path', './figures/'),
            os.path.dirname(self.config['logging'].get('log_file', './logs/analysis.log'))
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def initialize_components(self) -> None:
        """Initialize all analysis components."""
        logging.info("Initializing components...")
        
        # Data loader
        self.data_loader = DataLoader(
            base_path=self.config['data']['base_path']
        )
        
        # Activation capturer
        self.activation_capturer = ActivationCapturer(
            model_name=self.config['model']['name'],
            device=self.config['model']['device']
        )
        
        # Load model
        local_path = self.config['model'].get('local_path')
        self.activation_capturer.load_model(local_path)
        
        # Analysis components
        if "pca" in self.config['analysis']['methods']:
            self.pca_analyzer = PCAAnalyzer(
                n_components=self.config['analysis']['pca']['n_components']
            )
            
        if "sae" in self.config['analysis']['methods']:
            self.sae_interface = SAEInterface(
                sae_model_path=self.config['analysis']['sae']['model_path']
            )
            
        if "selfie" in self.config['analysis']['methods']:
            self.selfie_interpreter = SelfieInterpreter(
                model=self.activation_capturer.model
            )
            
        if "arithmetic" in self.config['analysis']['methods']:
            self.activation_arithmetic = ActivationArithmetic()
            
        logging.info("Components initialized successfully")
    
    def load_data(self) -> Dict[str, List[str]]:
        """Load cognitive pattern datasets."""
        logging.info("Loading cognitive pattern data...")
        
        pattern_names = self.config['data']['cognitive_patterns']
        cognitive_patterns = self.data_loader.load_cognitive_patterns(pattern_names)
        
        # Log data statistics
        stats = self.data_loader.get_pattern_stats()
        for pattern, stat in stats.items():
            logging.info(f"Pattern '{pattern}': {stat['count']} samples, "
                        f"avg length: {stat['avg_length']:.1f}")
        
        return cognitive_patterns
    
    def capture_activations(self, cognitive_patterns: Dict[str, List[str]]) -> None:
        """Capture activations for all cognitive patterns."""
        logging.info("Capturing activations...")
        
        layers = self.config['model']['layers']
        position = self.config['capture']['position']
        max_samples = self.config['data'].get('max_samples_per_pattern')
        
        for pattern_name, strings in cognitive_patterns.items():
            logging.info(f"Capturing activations for pattern: {pattern_name}")
            
            # Limit samples if specified
            if max_samples and len(strings) > max_samples:
                strings = strings[:max_samples]
                logging.info(f"Limited to {max_samples} samples")
            
            # Capture activations
            activations = self.activation_capturer.capture_activations(
                strings=strings,
                layer_nums=layers,
                cognitive_pattern=pattern_name,
                position=position
            )
            
            self.activations[pattern_name] = activations
            
            # Save activations if requested
            if self.config['capture']['save_activations']:
                save_path = os.path.join(
                    self.config['capture']['activation_save_path'],
                    f"{pattern_name}_activations.pt"
                )
                self.activation_capturer.save_activations(save_path, pattern_name)
    
    def run_pca_analysis(self) -> None:
        """Run PCA analysis on captured activations."""
        if not self.pca_analyzer:
            return
            
        logging.info("Running PCA analysis...")
        
        for pattern_name, activations in self.activations.items():
            logging.info(f"PCA analysis for pattern: {pattern_name}")
            
            # Compute PCA
            pca_results = self.pca_analyzer.compute_pca(
                activations=activations,
                pattern_name=pattern_name,
                standardize=self.config['analysis']['pca']['standardize']
            )
            
            if pattern_name not in self.results:
                self.results[pattern_name] = {}
            self.results[pattern_name]['pca'] = pca_results
            
            # Plot if requested
            if self.config['output']['plot_figures']:
                for layer_key in activations.keys():
                    fig_path = os.path.join(
                        self.config['output']['figures_path'],
                        f"{pattern_name}_{layer_key}_pca_variance.png"
                    )
                    self.pca_analyzer.plot_explained_variance(
                        pattern_name, layer_key, fig_path
                    )
    
    def run_sae_analysis(self) -> None:
        """Run SAE analysis on captured activations."""
        if not self.sae_interface:
            return
            
        logging.info("Running SAE analysis...")
        
        for pattern_name, activations in self.activations.items():
            logging.info(f"SAE analysis for pattern: {pattern_name}")
            
            sae_results = self.sae_interface.analyze_pattern_features(
                activations=activations,
                pattern_name=pattern_name,
                top_k=self.config['analysis']['sae']['top_k_features']
            )
            
            if pattern_name not in self.results:
                self.results[pattern_name] = {}
            self.results[pattern_name]['sae'] = sae_results
    
    def run_selfie_analysis(self, cognitive_patterns: Dict[str, List[str]]) -> None:
        """Run selfie interpretation analysis."""
        if not self.selfie_interpreter:
            return
            
        logging.info("Running selfie interpretation analysis...")
        
        for pattern_name, activations in self.activations.items():
            logging.info(f"Selfie analysis for pattern: {pattern_name}")
            
            contexts = cognitive_patterns.get(pattern_name, [])
            interpretations = self.selfie_interpreter.batch_interpret_activations(
                activations=activations,
                contexts=contexts,
                pattern_name=pattern_name
            )
            
            # Validate interpretations
            validation_results = {}
            for layer_key, layer_interpretations in interpretations.items():
                validation = self.selfie_interpreter.validate_interpretations(
                    layer_interpretations, pattern_name
                )
                validation_results[layer_key] = validation
            
            # Generate summary
            summary = self.selfie_interpreter.generate_pattern_summary(
                interpretations, pattern_name
            )
            
            if pattern_name not in self.results:
                self.results[pattern_name] = {}
            self.results[pattern_name]['selfie'] = {
                'interpretations': interpretations,
                'validation': validation_results,
                'summary': summary
            }
    
    def run_arithmetic_analysis(self) -> None:
        """Run activation arithmetic analysis."""
        if not self.activation_arithmetic:
            return
            
        logging.info("Running activation arithmetic analysis...")
        
        # Prepare activations for arithmetic
        pattern_activations = {}
        layers = self.config['model']['layers']
        
        for pattern_name, activations in self.activations.items():
            pattern_activations[pattern_name] = {}
            for layer_num in layers:
                layer_key = f"{pattern_name}_layer_{layer_num}"
                if layer_key in activations:
                    pattern_activations[pattern_name][layer_num] = activations[layer_key]
        
        # Compute similarity matrices for each layer
        arithmetic_results = {}
        for layer_num in layers:
            layer_patterns = {}
            for pattern_name, pattern_data in pattern_activations.items():
                if layer_num in pattern_data:
                    layer_patterns[pattern_name] = pattern_data[layer_num]
            
            if len(layer_patterns) > 1:
                similarities = self.activation_arithmetic.compute_similarity_matrix(
                    layer_patterns
                )
                arithmetic_results[f"layer_{layer_num}"] = {
                    'similarities': similarities
                }
        
        self.results['arithmetic'] = arithmetic_results
    
    def save_results(self) -> None:
        """Save all analysis results."""
        if not self.config['output']['save_results']:
            return
            
        logging.info("Saving results...")
        
        results_path = self.config['output']['results_path']
        
        # Save individual pattern results
        for pattern_name, pattern_results in self.results.items():
            if pattern_name != 'arithmetic':  # Skip global arithmetic results
                file_path = os.path.join(results_path, f"{pattern_name}_results.yaml")
                with open(file_path, 'w') as f:
                    yaml.dump(pattern_results, f, default_flow_style=False)
        
        # Save global results
        global_results_path = os.path.join(results_path, "global_results.yaml")
        with open(global_results_path, 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)
    
    def run_full_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        logging.info("Starting cognitive pattern analysis pipeline...")
        
        try:
            # Initialize
            self.initialize_components()
            
            # Load data
            cognitive_patterns = self.load_data()
            
            # Capture activations
            self.capture_activations(cognitive_patterns)
            
            # Run analyses
            self.run_pca_analysis()
            self.run_sae_analysis()
            self.run_selfie_analysis(cognitive_patterns)
            self.run_arithmetic_analysis()
            
            # Save results
            self.save_results()
            
            logging.info("Analysis pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            raise
    
    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        print("\n" + "="*50)
        print("COGNITIVE PATTERN ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"Model: {self.config['model']['name']}")
        print(f"Layers analyzed: {self.config['model']['layers']}")
        print(f"Analysis methods: {self.config['analysis']['methods']}")
        
        print("\nPatterns analyzed:")
        for pattern_name in self.activations.keys():
            activation_keys = list(self.activations[pattern_name].keys())
            print(f"  - {pattern_name}: {len(activation_keys)} layer combinations")
        
        print(f"\nResults saved to: {self.config['output']['results_path']}")
        if self.config['output']['plot_figures']:
            print(f"Figures saved to: {self.config['output']['figures_path']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cognitive Pattern Analysis")
    parser.add_argument(
        "--config", 
        default="./config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        help="Specific patterns to analyze (overrides config)"
    )
    parser.add_argument(
        "--methods",
        nargs="+", 
        choices=["pca", "sae", "selfie", "arithmetic"],
        help="Analysis methods to run (overrides config)"
    )
    parser.add_argument(
        "--check-devices",
        action="store_true",
        help="Check available devices and exit"
    )
    
    args = parser.parse_args()
    
    # Check devices and exit if requested
    if args.check_devices:
        print("Detecting available compute devices...")
        detect_and_print_devices()
        return
    
    # Create analyzer
    analyzer = CognitivePatternAnalyzer(config_path=args.config)
    
    # Override config if command line arguments provided
    if args.patterns:
        analyzer.config['data']['cognitive_patterns'] = args.patterns
    if args.methods:
        analyzer.config['analysis']['methods'] = args.methods
    
    # Run analysis
    analyzer.run_full_analysis()
    analyzer.print_summary()


if __name__ == "__main__":
    main()