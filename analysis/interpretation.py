import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from transformer_lens import HookedTransformer


class SelfieInterpreter:
    """Selfie interpretation method - model interprets its own activations."""
    
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.interpretation_cache = {}
        
    def interpret_activation(
        self, 
        activation: torch.Tensor, 
        context_prompt: str = "",
        layer_name: str = "",
        position: str = "last"
    ) -> str:
        """
        Use the model to interpret its own activation.
        
        Args:
            activation: Activation tensor to interpret
            context_prompt: Original context that produced the activation
            layer_name: Name of the layer
            position: Position in sequence
            
        Returns:
            Model's interpretation of the activation
        """
        # Create interpretation prompt
        interpretation_prompt = self._create_interpretation_prompt(
            context_prompt, layer_name, position
        )
        
        # Generate with activation steering
        interpretation = self._generate_with_activation_steering(
            interpretation_prompt, activation, layer_name
        )
        
        return interpretation
    
    def _create_interpretation_prompt(
        self, 
        context: str, 
        layer_name: str, 
        position: str
    ) -> str:
        """Create a prompt for activation interpretation."""
        base_prompt = f"""You are analyzing your own internal representations. 

Original context: "{context}"
Layer: {layer_name}
Position: {position}

The activation vector at this layer/position represents some cognitive pattern or concept. Based on the original context and your internal processing, what cognitive pattern, concept, or type of thinking does this activation most likely represent?

Be specific and focus on:
1. What type of cognitive processing is happening
2. What concepts or patterns are being activated
3. What this means for understanding the input

Interpretation:"""
        
        return base_prompt
    
    def _generate_with_activation_steering(
        self, 
        prompt: str, 
        activation: torch.Tensor, 
        layer_name: str
    ) -> str:
        """
        Generate text while steering with the activation.
        
        Args:
            prompt: Interpretation prompt
            activation: Activation to inject
            layer_name: Layer to inject at
            
        Returns:
            Generated interpretation
        """
        # PLACEHOLDER: Real implementation would use activation injection
        # For now, generate normally and simulate steering effect
        
        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        
        # Generate without steering first (baseline)
        with torch.no_grad():
            baseline_logits = self.model(tokens)
        
        # PLACEHOLDER: Actual steering would modify forward pass
        # For now, simulate by generating normal text
        generated_tokens = self.model.generate(
            tokens,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            stop_at_eos=True
        )
        
        # Extract just the generated part
        generated_text = self.model.to_string(generated_tokens[0][tokens.shape[1]:])
        
        # Add placeholder indication of steering effect
        activation_magnitude = torch.norm(activation).item()
        steering_note = f"\n[Activation magnitude: {activation_magnitude:.2f}]"
        
        return generated_text + steering_note
    
    def batch_interpret_activations(
        self, 
        activations: Dict[str, torch.Tensor], 
        contexts: List[str],
        pattern_name: str
    ) -> Dict[str, List[str]]:
        """
        Interpret multiple activations for a cognitive pattern.
        
        Args:
            activations: Dictionary of activations by layer
            contexts: Original context strings
            pattern_name: Name of cognitive pattern
            
        Returns:
            Dictionary of interpretations by layer
        """
        results = {}
        
        for layer_key, layer_activations in activations.items():
            print(f"Interpreting activations for {pattern_name} - {layer_key}")
            layer_interpretations = []
            
            for i, activation in enumerate(layer_activations):
                context = contexts[i] if i < len(contexts) else ""
                
                interpretation = self.interpret_activation(
                    activation, context, layer_key
                )
                layer_interpretations.append(interpretation)
            
            results[layer_key] = layer_interpretations
        
        return results
    
    def validate_interpretations(
        self, 
        interpretations: List[str], 
        expected_pattern: str
    ) -> Dict[str, float]:
        """
        Validate interpretations against expected cognitive pattern.
        
        Args:
            interpretations: List of interpretation strings
            expected_pattern: Expected cognitive pattern name
            
        Returns:
            Validation metrics
        """
        # PLACEHOLDER: Real implementation would use more sophisticated validation
        
        # Simple keyword matching for validation
        pattern_keywords = {
            "anxiety": ["worry", "fear", "anxious", "stress", "concern", "nervous"],
            "depression": ["sad", "hopeless", "negative", "down", "despair"],
            "reasoning": ["logic", "analyze", "conclude", "infer", "deduce", "rational"],
            "emotional": ["feel", "emotion", "mood", "affect", "sentiment"],
            "narrative": ["story", "sequence", "plot", "timeline", "narrative"]
        }
        
        keywords = pattern_keywords.get(expected_pattern.lower(), [])
        
        matches = 0
        total_length = 0
        
        for interpretation in interpretations:
            interpretation_lower = interpretation.lower()
            total_length += len(interpretation)
            
            for keyword in keywords:
                if keyword in interpretation_lower:
                    matches += 1
                    break
        
        return {
            "keyword_match_ratio": matches / len(interpretations) if interpretations else 0,
            "avg_interpretation_length": total_length / len(interpretations) if interpretations else 0,
            "total_interpretations": len(interpretations)
        }
    
    def generate_pattern_summary(
        self, 
        interpretations: Dict[str, List[str]], 
        pattern_name: str
    ) -> str:
        """
        Generate a summary of interpretations for a cognitive pattern.
        
        Args:
            interpretations: Dictionary of interpretations by layer
            pattern_name: Name of cognitive pattern
            
        Returns:
            Summary string
        """
        summary_prompt = f"""Based on the following activation interpretations from different layers of a neural network analyzing the cognitive pattern '{pattern_name}', provide a comprehensive summary of what this pattern represents:

"""
        
        for layer_key, layer_interpretations in interpretations.items():
            summary_prompt += f"\n{layer_key} interpretations:\n"
            for i, interp in enumerate(layer_interpretations[:3]):  # Limit to first 3
                summary_prompt += f"- {interp[:200]}...\n"
        
        summary_prompt += f"""
Cognitive Pattern Summary for '{pattern_name}':
Based on the activation interpretations across layers, this pattern appears to represent:"""
        
        # Generate summary
        tokens = self.model.to_tokens(summary_prompt, prepend_bos=True)
        summary_tokens = self.model.generate(
            tokens,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True
        )
        
        summary = self.model.to_string(summary_tokens[0][tokens.shape[1]:])
        return summary


class ActivationArithmetic:
    """Perform arithmetic operations in activation space."""
    
    def __init__(self):
        self.cached_operations = {}
    
    def compute_pattern_difference(
        self, 
        pattern1_activations: torch.Tensor, 
        pattern2_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the difference between two cognitive patterns.
        
        Args:
            pattern1_activations: Activations for first pattern
            pattern2_activations: Activations for second pattern
            
        Returns:
            Difference vector in activation space
        """
        # Compute mean activations for each pattern
        mean1 = torch.mean(pattern1_activations, dim=0)
        mean2 = torch.mean(pattern2_activations, dim=0)
        
        return mean1 - mean2
    
    def add_patterns(
        self, 
        pattern1_activations: torch.Tensor, 
        pattern2_activations: torch.Tensor,
        weight1: float = 1.0,
        weight2: float = 1.0
    ) -> torch.Tensor:
        """
        Add two cognitive patterns with optional weighting.
        
        Args:
            pattern1_activations: Activations for first pattern
            pattern2_activations: Activations for second pattern
            weight1: Weight for first pattern
            weight2: Weight for second pattern
            
        Returns:
            Combined activation vector
        """
        mean1 = torch.mean(pattern1_activations, dim=0)
        mean2 = torch.mean(pattern2_activations, dim=0)
        
        return weight1 * mean1 + weight2 * mean2
    
    def find_transition_vector(
        self, 
        from_pattern: torch.Tensor, 
        to_pattern: torch.Tensor
    ) -> torch.Tensor:
        """
        Find the vector that represents transition between patterns.
        
        Args:
            from_pattern: Source pattern activations
            to_pattern: Target pattern activations
            
        Returns:
            Transition vector
        """
        return self.compute_pattern_difference(to_pattern, from_pattern)
    
    def interpolate_patterns(
        self, 
        pattern1_activations: torch.Tensor, 
        pattern2_activations: torch.Tensor,
        steps: int = 10
    ) -> List[torch.Tensor]:
        """
        Create interpolation between two cognitive patterns.
        
        Args:
            pattern1_activations: Start pattern activations
            pattern2_activations: End pattern activations
            steps: Number of interpolation steps
            
        Returns:
            List of interpolated activation vectors
        """
        mean1 = torch.mean(pattern1_activations, dim=0)
        mean2 = torch.mean(pattern2_activations, dim=0)
        
        interpolated = []
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated_vector = (1 - alpha) * mean1 + alpha * mean2
            interpolated.append(interpolated_vector)
        
        return interpolated
    
    def compute_similarity_matrix(
        self, 
        patterns: Dict[str, torch.Tensor]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute similarity matrix between all pattern pairs.
        
        Args:
            patterns: Dictionary of pattern activations
            
        Returns:
            Dictionary mapping pattern pairs to similarity scores
        """
        similarities = {}
        pattern_names = list(patterns.keys())
        
        for i, pattern1 in enumerate(pattern_names):
            for j, pattern2 in enumerate(pattern_names[i:], i):
                if pattern1 == pattern2:
                    similarities[(pattern1, pattern2)] = 1.0
                else:
                    mean1 = torch.mean(patterns[pattern1], dim=0)
                    mean2 = torch.mean(patterns[pattern2], dim=0)
                    
                    # Cosine similarity
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        mean1.unsqueeze(0), mean2.unsqueeze(0)
                    ).item()
                    
                    similarities[(pattern1, pattern2)] = cosine_sim
                    similarities[(pattern2, pattern1)] = cosine_sim
        
        return similarities