"""
Multi-Text Activation Aggregation for SelfIE

This module implements advanced aggregation methods for combining activations 
from multiple text inputs while preserving semantic information.

Core Aggregation Methods:
1. Attention-Weighted Pooling - Uses learned attention to weight semantic importance
2. Weighted Mean Pooling - Learnable weights for different positions/heads  
3. Multi-Scale Aggregation - Combines mean + max pooling for comprehensive features

Semantic Preservation Strategies:
- Preserves directional information in activation space
- Handles residual stream properties correctly
- Uses similarity-based weighting for semantic coherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum
import pandas as pd
import numpy as np


class AggregationStrategy(Enum):
    """Available aggregation strategies for multi-text activation combining"""
    SIMPLE_MEAN = "simple_mean"
    WEIGHTED_MEAN = "weighted_mean" 
    ATTENTION_WEIGHTED = "attention_weighted"
    MULTI_SCALE = "multi_scale"
    SIMILARITY_WEIGHTED = "similarity_weighted"
    RESIDUAL_PRESERVING = "residual_preserving"
    PRINCIPAL_COMPONENT = "principal_component"


class SemanticAggregator(nn.Module):
    """
    Learnable aggregation module that preserves semantic information
    when combining activations from multiple texts.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Attention-based pooling components
        self.attention_query = nn.Parameter(torch.randn(hidden_dim))
        self.attention_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Multi-head attention for complex semantic relationships
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Position-aware weighting
        self.position_embeddings = nn.Parameter(torch.randn(512, hidden_dim))  # Max 512 texts
        
        # Similarity kernel for semantic coherence
        self.similarity_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Layer normalization for stable gradients
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def attention_weighted_pooling(self, activations: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Use learned attention to weight activations based on semantic importance.
        
        Args:
            activations: [batch_size, num_texts, hidden_dim] tensor
            mask: Optional mask for valid texts
            
        Returns:
            Aggregated activation: [batch_size, hidden_dim]
        """
        # Project activations to attention space
        projected = self.attention_proj(activations)  # [B, N, D]
        
        # Compute attention scores using learnable query
        query = self.attention_query.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        scores = torch.matmul(projected, query.transpose(-1, -2)).squeeze(-1)  # [B, N]
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(~mask, float('-inf'))
            
        # Softmax for attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, N]
        attention_weights = self.attention_dropout(attention_weights)
        
        # Weighted sum
        weighted = torch.sum(activations * attention_weights.unsqueeze(-1), dim=1)  # [B, D]
        return self.layer_norm(weighted)
    
    def multi_scale_aggregation(self, activations: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Combine mean and max pooling to capture both averaged and peak features.
        
        Args:
            activations: [batch_size, num_texts, hidden_dim] tensor
            mask: Optional mask for valid texts
            
        Returns:
            Multi-scale aggregated activation: [batch_size, hidden_dim * 2]
        """
        if mask is not None:
            # Apply mask before pooling
            activations = activations.masked_fill(~mask.unsqueeze(-1), 0)
            
        # Mean pooling (averaged features)
        if mask is not None:
            mean_pool = activations.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        else:
            mean_pool = activations.mean(dim=1)  # [B, D]
            
        # Max pooling (peak features)  
        max_pool, _ = activations.max(dim=1)  # [B, D]
        
        # Concatenate multi-scale features
        combined = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2*D]
        return self.layer_norm(combined[:, :self.hidden_dim])  # Project back to original dim
    
    def similarity_weighted_pooling(self, activations: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Weight activations based on semantic similarity using kernel methods.
        
        Args:
            activations: [batch_size, num_texts, hidden_dim] tensor
            mask: Optional mask for valid texts
            
        Returns:
            Similarity-weighted activation: [batch_size, hidden_dim]
        """
        # Project to similarity space
        projected = self.similarity_proj(activations)  # [B, N, D/2]
        
        # Compute pairwise similarities
        similarities = torch.matmul(projected, projected.transpose(-1, -2))  # [B, N, N]
        
        # Average similarity as importance weight
        importance = similarities.mean(dim=-1)  # [B, N]
        
        # Apply mask if provided
        if mask is not None:
            importance.masked_fill_(~mask, float('-inf'))
            
        # Softmax for weights
        weights = F.softmax(importance, dim=-1)  # [B, N]
        
        # Weighted combination
        weighted = torch.sum(activations * weights.unsqueeze(-1), dim=1)  # [B, D]
        return self.layer_norm(weighted)
    
    def residual_preserving_aggregation(self, activations: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate while preserving residual stream properties and directional information.
        
        Args:
            activations: [batch_size, num_texts, hidden_dim] tensor
            mask: Optional mask for valid texts
            
        Returns:
            Residual-preserving aggregated activation: [batch_size, hidden_dim]
        """
        # Use multi-head attention to preserve complex relationships
        if mask is not None:
            # Convert mask to attention mask format
            attn_mask = ~mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # [B, N, N]
        else:
            attn_mask = None
            
        # Self-attention to capture relationships between texts
        attn_output, _ = self.multihead_attn(
            activations, activations, activations, 
            key_padding_mask=~mask if mask is not None else None
        )
        
        # Residual connection to preserve original information
        residual_output = activations + attn_output
        
        # Mean pool with residual connection
        if mask is not None:
            pooled = (residual_output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        else:
            pooled = residual_output.mean(dim=1)
            
        return self.layer_norm(pooled)


class MultiTextActivationExtractor:
    """
    Extracts and aggregates activations from multiple text inputs for SelfIE interpretation.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer 
        self.device = device
        
        # Initialize semantic aggregator
        hidden_dim = getattr(model.config, 'd_model', 
                           getattr(model.config, 'hidden_size', 4096))
        self.aggregator = SemanticAggregator(hidden_dim).to(device)
        
    def extract_multi_text_activations(
        self, 
        texts: List[str],
        layers_to_extract: List[int],
        token_positions: Optional[List[int]] = None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.ATTENTION_WEIGHTED
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Extract activations from multiple texts and aggregate them.
        
        Args:
            texts: List of input texts
            layers_to_extract: Which layers to extract from
            token_positions: Which token positions (if None, uses last tokens)
            aggregation_strategy: How to aggregate across texts
            
        Returns:
            Dictionary mapping (layer, token_pos) -> aggregated_activation
        """
        all_activations = {}
        
        # Extract activations from each text
        for text in texts:
            text_activations = self._extract_single_text_activations(
                text, layers_to_extract, token_positions
            )
            
            # Organize by (layer, position) keys
            for (layer, pos), activation in text_activations.items():
                if (layer, pos) not in all_activations:
                    all_activations[(layer, pos)] = []
                all_activations[(layer, pos)].append(activation)
        
        # Aggregate activations for each (layer, position) combination
        aggregated_activations = {}
        for key, activation_list in all_activations.items():
            # Stack activations: [num_texts, hidden_dim]
            stacked = torch.stack(activation_list, dim=0).unsqueeze(0)  # [1, N, D]
            
            # Apply aggregation strategy
            aggregated = self._apply_aggregation_strategy(
                stacked, aggregation_strategy
            ).squeeze(0)  # [D]
            
            aggregated_activations[key] = aggregated
            
        return aggregated_activations
    
    def _extract_single_text_activations(
        self, 
        text: str, 
        layers_to_extract: List[int],
        token_positions: Optional[List[int]] = None
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """Extract activations from a single text."""
        from selfie.generate_wrappers import model_forward_interpret
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        seq_len = inputs['input_ids'].shape[-1]
        
        # Default to last few tokens if positions not specified
        if token_positions is None:
            token_positions = list(range(max(0, seq_len-3), seq_len))
        
        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = model_forward_interpret(
                self.model,
                **inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True,
            )
        
        # Extract activations
        activations = {}
        for layer_idx in layers_to_extract:
            for pos in token_positions:
                if pos < seq_len:
                    activation = outputs['hidden_states'][layer_idx][0][pos]  # [D]
                    activations[(layer_idx, pos)] = activation
                    
        return activations
    
    def _apply_aggregation_strategy(
        self, 
        activations: torch.Tensor, 
        strategy: AggregationStrategy
    ) -> torch.Tensor:
        """Apply the specified aggregation strategy."""
        if strategy == AggregationStrategy.SIMPLE_MEAN:
            return activations.mean(dim=1)
        
        elif strategy == AggregationStrategy.ATTENTION_WEIGHTED:
            return self.aggregator.attention_weighted_pooling(activations)
        
        elif strategy == AggregationStrategy.MULTI_SCALE:
            return self.aggregator.multi_scale_aggregation(activations)
        
        elif strategy == AggregationStrategy.SIMILARITY_WEIGHTED:
            return self.aggregator.similarity_weighted_pooling(activations)
        
        elif strategy == AggregationStrategy.RESIDUAL_PRESERVING:
            return self.aggregator.residual_preserving_aggregation(activations)

        elif strategy == AggregationStrategy.PRINCIPAL_COMPONENT:
            return self._principal_component_direction(activations)
        
        elif strategy == AggregationStrategy.WEIGHTED_MEAN:
            # Simple learnable weighted mean
            weights = F.softmax(torch.randn(activations.size(1)).to(activations.device), dim=0)
            return torch.sum(activations * weights.unsqueeze(0).unsqueeze(-1), dim=1)
        
        else:
            return activations.mean(dim=1)  # Fallback to simple mean

    def _principal_component_direction(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute the first principal component (direction of maximum variance)
        for a single set of activations.

        Args:
            activations: Tensor of shape [batch_size, num_texts, hidden_dim].

        Returns:
            Tensor of shape [batch_size, hidden_dim] representing the first
            principal component per batch. For typical usage here, batch_size=1.
        """
        # Ensure float computation for SVD stability
        X = activations
        B, N, D = X.shape
        pcs = []
        for b in range(B):
            # Center across the examples dimension
            Xb = X[b]
            Xb = Xb - Xb.mean(dim=0, keepdim=True)
            # Handle degenerate cases
            if N == 1:
                # With a single example, direction is undefined; return normalized vector
                v = Xb.squeeze(0)
                norm = torch.norm(v) + 1e-12
                pcs.append(v / norm)
                continue
            try:
                # Compute SVD in feature space; returns U, S, Vh with shapes [N,N], [min(N,D)], [D,D] (reduced)
                U, S, Vh = torch.linalg.svd(Xb, full_matrices=False)
                pc1 = Vh[0]
            except RuntimeError:
                # Fallback to CPU if GPU SVD is unavailable for current dtype/size
                U, S, Vh = torch.linalg.svd(Xb.cpu(), full_matrices=False)
                pc1 = Vh[0].to(Xb.device)
            # Normalize for stability (direction only)
            pc1 = pc1 / (torch.norm(pc1) + 1e-12)
            pcs.append(pc1)
        return torch.stack(pcs, dim=0)