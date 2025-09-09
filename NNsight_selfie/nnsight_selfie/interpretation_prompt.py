"""
InterpretationPrompt class for managing interpretation templates with placeholders.

This module provides functionality to create and manage prompts with special
placeholder tokens that can be replaced with activation vectors during interpretation.
"""

from typing import List, Union, Any
import torch


class InterpretationPrompt:
    """
    A class for managing interpretation prompts with placeholder tokens.
    
    This class handles prompts that contain special markers (represented as non-string objects)
    which will be replaced with activation vectors during interpretation.
    
    Args:
        tokenizer: The tokenizer to use for encoding the prompt
        interpretation_prompt_sequence: A sequence containing strings and placeholder objects
        
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> sequence = ["This represents the concept of ", None, " which means"]
        >>> prompt = InterpretationPrompt(tokenizer, sequence)
    """
    
    def __init__(self, tokenizer, interpretation_prompt_sequence: List[Union[str, Any]]):
        self.tokenizer = tokenizer
        self.interpretation_prompt = ""
        self.insert_locations = []
        
        # Build the prompt and track placeholder locations
        for part in interpretation_prompt_sequence:
            if isinstance(part, str):
                # Add string part to prompt
                self.interpretation_prompt += part
            else:
                # Handle placeholder - replace with special token
                insert_start = len(self.tokenizer.encode(self.interpretation_prompt))
                self.interpretation_prompt += "_ "  # Placeholder token
                insert_end = len(self.tokenizer.encode(self.interpretation_prompt))
                
                # Track all token positions for this placeholder
                for insert_idx in range(insert_start, insert_end):
                    self.insert_locations.append(insert_idx)
        
        # Tokenize the final prompt
        self.interpretation_prompt_inputs = self.tokenizer(
            self.interpretation_prompt, 
            return_tensors="pt"
        )
    
    def get_prompt(self) -> str:
        """Get the formatted interpretation prompt."""
        return self.interpretation_prompt
    
    def get_insert_locations(self) -> List[int]:
        """Get the token positions where activations should be injected."""
        return self.insert_locations
    
    def get_tokenized_inputs(self) -> dict:
        """Get the tokenized inputs for the interpretation prompt."""
        return self.interpretation_prompt_inputs
    
    def __repr__(self) -> str:
        return f"InterpretationPrompt('{self.interpretation_prompt}', insert_at={self.insert_locations})"
    
    @classmethod
    def create_simple(cls, tokenizer, prefix: str = "", suffix: str = "", placeholder_token: str = "_"):
        """
        Create a simple interpretation prompt with a single placeholder.
        
        Args:
            tokenizer: Tokenizer to use
            prefix: Text before the placeholder
            suffix: Text after the placeholder  
            placeholder_token: Token to use as placeholder
            
        Returns:
            InterpretationPrompt instance
        """
        sequence = []
        if prefix:
            sequence.append(prefix)
        sequence.append(None)  # Placeholder
        if suffix:
            sequence.append(suffix)
            
        return cls(tokenizer, sequence)
    
    @classmethod
    def create_concept_prompt(cls, tokenizer):
        """
        Create a standard concept interpretation prompt.
        
        Args:
            tokenizer: Tokenizer to use
            
        Returns:
            InterpretationPrompt for concept interpretation
        """
        return cls.create_simple(
            tokenizer,
            prefix="This represents the concept of ",
            suffix=""
        )
    
    @classmethod  
    def create_sentiment_prompt(cls, tokenizer):
        """
        Create a sentiment interpretation prompt.
        
        Args:
            tokenizer: Tokenizer to use
            
        Returns:
            InterpretationPrompt for sentiment interpretation
        """
        return cls.create_simple(
            tokenizer,
            prefix="This expresses the sentiment of ",
            suffix=""
        )
    
    @classmethod
    def create_entity_prompt(cls, tokenizer):
        """
        Create an entity interpretation prompt.
        
        Args:
            tokenizer: Tokenizer to use
            
        Returns:
            InterpretationPrompt for entity interpretation  
        """
        return cls.create_simple(
            tokenizer,
            prefix="This refers to the entity ",
            suffix=""
        )
    
    def validate_with_model(self, model) -> bool:
        """
        Validate that the prompt works with a given model.
        
        Args:
            model: The model to validate against
            
        Returns:
            True if validation passes
        """
        try:
            # Test that the prompt can be tokenized
            inputs = self.get_tokenized_inputs()
            
            # Test that insert locations are valid
            seq_len = inputs['input_ids'].shape[1]
            for loc in self.insert_locations:
                if loc >= seq_len:
                    return False
                    
            return True
        except Exception:
            return False