"""
Base embedder interface for vector embedding models.

This module defines the common interface that all embedding models must implement,
ensuring consistent behavior across different embedding approaches.
"""

import numpy as np
from typing import List, Any, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol defining the interface for embedding functions."""
    
    def __call__(self, input: List[str]) -> List[np.ndarray]: ...


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models.
    
    This class defines the interface that all embedder implementations
    must follow, ensuring consistent behavior across different embedding approaches.
    """
    
    @abstractmethod
    def name(self) -> str:
        """
        Return a unique identifier for this embedder.
        
        The name should include both the embedder type and specific model,
        e.g., 'sentence_transformer_all-mpnet-base-v2' or 'openai_text-embedding-3-small'.
        
        Returns:
            str: Unique identifier for this embedder
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a vector representation.
        
        Args:
            text: The text to embed
            
        Returns:
            np.ndarray: Vector representation of the input text
        """
        pass
    
    @abstractmethod
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """
        Embed a list of text strings into vector representations.
        
        This method makes the embedder callable, allowing it to be used
        as a function for embedding multiple texts at once.
        
        Args:
            input: List of text strings to embed
            
        Returns:
            List[np.ndarray]: List of vector representations for each input text
        """
        pass


# Utility function to check if an object implements the embedder interface
def is_embedder(obj: Any) -> bool:
    """
    Check if an object implements the embedder interface.
    
    Args:
        obj: Object to check
        
    Returns:
        bool: True if the object implements the embedder interface
    """
    return isinstance(obj, EmbedderProtocol)


if __name__ == "__main__":
    # This section is for testing the implementation
    print("BaseEmbedder interface defined.")
    print("To use this module, implement a concrete embedder class that inherits from BaseEmbedder.")
