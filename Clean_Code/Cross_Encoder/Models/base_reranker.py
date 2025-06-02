"""
Base Reranker interface for the Cross_Encoder module.
Defines the interface that all rerankers must implement.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod

class BaseReranker(ABC):
    """
    Abstract base class for all cross-encoder rerankers.
    
    A reranker takes a query and a list of candidates and re-ranks the candidates
    based on a more sophisticated cross-encoder scoring approach.
    """
    
    @abstractmethod
    def name(self) -> str:
        """
        Return a unique identifier for this reranker.
        
        Returns:
            str: A unique identifier for the reranker
        """
        pass
    
    @abstractmethod
    def score_pair(self, query: str, candidate: str) -> float:
        """
        Score a single query-candidate pair.
        
        Args:
            query: The query string
            candidate: The candidate string
            
        Returns:
            float: A similarity score between 0 and 1
            
        Raises:
            ValueError: If the input strings are not valid
        """
        pass
    
    @abstractmethod
    def score_pairs(self, pairs: List[List[str]]) -> List[float]:
        """
        Score a list of query-candidate pairs.
        
        Args:
            pairs: A list of [query, candidate] pairs
            
        Returns:
            List[float]: A list of similarity scores
            
        Raises:
            ValueError: If the input pairs are not valid
        """
        pass
    
    @abstractmethod
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
              batch_size: int = 32, debug: bool = False) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using cross-encoder scores.
        
        Args:
            query: The query string to match
            candidates: List of candidate matches (each with 'usda_code' and 'similarity' fields)
            batch_size: Batch size for cross-encoder predictions
            debug: Whether to print debug information
            
        Returns:
            List[Dict[str, Any]]: Re-ranked list of candidates with updated similarity scores
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If the reranking process fails
        """
        pass

def check_is_reranker(obj: Any) -> bool:
    """
    Check if an object is a valid reranker (implements BaseReranker).
    
    Args:
        obj: Object to check
        
    Returns:
        bool: True if the object is a valid reranker, False otherwise
    """
    return isinstance(obj, BaseReranker)

# Test code
if __name__ == "__main__":
    print("This module defines the BaseReranker abstract class.")
    print("It cannot be instantiated directly and should be inherited by concrete reranker classes.")
