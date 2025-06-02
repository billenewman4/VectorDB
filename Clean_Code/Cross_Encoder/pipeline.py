"""
Main pipeline functions for the Cross_Encoder module.
Provides high-level functions for creating and using cross-encoder rerankers.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Import models and utilities
from Models.base_reranker import BaseReranker, check_is_reranker
from Models.sentence_reranker import SentenceReranker
from Utilities.batch_processor import process_in_batches
from Utilities.metrics import find_rank, calculate_reranking_metrics

def create_reranker(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                   cross_encoder_weight: float = 0.7,
                   embedding_weight: float = 0.3) -> BaseReranker:
    """
    Create and return an appropriate cross-encoder reranker based on the specified model.
    
    Args:
        model_name: Name of the pre-trained cross-encoder model to use
        cross_encoder_weight: Weight to apply to cross-encoder scores
        embedding_weight: Weight to apply to embedding similarity scores
        
    Returns:
        BaseReranker: An initialized reranker instance
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If reranker creation fails
    """
    try:
        return SentenceReranker(
            model_name=model_name,
            cross_encoder_weight=cross_encoder_weight,
            embedding_weight=embedding_weight
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create reranker with model {model_name}: {str(e)}")

def rerank_candidates(query: str, 
                     candidates: List[Dict[str, Any]],
                     reranker: Union[BaseReranker, str] = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                     cross_encoder_weight: float = 0.7,
                     embedding_weight: float = 0.3,
                     batch_size: int = 32,
                     debug: bool = False) -> List[Dict[str, Any]]:
    """
    Re-rank candidates using a cross-encoder.
    
    This function handles both cases where reranker is provided as an instance
    or as a model name string.
    
    Args:
        query: The query string to match
        candidates: List of candidate matches (each with 'usda_code' and 'similarity' fields)
        reranker: Either a BaseReranker instance or a model name string
        cross_encoder_weight: Weight to apply to cross-encoder scores (only used if reranker is a string)
        embedding_weight: Weight to apply to embedding similarity scores (only used if reranker is a string)
        batch_size: Batch size for cross-encoder predictions
        debug: Whether to print debug information
        
    Returns:
        List[Dict[str, Any]]: Re-ranked list of candidates with updated similarity scores
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If reranking fails
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")
    
    if not candidates:
        return []
    
    # If reranker is a string, create a reranker instance
    if isinstance(reranker, str):
        try:
            reranker = create_reranker(
                model_name=reranker,
                cross_encoder_weight=cross_encoder_weight,
                embedding_weight=embedding_weight
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create reranker from model name: {str(e)}")
    
    # Verify that reranker is a valid BaseReranker
    if not check_is_reranker(reranker):
        raise ValueError("reranker must be either a BaseReranker instance or a model name string")
    
    # Perform reranking
    return reranker.rerank(query, candidates, batch_size=batch_size, debug=debug)

def evaluate_reranking(test_queries: List[str], 
                      candidate_sets: List[List[Dict[str, Any]]],
                      correct_values: List[str],
                      reranker: Union[BaseReranker, str] = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                      cross_encoder_weight: float = 0.7,
                      embedding_weight: float = 0.3,
                      batch_size: int = 32) -> Dict[str, Any]:
    """
    Evaluate reranking performance on a test set.
    
    Args:
        test_queries: List of query strings
        candidate_sets: List of candidate sets (one per query)
        correct_values: List of correct values (one per query)
        reranker: Either a BaseReranker instance or a model name string
        cross_encoder_weight: Weight for cross-encoder scores (only used if reranker is a string)
        embedding_weight: Weight for embedding scores (only used if reranker is a string)
        batch_size: Batch size for predictions
        
    Returns:
        Dict with evaluation metrics
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If evaluation fails
    """
    if len(test_queries) != len(candidate_sets) or len(test_queries) != len(correct_values):
        raise ValueError(
            f"Length mismatch: {len(test_queries)} queries, "
            f"{len(candidate_sets)} candidate sets, "
            f"{len(correct_values)} correct values"
        )
    
    if not test_queries:
        return {}
    
    # If reranker is a string, create a reranker instance
    if isinstance(reranker, str):
        try:
            reranker = create_reranker(
                model_name=reranker,
                cross_encoder_weight=cross_encoder_weight,
                embedding_weight=embedding_weight
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create reranker from model name: {str(e)}")
    
    # Verify that reranker is a valid BaseReranker
    if not check_is_reranker(reranker):
        raise ValueError("reranker must be either a BaseReranker instance or a model name string")
    
    # Lists to store ranks
    original_ranks = []
    reranked_ranks = []
    
    # Process each query
    for i, (query, candidates, correct) in enumerate(zip(test_queries, candidate_sets, correct_values)):
        # Find original rank
        original_rank = find_rank(correct, candidates)
        original_ranks.append(original_rank)
        
        # Rerank candidates
        reranked = reranker.rerank(query, candidates, batch_size=batch_size)
        
        # Find new rank
        reranked_rank = find_rank(correct, reranked)
        reranked_ranks.append(reranked_rank)
    
    # Calculate metrics
    metrics = calculate_reranking_metrics(original_ranks, reranked_ranks)
    
    # Add reranker info
    metrics['reranker'] = reranker.name()
    metrics['cross_encoder_weight'] = reranker.cross_encoder_weight
    metrics['embedding_weight'] = reranker.embedding_weight
    
    return metrics

# Test code
if __name__ == "__main__":
    print("Testing Cross_Encoder pipeline...")
    
    # Create a test reranker
    reranker = create_reranker()
    print(f"Created reranker: {reranker.name()}")
    
    # Test candidates
    query = "fresh apple"
    candidates = [
        {"usda_code": "Apple, raw", "similarity": 0.85},
        {"usda_code": "Apple juice", "similarity": 0.76},
        {"usda_code": "Apple, dried", "similarity": 0.72},
        {"usda_code": "Pear, raw", "similarity": 0.60}
    ]
    
    # Test rerank_candidates
    reranked = rerank_candidates(query, candidates, reranker=reranker, debug=True)
    
    print("\nRe-ranked results:")
    for i, result in enumerate(reranked[:3]):
        print(f"{i+1}. {result['usda_code']} - Score: {result['similarity']:.4f}")
    
    # Test evaluation
    test_queries = ["fresh apple", "orange juice"]
    candidate_sets = [
        candidates,
        [
            {"usda_code": "Orange juice", "similarity": 0.88},
            {"usda_code": "Orange, raw", "similarity": 0.75}
        ]
    ]
    correct_values = ["Apple, raw", "Orange juice"]
    
    metrics = evaluate_reranking(
        test_queries=test_queries,
        candidate_sets=candidate_sets,
        correct_values=correct_values,
        reranker=reranker
    )
    
    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nPipeline tests passed!")
