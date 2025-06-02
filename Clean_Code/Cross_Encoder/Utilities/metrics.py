"""
Evaluation metrics for the Cross_Encoder module.
Provides utilities for measuring reranking performance.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

def find_rank(target_value: str, candidates: List[Dict[str, Any]], key: str = 'usda_code') -> Optional[int]:
    """
    Find the rank (1-based) of a target value in a list of candidates.
    
    Args:
        target_value: The value to search for
        candidates: List of candidate dictionaries
        key: The key in each candidate dictionary to check
        
    Returns:
        The 1-based rank of the target value, or None if not found
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not target_value or not isinstance(target_value, str):
        raise ValueError("target_value must be a non-empty string")
    
    if not candidates:
        return None
    
    for i, candidate in enumerate(candidates):
        if key not in candidate:
            raise ValueError(f"Key '{key}' not found in candidate at index {i}")
        
        if candidate[key] == target_value:
            return i + 1  # Convert to 1-based ranking
    
    return None  # Not found

def calculate_precision_at_k(ranks: List[int], k: int = 1) -> float:
    """
    Calculate precision@k metric (proportion of ranks <= k).
    
    Args:
        ranks: List of ranks (1-based)
        k: The k value for precision@k
        
    Returns:
        float: Precision@k value between 0 and 1
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not ranks:
        return 0.0
    
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    
    # Count how many ranks are <= k
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    
    return hits / len(ranks)

def calculate_mean_reciprocal_rank(ranks: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        ranks: List of ranks (1-based)
        
    Returns:
        float: MRR value between 0 and 1
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not ranks:
        return 0.0
    
    # Calculate reciprocal ranks (1/rank), using 0 for None
    reciprocal_ranks = [1.0/rank if rank is not None else 0.0 for rank in ranks]
    
    # Return mean
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

def calculate_reranking_metrics(original_ranks: List[int], reranked_ranks: List[int]) -> Dict[str, Any]:
    """
    Calculate metrics to evaluate reranking performance.
    
    Args:
        original_ranks: List of original ranks (1-based)
        reranked_ranks: List of ranks after reranking (1-based)
        
    Returns:
        Dict with various performance metrics
        
    Raises:
        ValueError: If inputs are invalid
    """
    if len(original_ranks) != len(reranked_ranks):
        raise ValueError(
            f"Length mismatch: original_ranks has {len(original_ranks)} items, "
            f"reranked_ranks has {len(reranked_ranks)} items"
        )
    
    if not original_ranks:
        return {
            "precision@1": 0.0,
            "precision@3": 0.0,
            "precision@5": 0.0,
            "original_mrr": 0.0,
            "reranked_mrr": 0.0,
            "avg_rank_improvement": 0.0,
            "pct_improved": 0.0,
            "pct_worsened": 0.0,
            "pct_unchanged": 0.0
        }
    
    # Calculate precision@k metrics
    original_p1 = calculate_precision_at_k(original_ranks, k=1)
    original_p3 = calculate_precision_at_k(original_ranks, k=3)
    original_p5 = calculate_precision_at_k(original_ranks, k=5)
    
    reranked_p1 = calculate_precision_at_k(reranked_ranks, k=1)
    reranked_p3 = calculate_precision_at_k(reranked_ranks, k=3)
    reranked_p5 = calculate_precision_at_k(reranked_ranks, k=5)
    
    # Calculate MRR
    original_mrr = calculate_mean_reciprocal_rank(original_ranks)
    reranked_mrr = calculate_mean_reciprocal_rank(reranked_ranks)
    
    # Calculate rank changes
    rank_changes = []
    for orig, new in zip(original_ranks, reranked_ranks):
        if orig is not None and new is not None:
            rank_changes.append(orig - new)  # Positive means improvement
        else:
            rank_changes.append(0)  # No change if either is None
    
    # Count improvements, worsenings, and no changes
    improved = sum(1 for change in rank_changes if change > 0)
    worsened = sum(1 for change in rank_changes if change < 0)
    unchanged = sum(1 for change in rank_changes if change == 0)
    
    # Average rank improvement
    avg_rank_improvement = sum(rank_changes) / len(rank_changes) if rank_changes else 0.0
    
    # Calculate percentages
    total = len(rank_changes)
    pct_improved = improved / total if total > 0 else 0.0
    pct_worsened = worsened / total if total > 0 else 0.0
    pct_unchanged = unchanged / total if total > 0 else 0.0
    
    return {
        "precision@1": {
            "original": original_p1,
            "reranked": reranked_p1,
            "change": reranked_p1 - original_p1
        },
        "precision@3": {
            "original": original_p3,
            "reranked": reranked_p3,
            "change": reranked_p3 - original_p3
        },
        "precision@5": {
            "original": original_p5,
            "reranked": reranked_p5,
            "change": reranked_p5 - original_p5
        },
        "mrr": {
            "original": original_mrr,
            "reranked": reranked_mrr,
            "change": reranked_mrr - original_mrr
        },
        "avg_rank_improvement": avg_rank_improvement,
        "pct_improved": pct_improved,
        "pct_worsened": pct_worsened,
        "pct_unchanged": pct_unchanged
    }

# Test code
if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Test find_rank
    candidates = [
        {"usda_code": "Apple, raw", "similarity": 0.9},
        {"usda_code": "Banana, raw", "similarity": 0.8},
        {"usda_code": "Orange, raw", "similarity": 0.7},
    ]
    
    rank = find_rank("Banana, raw", candidates)
    print(f"Rank of 'Banana, raw': {rank}")
    assert rank == 2
    
    # Test precision@k
    ranks = [1, 3, 5, 10, None]
    p1 = calculate_precision_at_k(ranks, k=1)
    p3 = calculate_precision_at_k(ranks, k=3)
    p5 = calculate_precision_at_k(ranks, k=5)
    
    print(f"Precision@1: {p1:.2f}")
    print(f"Precision@3: {p3:.2f}")
    print(f"Precision@5: {p5:.2f}")
    
    # Test MRR
    mrr = calculate_mean_reciprocal_rank(ranks)
    print(f"MRR: {mrr:.2f}")
    
    # Test reranking metrics
    original_ranks = [3, 5, 2, 10, 1]
    reranked_ranks = [1, 2, 4, 8, 3]
    
    metrics = calculate_reranking_metrics(original_ranks, reranked_ranks)
    print("\nReranking metrics:")
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("Metrics tests passed!")
