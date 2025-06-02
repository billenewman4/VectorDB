"""Test script for the Cross_Encoder module.

This script tests all major components of the Cross_Encoder module including:
1. Basic functionality of the SentenceReranker
2. Batch processing utilities
3. Evaluation metrics
4. End-to-end reranking pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Import Cross_Encoder components
from Models.base_reranker import BaseReranker, check_is_reranker
from Models.sentence_reranker import SentenceReranker
from Utilities.batch_processor import process_in_batches
from Utilities.metrics import find_rank, calculate_reranking_metrics, calculate_precision_at_k
from pipeline import create_reranker, rerank_candidates, evaluate_reranking

def test_sentence_reranker():
    """Test the basic functionality of SentenceReranker."""
    print("\n=== Testing SentenceReranker ===")
    
    # Initialize reranker
    reranker = SentenceReranker()
    print(f"Created reranker: {reranker.name()}")
    
    # Test single pair scoring
    query = "fresh apple"
    candidate = "Apple, raw"
    score = reranker.score_pair(query, candidate)
    print(f"Query: '{query}', Candidate: '{candidate}', Score: {score:.4f}")
    
    # Test multiple pairs scoring
    pairs = [
        ["fresh apple", "Apple, raw"],
        ["fresh apple", "Apple juice"],
        ["organic banana", "Banana, raw"]
    ]
    scores = reranker.score_pairs(pairs)
    for i, (pair, score) in enumerate(zip(pairs, scores)):
        print(f"Pair {i+1}: {pair[0]} - {pair[1]}, Score: {score:.4f}")
    
    # Create some test candidates
    candidates = [
        {"usda_code": "Apple, raw", "similarity": 0.85},
        {"usda_code": "Apple juice", "similarity": 0.76},
        {"usda_code": "Apple, dried", "similarity": 0.72},
        {"usda_code": "Applesauce", "similarity": 0.65},
        {"usda_code": "Pear, raw", "similarity": 0.60}
    ]
    
    # Test reranking
    reranked = reranker.rerank("fresh apple", candidates)
    print("\nReranked results:")
    for i, result in enumerate(reranked[:3]):
        print(f"{i+1}. {result['usda_code']} - Score: {result['similarity']:.4f}")
    
    # Verify that reranker check works
    assert check_is_reranker(reranker) == True
    assert check_is_reranker("not a reranker") == False
    
    print("SentenceReranker tests passed!")
    return reranker

def test_batch_processing():
    """Test the batch processing utility."""
    print("\n=== Testing Batch Processing ===")
    
    # Create test data
    pairs = [
        ["query1", "candidate1"],
        ["query1", "candidate2"],
        ["query2", "candidate1"],
        ["query2", "candidate2"],
        ["query3", "candidate1"]
    ]
    
    # Create a simple scoring function for testing
    def mock_scorer(batch):
        return [0.9 if p[0] == p[1] else 0.5 for p in batch]
    
    # Test batch processing
    batch_size = 2
    scores = process_in_batches(pairs, mock_scorer, batch_size, show_progress=True)
    
    print(f"Processed {len(pairs)} pairs in batches of {batch_size}")
    print(f"Scores: {[f'{s:.2f}' for s in scores]}")
    
    # Verify correct number of scores
    assert len(scores) == len(pairs)
    print("Batch processing tests passed!")

def test_metrics():
    """Test the evaluation metrics."""
    print("\n=== Testing Evaluation Metrics ===")
    
    # Test find_rank
    candidates = [
        {"usda_code": "code1", "similarity": 0.9},
        {"usda_code": "code2", "similarity": 0.8},
        {"usda_code": "code3", "similarity": 0.7},
        {"usda_code": "code4", "similarity": 0.6},
        {"usda_code": "code5", "similarity": 0.5}
    ]
    
    rank = find_rank("code3", candidates)
    print(f"Rank of 'code3': {rank}")
    assert rank == 3
    
    # Test non-existent code
    rank = find_rank("code99", candidates)
    print(f"Rank of non-existent code: {rank}")
    assert rank is None
    
    # Test reranking metrics
    original_ranks = [3, 5, 2, 10, 1]
    reranked_ranks = [1, 2, 4, 8, 3]
    
    metrics = calculate_reranking_metrics(original_ranks, reranked_ranks)
    print("Reranking metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {value[k]:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # Test precision@k
    ranks = [1, 3, 5, 10, 15]
    p1 = calculate_precision_at_k(ranks, k=1)
    p3 = calculate_precision_at_k(ranks, k=3)
    print(f"Precision@1: {p1:.2f}, Precision@3: {p3:.2f}")
    
    print("Metrics tests passed!")

def test_pipeline():
    """Test the complete reranking pipeline."""
    print("\n=== Testing Complete Pipeline ===")
    
    # Test create_reranker
    reranker = create_reranker()
    print(f"Created reranker: {reranker.name()}")
    assert check_is_reranker(reranker)
    
    # Test candidates
    candidates = [
        {"usda_code": "Apple, raw", "similarity": 0.85},
        {"usda_code": "Apple juice", "similarity": 0.76},
        {"usda_code": "Apple, dried", "similarity": 0.72},
        {"usda_code": "Pear, raw", "similarity": 0.60},
        {"usda_code": "Orange, raw", "similarity": 0.55}
    ]
    
    # Test rerank_candidates
    query = "fresh apple"
    reranked = rerank_candidates(query, candidates, reranker=reranker)
    print(f"\nQuery: '{query}'")
    print("Top 3 reranked results:")
    for i, result in enumerate(reranked[:3]):
        print(f"{i+1}. {result['usda_code']} - Score: {result['similarity']:.4f}")
    
    # Test evaluate_reranking
    test_queries = [
        "fresh apple",
        "orange juice",
        "ripe banana"
    ]
    
    # Create candidate sets
    candidate_sets = [
        candidates,  # For "fresh apple"
        [
            {"usda_code": "Orange juice", "similarity": 0.88},
            {"usda_code": "Orange, raw", "similarity": 0.75},
            {"usda_code": "Tangerine juice", "similarity": 0.70}
        ],  # For "orange juice"
        [
            {"usda_code": "Banana, raw", "similarity": 0.82},
            {"usda_code": "Banana, dried", "similarity": 0.76},
            {"usda_code": "Plantain", "similarity": 0.65}
        ]   # For "ripe banana"
    ]
    
    correct_values = ["Apple, raw", "Orange juice", "Banana, raw"]
    
    print("\nEvaluating reranking performance...")
    metrics = evaluate_reranking(
        test_queries=test_queries,
        candidate_sets=candidate_sets,
        correct_values=correct_values,
        reranker=reranker
    )
    
    print("Evaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        else:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("Pipeline tests passed!")

def run_all_tests():
    """Run all test functions."""
    print("===== Cross Encoder Module Tests =====")
    
    try:
        reranker = test_sentence_reranker()
        test_batch_processing()
        test_metrics()
        test_pipeline()
        
        print("\n===== All Tests Passed! =====")
        print("The Cross_Encoder module is functioning correctly.")
    except Exception as e:
        print(f"\n===== Test Failed! =====")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    run_all_tests()
