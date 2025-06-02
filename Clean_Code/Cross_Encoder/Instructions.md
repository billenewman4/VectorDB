# Cross Encoder Module Implementation Plan

This document outlines the functions and classes that need to be implemented in the Clean_Code/Cross_Encoder module to handle all cross-encoder-related functionality in the VectorDB project. The cross encoder is used for re-ranking product matches and improving similarity calculations.

## Directory Structure

```
Cross_Encoder/
├── Models/                    # Cross-encoder model implementations
│   ├── base_reranker.py       # Base class and interfaces
│   └── sentence_reranker.py    # Sentence transformer cross-encoder implementation
├── Utilities/                 # Helper functions
│   ├── batch_processor.py     # Batch processing for cross-encoder scoring
│   └── metrics.py             # Evaluation metrics for reranking
├── pipeline.py               # Main cross-encoder pipeline functions
├── README.md                 # Documentation
└── Instructions.md           # This file
```

## Functions and Classes to Implement

### 1. Base Reranker Interface (Models/base_reranker.py)

```python
class BaseReranker:
    """Base interface for all cross-encoder rerankers"""
    
    def name(self) -> str:
        """Return a unique identifier for this reranker"""
        pass
    
    def score_pair(self, query: str, candidate: str) -> float:
        """Score a single query-candidate pair"""
        pass
    
    def score_pairs(self, pairs: List[List[str]]) -> List[float]:
        """Score a list of query-candidate pairs"""
        pass
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
              batch_size: int = 32, debug: bool = False) -> List[Dict[str, Any]]:
        """Re-rank candidates using cross-encoder scores"""
        pass
```

### 2. Sentence Transformer Reranker (Models/sentence_reranker.py)

Implementation of the `CrossEncoder` class from `src/VectorDB/CrossEncoder.py` with these key methods:

- `__init__(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', cross_encoder_weight=0.7, embedding_weight=0.3)`
- `name()` - Returns identifier for this reranker
- `score_pair(query, candidate)` - Scores a single query-candidate pair
- `score_pairs(pairs)` - Scores multiple query-candidate pairs
- `rerank(query, candidates, batch_size=32, debug=False)` - Re-ranks candidates using cross-encoder scores
- `analyze_matches(query, candidates, correct_code=None)` - Analyzes reranker performance

### 3. Batch Processing (Utilities/batch_processor.py)

```python
def process_in_batches(pairs: List[List[str]], scorer_func: Callable, batch_size: int = 32, 
                       show_progress: bool = True) -> List[float]:
    """Process a list of text pairs in batches for efficient cross-encoder scoring."""
```

### 4. Evaluation Metrics (Utilities/metrics.py)

```python
def calculate_reranking_metrics(original_ranks: List[int], reranked_ranks: List[int]) -> Dict[str, Any]:
    """Calculate metrics to evaluate reranking performance"""

def find_rank(target_value: str, candidates: List[Dict[str, Any]], key: str = 'usda_code') -> Optional[int]:
    """Find the rank of a target value in a list of candidates"""

def calculate_precision_at_k(ranks: List[int], k: int = 1) -> float:
    """Calculate precision@k metric for evaluation"""
```

### 5. Main Cross-Encoder Pipeline (pipeline.py)

Implementation of high-level cross-encoder functions:

#### 5.1 create_reranker

```python
def create_reranker(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                   cross_encoder_weight: float = 0.7,
                   embedding_weight: float = 0.3) -> BaseReranker:
    """Create and return an appropriate cross-encoder reranker based on the specified model."""
```

#### 5.2 rerank_candidates

```python
def rerank_candidates(query: str, 
                     candidates: List[Dict[str, Any]],
                     reranker: Union[BaseReranker, str] = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                     cross_encoder_weight: float = 0.7,
                     embedding_weight: float = 0.3,
                     batch_size: int = 32,
                     debug: bool = False) -> List[Dict[str, Any]]:
    """Re-rank candidates using a cross-encoder."""
```

#### 5.3 evaluate_reranking

```python
def evaluate_reranking(test_queries: List[str], 
                      candidate_sets: List[List[Dict[str, Any]]],
                      correct_values: List[str],
                      reranker: BaseReranker,
                      batch_size: int = 32) -> Dict[str, Any]:
    """Evaluate reranking performance on a test set."""
```

## Implementation Notes

- **Cross-Encoder Model**: The default model is 'cross-encoder/ms-marco-MiniLM-L-6-v2', but the implementation should support other cross-encoder models.

- **Weighting Strategy**: The default weighting is 70% cross-encoder score and 30% embedding similarity, but these should be configurable.

- **Error Handling**: Include explicit error handling that raises appropriate exceptions rather than falling back to default behaviors.

- **Integration with Vector Embeddings**: The cross-encoder module should work seamlessly with the Vector_Embedding module for an integrated reranking pipeline.

- **Performance Metrics**: Include metrics to evaluate reranking performance (precision@k, mean reciprocal rank, etc.)

- **Candidate Format**: Candidates should be provided as a list of dictionaries, each with at least 'usda_code' and 'similarity' fields.

- **Batch Processing**: Efficient batch processing for cross-encoder scoring to handle large candidate sets.

## Improvement Opportunities

- **Adaptive Weighting**: Consider implementing adaptive weighting based on candidate set properties rather than fixed weights.

- **Alternative Models**: Support for newer and more specialized cross-encoder models beyond the default.

- **Caching**: Consider adding a caching layer for frequent query-candidate pairs to improve performance.

- **Parallelization**: For large candidate sets, consider adding parallel processing options.

## Testing

### Comprehensive Test Script (test_cross_encoder.py)

Implement a comprehensive test script to validate all components of the Cross_Encoder module:

```python
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
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

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
        print(f"  {key}: {value}")
    
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
                print(f"    {k}: {v}")
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
```

This test script should be placed in the main Cross_Encoder directory and will test all components of the module. It includes tests for:

1. **SentenceReranker**: Tests basic functionality, single and batch scoring, and reranking
2. **Batch Processing**: Tests the batch processing utility with different batch sizes
3. **Metrics**: Tests rank calculation, reranking metrics, and precision calculations
4. **Pipeline**: Tests the complete pipeline from creating a reranker to evaluating performance

To run the tests, execute the script after implementing all required components.