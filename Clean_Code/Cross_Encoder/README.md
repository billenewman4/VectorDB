# Cross Encoder Module

A robust module for re-ranking candidates with cross-encoder models, providing more accurate similarity assessments by considering pairs of texts together rather than independently.

## Overview

This module implements cross-encoder functionality for the VectorDB project, focused on improving search and matching results through re-ranking. Cross-encoders excel at determining the relevance between two pieces of text by processing them together through a transformer model, which leads to more contextually aware similarity scores compared to traditional embedding approaches.

## Key Features

- **Weighted Re-ranking**: Combines cross-encoder scores with existing embedding similarities using configurable weights
- **Batch Processing**: Efficiently scores large sets of query-candidate pairs in batches
- **Flexible Model Support**: Works with various cross-encoder models from the Sentence Transformers library
- **Performance Metrics**: Includes utilities for evaluating re-ranking performance
- **Seamless Integration**: Designed to work with the Vector_Embedding module

## Usage Examples

### Basic Re-ranking

```python
from Clean_Code.Cross_Encoder.pipeline import rerank_candidates

# Candidates from embedding search
candidates = [
    {"usda_code": "Apple, raw", "similarity": 0.85},
    {"usda_code": "Apple, dried", "similarity": 0.76},
    {"usda_code": "Apple juice", "similarity": 0.72},
    # More candidates...
]

# Re-rank using cross-encoder
reranked = rerank_candidates(
    query="Fresh red apple",
    candidates=candidates,
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Default model
    cross_encoder_weight=0.7,
    embedding_weight=0.3,
    batch_size=32
)

# Now candidates are re-ranked with improved similarity scores
for i, match in enumerate(reranked[:3]):
    print(f"{i+1}. {match['usda_code']} - Score: {match['similarity']:.4f}")
```

### Creating a Reusable Reranker

```python
from Clean_Code.Cross_Encoder.pipeline import create_reranker

# Create a reranker instance
reranker = create_reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    cross_encoder_weight=0.7,
    embedding_weight=0.3
)

# Use the same reranker for multiple queries
results1 = reranker.rerank("Fresh apples", candidates1)
results2 = reranker.rerank("Organic bananas", candidates2)
```

### Evaluating Reranking Performance

```python
from Clean_Code.Cross_Encoder.pipeline import evaluate_reranking
from Clean_Code.Cross_Encoder.pipeline import create_reranker

# Create a reranker
reranker = create_reranker()

# Test data
test_queries = ["Fresh apple", "Organic banana", "Red tomato"]
candidate_sets = [candidates1, candidates2, candidates3]  # Lists of candidate dictionaries
correct_values = ["Apple, raw", "Banana, raw", "Tomato, red, raw"]

# Evaluate performance
metrics = evaluate_reranking(
    test_queries=test_queries,
    candidate_sets=candidate_sets,
    correct_values=correct_values,
    reranker=reranker
)

print(f"Precision@1: {metrics['precision@1']:.4f}")
print(f"MRR: {metrics['mrr']:.4f}")
print(f"Average Rank Improvement: {metrics['avg_rank_improvement']:.2f}")
```

## Module Components

### Models

- **BaseReranker**: Abstract base class defining the interface for all rerankers
- **SentenceReranker**: Implementation using Sentence Transformers cross-encoder models

### Utilities

- **batch_processor**: Functions for efficiently processing data in batches
- **metrics**: Evaluation metrics for assessing reranking performance

### Pipeline

- **create_reranker**: Factory function for creating reranker instances
- **rerank_candidates**: High-level function for re-ranking candidates
- **evaluate_reranking**: Functions for evaluating reranker performance

## Design Decisions

1. **Explicit Error Handling**: The module raises explicit errors rather than using fallbacks to ensure issues are clearly identified and addressed.

2. **Weighted Scoring Approach**: The default approach uses a weighted combination of cross-encoder scores (70%) and original embedding similarity scores (30%), which generally provides better results than using cross-encoder scores alone.

3. **Model Selection**: The default model 'cross-encoder/ms-marco-MiniLM-L-6-v2' provides a good balance between performance and efficiency, but other models can be specified.

4. **Batch Processing**: All operations that might involve multiple items use batch processing with progress tracking to optimize performance and provide feedback.

5. **Modular Architecture**: The module follows a clean separation of concerns with Models, Utilities, and high-level pipeline functions.

## Key Notes

- Cross-encoders process pairs of texts together and are more accurate but slower than bi-encoders (regular embeddings)
- The weighted approach leverages both the efficiency of pre-computed embeddings and the accuracy of cross-encoders
- Batch sizes can be adjusted based on available memory and performance requirements
- The module works best with the Vector_Embedding module as the first-stage retrieval system

## Dependencies

- sentence-transformers
- numpy
- pandas
- tqdm (for progress tracking)

## Improvement Areas

- Potential for incorporating adaptive weighting based on confidence scores
- Support for specialized domain-specific cross-encoder models
- Caching mechanisms for frequently reused query-candidate pairs
- Parallelization options for very large candidate sets