"""
Sentence Transformer Cross-Encoder implementation.
Provides a reranker that uses Sentence Transformers cross-encoder models.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers.cross_encoder import CrossEncoder as SentenceCrossEncoder

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
cross_encoder_dir = os.path.dirname(models_dir)
sys.path.append(os.path.dirname(cross_encoder_dir))

# Import base reranker interface
from .base_reranker import BaseReranker

class SentenceReranker(BaseReranker):
    """
    Cross-encoder implementation using the Sentence Transformers library.
    
    This reranker uses a weighted combination of embedding similarity and
    cross-encoder scores for more accurate matching.
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                cross_encoder_weight: float = 0.7, embedding_weight: float = 0.3):
        """
        Initialize the SentenceReranker with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained cross-encoder model to use.
                        Default is 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            cross_encoder_weight: Weight to apply to cross-encoder scores.
                                Default is 0.7
            embedding_weight: Weight to apply to embedding similarity scores.
                            Default is 0.3
                            
        Raises:
            ValueError: If the weights don't sum to 1.0
            RuntimeError: If model initialization fails
        """
        if not 0 <= cross_encoder_weight <= 1 or not 0 <= embedding_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
        
        if abs(cross_encoder_weight + embedding_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {cross_encoder_weight + embedding_weight}")
            
        self.model_name = model_name
        self.cross_encoder_weight = cross_encoder_weight
        self.embedding_weight = embedding_weight
        
        print(f"Initializing SentenceReranker with model: {model_name}")
        print(f"Weights: Cross-encoder={cross_encoder_weight:.2f}, Embedding={embedding_weight:.2f}")
        
        try:
            self.model = SentenceCrossEncoder(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize cross-encoder model: {str(e)}")
    
    def name(self) -> str:
        """
        Return a unique identifier for this reranker.
        
        Returns:
            str: A unique identifier for the reranker
        """
        return f"sentence_reranker_{self.model_name.replace('/', '_')}"
    
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
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if not candidate or not isinstance(candidate, str):
            raise ValueError("Candidate must be a non-empty string")
        
        try:
            return float(self.model.predict([[query, candidate]])[0])
        except Exception as e:
            raise RuntimeError(f"Error scoring query-candidate pair: {str(e)}")
    
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
        if not pairs:
            raise ValueError("Pairs list cannot be empty")
        
        # Validate all pairs
        for i, pair in enumerate(pairs):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Pair at index {i} must be a list with exactly 2 elements")
            
            if not pair[0] or not isinstance(pair[0], str):
                raise ValueError(f"Query in pair at index {i} must be a non-empty string")
            
            if not pair[1] or not isinstance(pair[1], str):
                raise ValueError(f"Candidate in pair at index {i} must be a non-empty string")
        
        try:
            return self.model.predict(pairs).tolist()
        except Exception as e:
            raise RuntimeError(f"Error scoring pairs: {str(e)}")
    
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
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        if debug:
            print(f"Re-ranking {len(candidates)} candidates for query: '{query}'")
        
        # Prepare text pairs for the cross-encoder
        text_pairs = []
        for candidate in candidates:
            if 'usda_code' not in candidate:
                raise ValueError("Each candidate must have a 'usda_code' field")
            
            text_pairs.append([query, candidate['usda_code']])
        
        # Generate cross-encoder scores in batches
        scores = []
        for i in range(0, len(text_pairs), batch_size):
            batch = text_pairs[i:i+batch_size]
            try:
                batch_scores = self.model.predict(batch)
                scores.extend(batch_scores)
            except Exception as e:
                raise RuntimeError(f"Error predicting batch scores: {str(e)}")
        
        reranked_candidates = []
        for i, candidate in enumerate(candidates):
            # Create a copy of the candidate to avoid modifying the original
            reranked = dict(candidate)
            
            # Get original embedding similarity
            if 'similarity' not in candidate:
                raise ValueError("Each candidate must have a 'similarity' field")
                
            embedding_score = candidate.get('similarity', 0.0)
            
            # Get cross-encoder score
            cross_encoder_score = float(scores[i])
            
            # Store both scores for reference
            reranked['embedding_score'] = embedding_score
            reranked['cross_encoder_score'] = cross_encoder_score
            
            # Calculate weighted score - simple weighted average
            weighted_score = (
                self.cross_encoder_weight * cross_encoder_score + 
                self.embedding_weight * embedding_score
            )
            
            # Update the similarity score
            reranked['similarity'] = weighted_score
            
            # Debug output
            if debug:
                print(f"  {candidate['usda_code']} - Emb: {embedding_score:.4f}, CE: {cross_encoder_score:.4f}, "
                      f"Final: {weighted_score:.4f}")
            
            reranked_candidates.append(reranked)
        
        # Sort by weighted similarity scores
        reranked_candidates = sorted(reranked_candidates, key=lambda x: x['similarity'], reverse=True)
        
        return reranked_candidates
    
    def analyze_matches(self, query: str, candidates: List[Dict[str, Any]], 
                       correct_code: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyze and compare embedding and cross-encoder performance for a query.
        
        Args:
            query: The product description to match
            candidates: List of candidate matches from embeddings
            correct_code: The correct USDA code (if known)
            
        Returns:
            Tuple of (reranked candidates, performance metrics)
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Get original rankings if we have a correct code
        original_rank = None
        if correct_code:
            for i, candidate in enumerate(candidates):
                if candidate['usda_code'] == correct_code:
                    original_rank = i + 1
                    break
        
        # Rerank with cross-encoder
        reranked = self.rerank(query, candidates)
        
        # Get new ranking if we have a correct code
        new_rank = None
        if correct_code:
            for i, candidate in enumerate(reranked):
                if candidate['usda_code'] == correct_code:
                    new_rank = i + 1
                    break
        
        # Compute performance metrics
        metrics = {
            'original_rank': original_rank,
            'new_rank': new_rank,
            'rank_improvement': (original_rank - new_rank) if (original_rank and new_rank) else None,
            'original_in_top_k': {
                1: original_rank == 1 if original_rank else False,
                3: original_rank <= 3 if original_rank else False,
                5: original_rank <= 5 if original_rank else False
            },
            'reranked_in_top_k': {
                1: new_rank == 1 if new_rank else False,
                3: new_rank <= 3 if new_rank else False,
                5: new_rank <= 5 if new_rank else False
            }
        }
        
        return reranked, metrics

# Test code
if __name__ == "__main__":
    print("Testing SentenceReranker...")
    
    # Create a reranker
    reranker = SentenceReranker()
    
    # Test a simple pair
    query = "fresh apple"
    candidate = "Apple, raw"
    score = reranker.score_pair(query, candidate)
    print(f"Score for '{query}' - '{candidate}': {score:.4f}")
    
    # Test with some candidates
    candidates = [
        {"usda_code": "Apple, raw", "similarity": 0.85},
        {"usda_code": "Apple juice", "similarity": 0.76},
        {"usda_code": "Pear, raw", "similarity": 0.60}
    ]
    
    reranked = reranker.rerank(query, candidates, debug=True)
    
    print("\nReranked results:")
    for i, result in enumerate(reranked):
        print(f"{i+1}. {result['usda_code']} - Score: {result['similarity']:.4f}")
