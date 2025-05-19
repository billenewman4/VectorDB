"""
CrossEncoder for re-ranking USDA code matches.
This provides more accurate similarity scores by analyzing product descriptions and USDA codes together.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers.cross_encoder import CrossEncoder as SentenceCrossEncoder
import os
import time
from tqdm import tqdm

class CrossEncoder:
    """
    CrossEncoder for more accurate matching between product descriptions and USDA codes.
    Uses a cross-encoder model to compare pairs of texts directly, rather than separate embeddings.
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the CrossEncoder with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained cross-encoder model to use.
                        Default is 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        """
        self.model_name = model_name
        print(f"Initializing CrossEncoder with model: {model_name}")
        self.model = SentenceCrossEncoder(model_name)
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
               batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Re-rank candidate USDA codes based on cross-encoder scores.
        
        Args:
            query: The product description to match
            candidates: List of candidate matches (each with 'usda_code' and other fields)
            batch_size: Batch size for cross-encoder predictions
            
        Returns:
            Re-ranked list of candidates with updated similarity scores
        """
        # Prepare text pairs for the cross-encoder
        text_pairs = []
        for candidate in candidates:
            text_pairs.append([query, candidate['usda_code']])
        
        # Generate cross-encoder scores in batches
        scores = []
        for i in range(0, len(text_pairs), batch_size):
            batch = text_pairs[i:i+batch_size]
            batch_scores = self.model.predict(batch)
            scores.extend(batch_scores)
        
        # Create copies of candidates with updated scores
        reranked_candidates = []
        for i, candidate in enumerate(candidates):
            # Create a copy of the candidate to avoid modifying the original
            reranked = dict(candidate)
            
            # Store both the original and cross-encoder scores
            reranked['embedding_similarity'] = candidate['similarity']
            reranked['cross_encoder_score'] = float(scores[i])
            
            # Update the main similarity score to use the cross-encoder score
            reranked['similarity'] = float(scores[i])
            
            reranked_candidates.append(reranked)
        
        # Sort by new similarity scores
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
