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
    Uses a weighted combination of embedding similarity and cross-encoder scores.
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                cross_encoder_weight: float = 0.7, embedding_weight: float = 0.3):
        """
        Initialize the CrossEncoder with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained cross-encoder model to use.
                        Default is 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            cross_encoder_weight: Weight to apply to cross-encoder scores.
                                Default is 0.7
            embedding_weight: Weight to apply to embedding similarity scores.
                            Default is 0.3
        """
        self.model_name = model_name
        self.cross_encoder_weight = cross_encoder_weight
        self.embedding_weight = embedding_weight
        print(f"Initializing CrossEncoder with model: {model_name}")
        print(f"Weights: Cross-encoder={cross_encoder_weight:.2f}, Embedding={embedding_weight:.2f}")
        self.model = SentenceCrossEncoder(model_name)
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
               batch_size: int = 32, debug: bool = False) -> List[Dict[str, Any]]:
        """
        Re-rank candidate USDA codes using a weighted combination of
        cross-encoder scores and embedding similarity.
        
        Args:
            query: The product description to match
            candidates: List of candidate matches (each with 'usda_code' and 'similarity' fields)
            batch_size: Batch size for cross-encoder predictions
            debug: Whether to print debug information
            
        Returns:
            Re-ranked list of candidates with updated similarity scores
        """
        if debug:
            print(f"Re-ranking {len(candidates)} candidates for query: '{query}'")
        
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
        
        reranked_candidates = []
        for i, candidate in enumerate(candidates):
            # Create a copy of the candidate to avoid modifying the original
            reranked = dict(candidate)
            
            # Get original embedding similarity
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
