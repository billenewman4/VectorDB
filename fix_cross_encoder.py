#!/usr/bin/env python3
"""
Script to analyze and fix cross-encoder ranking issues for specific problematic products.
Tests different approaches to improve cross-encoder performance on specialized meat terminology.
"""
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.VectorDB.CrossEncoder import CrossEncoder
try:
    from src.VectorDB.localEmbedder import LocalEmbedder
except ImportError:
    print("Warning: LocalEmbedder import failed - local embedding may not be available")

def analyze_problematic_products():
    """Analyze the specific products that have issues with cross-encoder ranking"""
    print("\n" + "="*80)
    print("Cross-Encoder Problem Analysis")
    print("="*80)
    
    # Load the previous analysis results
    try:
        df = pd.read_csv('cross_encoder_analysis_results.csv')
        print(f"Loaded {len(df)} products from analysis results")
    except Exception as e:
        print(f"Error loading analysis results: {e}")
        return
    
    # Identify problematic products (those with worse ranking or not found)
    problem_df = df[(df['status'] == 'Worse') | 
                    ((df['embedding_rank'] == 1) & df['cross_encoder_rank'].isna())]
    
    print(f"\nFound {len(problem_df)} problematic products")
    for idx, row in problem_df.iterrows():
        print(f"\nProduct: {row['product_code']}")
        print(f"Description: {row['description']}")
        print(f"Correct USDA: {row['correct_usda']}")
        print(f"Embedding Rank: {row['embedding_rank']}")
        print(f"Cross-Encoder Rank: {row['cross_encoder_rank']}")
        print(f"Top Match: {row['top_match']}")
    
    # Test different approaches for the problematic products
    print("\n" + "="*80)
    print("Testing Approaches to Fix Problematic Cases")
    print("="*80)
    
    # Initialize models
    try:
        cross_encoder = CrossEncoder()
        
        # Try an alternative cross-encoder model
        alt_cross_encoder = CrossEncoder(model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2')
        
        # We'll also test with embedding-based comparisons as fallback
        embedder = LocalEmbedder()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return
    
    # Define the problematic product descriptions to test
    test_cases = [
        {"description": "beef boneless rib lifter meat choice", "correct": "Cap & Wedge"},
        {"description": "bf rib meat lifter/blade ch oma", "correct": "Cap & Wedge"},
        {"description": "bf skirt outer ch ibp d3157ah", "correct": "121 C 4 Outside skirt"}
    ]
    
    # Define candidate USDA codes to test against
    candidates = [
        "Cap & Wedge",
        "123 A 3 short rib choice",
        "121 C 4 Outside skirt",
        "121 D 4 Inside skirt",
        "130 4 chuck short rib",
        "112a HVY Lipon Ribeye",
        "109E 1 Rib ribeye lip-on bn-in_Choice",
        "116g Chuck Flap"
    ]
    
    # Test each approach
    for test_case in test_cases:
        query = test_case["description"]
        correct = test_case["correct"]
        
        print(f"\nTesting product: {query}")
        print(f"Correct USDA: {correct}")
        
        # Create candidate dictionaries for the cross-encoder
        candidate_dicts = [{"usda_code": code, "similarity": 0.5} for code in candidates]
        
        # Approach 1: Standard cross-encoder
        print("\nApproach 1: Standard cross-encoder")
        reranked = cross_encoder.rerank(query, candidate_dicts)
        for i, match in enumerate(reranked[:3]):
            print(f"  {i+1}. {match['usda_code']} - Score: {match['cross_encoder_score']:.4f}")
        
        # Find correct position
        correct_pos = next((i+1 for i, match in enumerate(reranked) 
                          if match['usda_code'] == correct), None)
        print(f"  Correct USDA code position: {correct_pos}")
        
        # Approach 2: Alternative cross-encoder model
        print("\nApproach 2: Alternative cross-encoder")
        reranked_alt = alt_cross_encoder.rerank(query, candidate_dicts)
        for i, match in enumerate(reranked_alt[:3]):
            print(f"  {i+1}. {match['usda_code']} - Score: {match['cross_encoder_score']:.4f}")
        
        correct_pos_alt = next((i+1 for i, match in enumerate(reranked_alt) 
                              if match['usda_code'] == correct), None)
        print(f"  Correct USDA code position: {correct_pos_alt}")
        
        # Approach 3: Enhanced input formatting
        print("\nApproach 3: Enhanced input formatting")
        # Add domain-specific context to the query and candidates
        enhanced_query = f"Meat product description: {query}"
        enhanced_candidate_dicts = [
            {"usda_code": f"USDA meat code: {code}", "similarity": 0.5} 
            for code in candidates
        ]
        
        reranked_enhanced = cross_encoder.rerank(enhanced_query, enhanced_candidate_dicts)
        for i, match in enumerate(reranked_enhanced[:3]):
            code = match['usda_code'].replace("USDA meat code: ", "")
            print(f"  {i+1}. {code} - Score: {match['cross_encoder_score']:.4f}")
        
        # Find correct position
        correct_pos_enhanced = next((i+1 for i, match in enumerate(reranked_enhanced) 
                                  if match['usda_code'] == f"USDA meat code: {correct}"), None)
        print(f"  Correct USDA code position: {correct_pos_enhanced}")
        
        # Approach 4: Domain-specific synonyms and patterns
        print("\nApproach 4: Domain-specific knowledge integration")
        
        # Define domain-specific patterns and synonyms
        domain_patterns = {
            "lifter": ["cap & wedge", "cap and wedge", "rib lifter"],
            "skirt outer": ["outside skirt", "121 c", "exterior skirt"],
            "skirt inner": ["inside skirt", "121 d", "interior skirt"]
        }
        
        # Preprocessing function with domain knowledge
        def domain_preprocess(text):
            text = text.lower()
            # Apply domain-specific mappings
            if "lifter" in text:
                print("  Domain knowledge: Detected 'lifter' term in description")
                return "cap and wedge meat"
            elif "skirt outer" in text or "outside skirt" in text:
                print("  Domain knowledge: Detected 'outside skirt' pattern")
                return "outside skirt meat 121 c"
            elif "skirt inner" in text or "inside skirt" in text:
                print("  Domain knowledge: Detected 'inside skirt' pattern")
                return "inside skirt meat 121 d"
            return text
        
        # Apply domain-specific preprocessing
        domain_query = domain_preprocess(query)
        print(f"  Preprocessed query: {domain_query}")
        
        # Use the preprocessed query
        reranked_domain = cross_encoder.rerank(domain_query, candidate_dicts)
        for i, match in enumerate(reranked_domain[:3]):
            print(f"  {i+1}. {match['usda_code']} - Score: {match['cross_encoder_score']:.4f}")
        
        correct_pos_domain = next((i+1 for i, match in enumerate(reranked_domain) 
                                if match['usda_code'] == correct), None)
        print(f"  Correct USDA code position: {correct_pos_domain}")
        
        # Approach 5: Hybrid - use embedding-based similarity
        print("\nApproach 5: Hybrid with embedding-based fallback")
        print("  Generating embeddings...")
        
        # Get embeddings
        query_embedding = embedder([query])[0]
        candidate_embeddings = embedder(candidates)
        
        # Calculate cosine similarities
        similarities = []
        for i, candidate in enumerate(candidates):
            candidate_embedding = candidate_embeddings[i]
            query_embedding_2d = np.array(query_embedding).reshape(1, -1)
            candidate_embedding_2d = np.array(candidate_embedding).reshape(1, -1)
            
            similarity = float(cosine_similarity(query_embedding_2d, candidate_embedding_2d)[0][0])
            similarities.append(similarity)
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Display top embedding matches
        for i in range(min(3, len(sorted_indices))):
            idx = sorted_indices[i]
            print(f"  {i+1}. {candidates[idx]} - Similarity: {similarities[idx]:.4f}")
        
        # Find correct position in embedding results
        correct_idx = candidates.index(correct) if correct in candidates else -1
        if correct_idx >= 0:
            correct_pos_embed = sorted_indices.tolist().index(correct_idx) + 1
            print(f"  Correct USDA code position: {correct_pos_embed}")
        else:
            print("  Correct USDA code not in candidates")
        
        # Approach 6: Score normalization and hybrid scoring
        print("\nApproach 6: Score normalization and hybrid scoring")
        
        # Create a dictionary mapping from USDA code to scores from different methods
        combined_scores = {}
        
        # Normalize scores across different methods
        def normalize_scores(scores):
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [0.5 for _ in scores]  # Avoid division by zero
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        # Collect scores from multiple methods
        for i, code in enumerate(candidates):
            # Get cross-encoder score
            cross_encoder_score = next((match['cross_encoder_score'] for match in reranked 
                                      if match['usda_code'] == code), -999)
            
            # Get embedding score
            embedding_score = similarities[i]
            
            # Collect scores
            combined_scores[code] = {
                'cross_encoder': cross_encoder_score,
                'embedding': embedding_score
            }
        
        # Extract and normalize scores
        cross_encoder_scores = [score['cross_encoder'] for code, score in combined_scores.items()]
        embedding_scores = [score['embedding'] for code, score in combined_scores.items()]
        
        # Normalize scores
        norm_cross_encoder = normalize_scores(cross_encoder_scores)
        norm_embedding = normalize_scores(embedding_scores)
        
        # Apply hybrid scoring with different weights based on observed patterns
        hybrid_scores = []
        for i, code in enumerate(candidates):
            # If cross-encoder score is extremely negative, use more embedding weight
            if combined_scores[code]['cross_encoder'] < -8:
                weight_cross = 0.2
                weight_embedding = 0.8
                print(f"  Using higher embedding weight (0.8) for {code} due to very negative cross-encoder score")
            else:
                weight_cross = 0.6
                weight_embedding = 0.4
            
            # Calculate hybrid score
            hybrid_score = weight_cross * norm_cross_encoder[i] + weight_embedding * norm_embedding[i]
            hybrid_scores.append(hybrid_score)
        
        # Sort by hybrid score
        hybrid_sorted_indices = np.argsort(hybrid_scores)[::-1]
        
        # Display top hybrid matches
        for i in range(min(3, len(hybrid_sorted_indices))):
            idx = hybrid_sorted_indices[i]
            code = candidates[idx]
            print(f"  {i+1}. {code} - Hybrid Score: {hybrid_scores[idx]:.4f} "
                  f"(Cross: {norm_cross_encoder[idx]:.2f}, Embed: {norm_embedding[idx]:.2f})")
        
        # Find correct position in hybrid results
        if correct in candidates:
            correct_idx = candidates.index(correct)
            correct_pos_hybrid = hybrid_sorted_indices.tolist().index(correct_idx) + 1
            print(f"  Correct USDA code position: {correct_pos_hybrid}")
        else:
            print("  Correct USDA code not in candidates")
        
        print("-" * 60)
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("Summary and Recommendations")
    print("="*80)
    print("""
Based on the tests, here are the recommended approaches to fix the cross-encoder issues:

1. Implement a hybrid scoring system that:
   - Detects extremely negative cross-encoder scores (< -8)
   - Falls back to embedding-based similarity in those cases
   - Uses a weighted combination otherwise

2. Add domain-specific preprocessing for known problematic terms:
   - Map "lifter meat" to "Cap & Wedge"
   - Distinguish between "inside skirt" and "outside skirt"

3. Enhance input formatting:
   - Add context prefixes to both queries and USDA codes

These changes should help address the specific cases where the cross-encoder performs poorly
while maintaining its good performance on the majority of products.
""")

if __name__ == "__main__":
    analyze_problematic_products()
