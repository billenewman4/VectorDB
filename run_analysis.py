#!/usr/bin/env python3
"""Interactive analysis script for VectorDB product matching.
This script allows the user to select different options for analysis:
1. Embedding type (OpenAI vs other local models)
2. Data scope (limited products, only USDA coded products, or all products)
3. Analysis direction (products for USDA codes OR USDA codes for products)
4. Optional LLM verification of import time

This script has been refactored for better readability and maintainability.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import requests
import time
import re
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from openai import OpenAI  # Updated OpenAI import
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.VectorDB.helper import build_usda_lookup, preprocess_text_for_matching, normalize_mapping_id
from src.VectorDB.OpenAIEmbedder import OpenAIEmbedder
from src.VectorDB.CrossEncoder import CrossEncoder
try:
    from src.VectorDB.localEmbedder import LocalEmbedder
except ImportError:
    print("Warning: LocalEmbedder import failed - local embedding may not be available")

try:
    from src.llm_selector import GPT4Selector
except ImportError:
    print("Warning: GPT4Selector import failed - LLM verification may not be available")

import src.config as config
from src.data_processing import load_transaction_data, process_transaction_data

# Helper function to get USDA code lookup map
def get_usda_lookup():
    """Build a lookup map from product codes to USDA codes.
    This is a wrapper around build_usda_lookup from helper.py
    """
    # Use the build_usda_lookup function from helper.py
    lookup_map = build_usda_lookup(
        mapping_file=config.GROUND_TRUTH_FILE,
        sheet_name=config.GROUND_TRUTH_SHEET_NAME,
        id_cols=config.GROUND_TRUTH_ID_COLS,
        usda_col=config.GROUND_TRUTH_USDA_COL
    )
    
    # Error handling is done in the build_usda_lookup function
    return lookup_map


def load_transaction_data_with_usda_codes():
    """Load transaction data and add USDA codes to products.
    
    Returns:
        DataFrame: Processed transaction data with USDA codes added
    """
    print("\n1. Loading transaction data...")
    raw_data = load_transaction_data()
    if raw_data is None:
        print("Error: Failed to load transaction data")
        return None
    
    unique_products_df = process_transaction_data(raw_data)
    if unique_products_df is None or len(unique_products_df) == 0:
        print("Error: No products found after processing")
        return None
    
    # Get USDA mapping
    usda_lookup = get_usda_lookup()
    
    # Add USDA code to products
    def get_usda_code(product_code):
        # Use normalize_mapping_id from helper.py for consistent normalization
        normalized_code = normalize_mapping_id(product_code)
        return usda_lookup.get(normalized_code, 'NOT_FOUND')
    
    unique_products_df['usda_code'] = unique_products_df['product_code'].apply(get_usda_code)
    return unique_products_df


def filter_data_by_scope(unique_products_df, data_scope, limit=None):
    """Filter data based on specified scope.
    
    Args:
        unique_products_df: DataFrame with all products
        data_scope: 'limited', 'usda_only', or 'all'
        limit: Maximum number of products to process for 'limited' scope
        
    Returns:
        DataFrame: Filtered data according to scope
    """
    print(f"\n2. Applying data scope: {data_scope}")
    
    if data_scope == 'limited':
        # First filter for only products with valid USDA codes
        valid_usda_df = unique_products_df[unique_products_df['usda_code'] != 'NOT_FOUND'].copy()
        print(f"Found {len(valid_usda_df)} products with valid USDA codes out of {len(unique_products_df)}")
        
        # Then apply the limit if specified
        if limit and limit > 0 and limit < len(valid_usda_df):
            filtered_df = valid_usda_df.sample(limit, random_state=42)
            print(f"Randomly selected {limit} products with valid USDA codes")
        else:
            filtered_df = valid_usda_df
            print(f"Using all {len(filtered_df)} products with valid USDA codes (limit not applied)")
    
    elif data_scope == 'usda_only':
        filtered_df = unique_products_df[unique_products_df['usda_code'] != 'NOT_FOUND'].copy()
        print(f"Selected {len(filtered_df)} products with valid USDA codes out of {len(unique_products_df)}")
    
    else:  # 'all'
        filtered_df = unique_products_df
        print(f"Using all {len(filtered_df)} products")
    
    return filtered_df


def initialize_embedder(embedding_type):
    """Initialize the appropriate embedder based on embedding type.
    
    Args:
        embedding_type: 'openai' or 'local'
    
    Returns:
        embedder: Initialized embedder object
    """
    print(f"\n3. Initializing {embedding_type} embedder...")
    
    if embedding_type == 'openai':
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key not found in environment")
            return None
        
        embedder = OpenAIEmbedder(api_key=api_key)
    else:  # 'local'
        try:
            embedder = LocalEmbedder(model_name=config.SENTENCE_TRANSFORMER_MODEL)
        except Exception as e:
            print(f"Error initializing local embedder: {e}")
            return None
    
    return embedder


def generate_embeddings(embedder, filtered_df):
    """Generate embeddings for all products in the dataset.
    
    Args:
        embedder: Initialized embedder object
        filtered_df: DataFrame with products to embed
    
    Returns:
        DataFrame: Updated DataFrame with embeddings added
    """
    print(f"\n4. Generating embeddings for {len(filtered_df)} products...")
    
    try:
        # Get product descriptions
        descriptions = filtered_df['product_description'].tolist()
        
        # For large datasets, process in batches to show progress
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            end_idx = min(i + batch_size, len(descriptions))
            batch = descriptions[i:end_idx]
            print(f"  Processing batch {i//batch_size + 1}/{(len(descriptions)-1)//batch_size + 1} "
                  f"({i}-{end_idx-1})...")
            
            batch_embeddings = embedder(batch)
            all_embeddings.extend(batch_embeddings)
        
        print(f"Successfully generated {len(all_embeddings)} embeddings")
        
        # Add embeddings to dataframe
        filtered_df = filtered_df.copy()  # Create a copy to avoid modifying the original
        filtered_df['embedding'] = all_embeddings
        
        return filtered_df
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def generate_usda_code_embeddings(embedding_type, unique_usda_codes):
    """Generate embeddings for USDA codes for reverse direction search.
    
    Args:
        embedding_type: 'openai' or 'local'
        unique_usda_codes: List of unique USDA codes
    
    Returns:
        dict: Dictionary mapping USDA codes to their embeddings
    """
    print(f"Generating USDA code embeddings using {embedding_type}...")
    usda_embeddings = {}
    
    if embedding_type == 'openai':
        # Ensure API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key not found in environment")
            return {}
        
        # Using direct API calls for USDA codes
        import requests
        
        print("Generating USDA code embeddings using OpenAI...")
        for usda_code in unique_usda_codes:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                payload = {
                    "input": usda_code,
                    "model": config.OPENAI_EMBEDDING_MODEL
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                usda_embeddings[usda_code] = np.array(result["data"][0]["embedding"])
            except Exception as e:
                print(f"Error embedding USDA code '{usda_code}': {e}")
    else:
        # Using local embeddings
        print("Generating USDA code embeddings using local embedder...")
        try:
            embedder = LocalEmbedder(model_name=config.SENTENCE_TRANSFORMER_MODEL)
            for usda_code in unique_usda_codes:
                try:
                    usda_embeddings[usda_code] = embedder([usda_code])[0]
                except Exception as e:
                    print(f"Error embedding USDA code '{usda_code}': {e}")
        except Exception as e:
            print(f"Error initializing local embedder: {e}")
    
    print(f"Generated embeddings for {len(usda_embeddings)} USDA codes")
    return usda_embeddings


def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        float: Cosine similarity score
    """
    # Ensure compatible shapes and convert to 1D arrays if needed
    vec1 = np.squeeze(vec1)
    vec2 = np.squeeze(vec2)
    
    # Calculate cosine similarity
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_similar_products(query_embedding, filtered_df, idx):
    """Find products similar to the query product (original direction).
    
    Args:
        query_embedding: Embedding of the query product
        filtered_df: DataFrame with product data
        idx: Index of the query product to skip self-comparison
    
    Returns:
        list: List of dictionaries with similarity results
    """
    similarities = []
    
    for _, row in filtered_df.iterrows():
        if _ == idx:  # Skip self
            continue
        
        # Calculate cosine similarity
        similarity = calculate_cosine_similarity(query_embedding, row['embedding'])
        
        similarities.append({
            'product_description': row['product_description'],
            'product_code': row['product_code'],
            'usda_code': row['usda_code'],
            'similarity': similarity
        })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities


def find_similar_usda_codes(query, query_embedding, usda_embeddings):
    """Find USDA codes similar to the query product (reverse direction).
    
    Args:
        query: Product description text
        query_embedding: Embedding of the query product
        usda_embeddings: Dictionary mapping USDA codes to embeddings
    
    Returns:
        list: List of dictionaries with similarity results
    """
    similarities = []
    
    for usda_code, usda_embedding in usda_embeddings.items():
        # Calculate embedding similarity
        embedding_similarity = calculate_cosine_similarity(query_embedding, usda_embedding)
        
        # Apply text preprocessing for better matching
        processed_query = preprocess_text_for_matching(query)
        processed_usda = preprocess_text_for_matching(usda_code)
        
        # Calculate text similarity bonus (currently disabled)
        text_similarity = 0.0
        
        # Calculate final weighted similarity score
        final_similarity = embedding_similarity * 0.8 + text_similarity
        
        similarities.append({
            'product_description': f"USDA Code: {usda_code}",
            'product_code': 'N/A',
            'usda_code': usda_code,
            'similarity': final_similarity,
            'embedding_similarity': embedding_similarity,
            'text_similarity': text_similarity
        })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities


def apply_cross_encoder_reranking(query, similarities, top_matches):
    """Apply cross-encoder reranking to improve similarity ranking.
    
    Args:
        query: Product description text
        similarities: List of similarity results
        top_matches: Number of top matches to rerank
    
    Returns:
        list: Reranked list of similarity results
    """
    print(f"\nApplying cross-encoder re-ranking to top {top_matches} candidates...")
    try:
        cross_encoder = CrossEncoder()
        
        # Track the original top match for comparison
        original_top_match = similarities[0]['usda_code'] if similarities else None
        
        # Apply cross-encoder reranking to top n candidates
        reranked_similarities = cross_encoder.rerank(
            query, similarities[:min(top_matches, len(similarities))]
        )
        
        # Report if ranking changed
        if reranked_similarities and original_top_match and original_top_match != reranked_similarities[0]['usda_code']:
            print(f"Cross-encoder changed top match: {original_top_match} → {reranked_similarities[0]['usda_code']}")
            print(f"Original embedding score: {reranked_similarities[0].get('embedding_similarity', 0):.4f}, "
                  f"Cross-encoder score: {reranked_similarities[0].get('cross_encoder_score', 0):.4f}")
        
        return reranked_similarities
    except Exception as e:
        print(f"Error in cross-encoder reranking: {e}")
        print("Falling back to embedding-only similarity.")
        return similarities


def apply_llm_verification(query, similarities, k_samples):
    """Apply LLM verification to further improve rankings.
    
    Args:
        query: Product description text
        similarities: List of similarity results
        k_samples: Number of top candidates to verify
    
    Returns:
        list: List of similarities with LLM verification applied
    """
    print(f"\nApplying LLM verification on top {k_samples} candidates...")
    
    try:
        # Prepare candidates for LLM
        candidates = []
        for match in similarities[:k_samples]:
            candidates.append((match['usda_code'], match.get('cross_encoder_score', match['similarity'])))
        
        # Use direct API calls for LLM verification
        import requests
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment")
        
        # Format candidates for the prompt
        candidates_text = "\n".join([
            f"{i+1}. {code} (score: {score:.4f})"
            for i, (code, score) in enumerate(candidates)
        ])
        
        # Process the query for better analysis
        processed_query = preprocess_text_for_matching(query)
        
        # Extract product characteristics
        potential_cuts = []
        cut_indicators = ["ribeye", "strip", "loin", "brisket", "chuck", "round", "flat", 
                         "inside", "outside", "skirt", "flank", "tenderloin", "sirloin"]
        for cut in cut_indicators:
            if cut in processed_query:
                potential_cuts.append(cut)
        
        # Extract grade information
        potential_grade = ""
        if "choice" in processed_query:
            potential_grade = "Choice"
        elif "select" in processed_query:
            potential_grade = "Select"
        elif "prime" in processed_query:
            potential_grade = "Prime"
        
        # Extract bone status
        bone_status = ""
        if "boneless" in processed_query or "bnls" in processed_query:
            bone_status = "Boneless"
        elif "bone in" in processed_query or "bonein" in processed_query:
            bone_status = "Bone-in"
        
        # Create a formatted product analysis
        product_analysis = f"""DETAILED PRODUCT ANALYSIS:
1. Original Description: {query}
2. Processed Description: {processed_query}
3. Potential Cuts Identified: {', '.join(potential_cuts) if potential_cuts else 'Not specified'}
4. Grade: {potential_grade if potential_grade else 'Not specified'}
5. Bone Status: {bone_status if bone_status else 'Not specified'}
"""
        
        # Create enhanced prompt
        prompt = f"""
You are a meat product classification expert specializing in USDA codes and meat cut identification. Your task is to analyze a product description and select the most appropriate USDA code from the given candidates.

PRODUCT DESCRIPTION: {query}

{product_analysis}

TOP CANDIDATE USDA CODES (already ranked by a cross-encoder model):
{candidates_text}

ANALYSIS CRITERIA:
1. Primary Cut Match: Does the USDA code correctly identify the primary cut of meat (ribeye, strip, round, etc.)?
2. Secondary Characteristics: Does it match secondary attributes (bone status, trim level, grade)?
3. Specific Terminology: Look for specific industry terms that match exactly ("flap meat", "lip-on", etc.)

USDA CODE STRUCTURE REFERENCE:
- Numeric codes (e.g., 109, 112, 180) refer to specific primary cuts
- Letters often indicate sub-primal variations
- Numbers after letters often indicate trim specifications

IMPORTANT: You MUST select one of the USDA codes from the numbered list above. DO NOT suggest any other code.

Based on your analysis, which numbered USDA code is the MOST appropriate match?

Provide your answer in the following format:
SELECTED CODE: [exact USDA code from the list above]
CONFIDENCE: [score between 0 and 1]
REASONING: [detailed explanation of your selection based on the analysis criteria]
"""
        
        # Set up API call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a meat product classification expert with extensive knowledge of USDA codes, meat cuts, and butchery terminology."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 600,
            "top_p": 0.95
        }
        
        # Implement retry logic
        max_retries = 3
        retry_delay = 2
        attempt = 0
        response_text = ""
        
        while attempt < max_retries:
            try:
                api_url = "https://api.openai.com/v1/chat/completions"
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                response_data = response.json()
                response_text = response_data["choices"][0]["message"]["content"]
                break
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt >= max_retries:
                    print(f"Failed to get LLM response after {max_retries} attempts: {e}")
                    break
                print(f"Retry {attempt}/{max_retries} after error: {e}")
                time.sleep(retry_delay * attempt)
        
        # Parse the response
        try:
            import re
            
            selected_code_match = re.search(r'SELECTED CODE:\s*([^\n]+)', response_text)
            selected_code = selected_code_match.group(1).strip() if selected_code_match else ""
            
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', response_text)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            
            reasoning_match = re.search(r'REASONING:\s*([^\n]*(?:\n(?!SELECTED|CONFIDENCE)[^\n]*)*)', response_text)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # Validate the selection
            valid_selection = False
            candidate_codes = [code for code, _ in candidates]
            
            # Check for exact match
            if selected_code in candidate_codes:
                valid_selection = True
            else:
                # Try fuzzy matching for minor formatting differences
                for candidate_code in candidate_codes:
                    norm_selected = re.sub(r'\s+', ' ', selected_code.lower())
                    norm_candidate = re.sub(r'\s+', ' ', candidate_code.lower())
                    
                    if norm_selected == norm_candidate or norm_selected.replace(' ', '') == norm_candidate.replace(' ', ''):
                        selected_code = candidate_code
                        valid_selection = True
                        break
            
            if valid_selection and confidence > 0:
                # Remember original top choice
                original_top = similarities[0]['usda_code']
                
                # Apply confidence-based boost
                hybrid_boost_factor = 0.5 + (confidence * 0.5)
                
                for idx, match in enumerate(similarities):
                    if match['usda_code'].strip() == selected_code.strip():
                        # Store original score
                        cross_encoder_score = match.get('cross_encoder_score', match['similarity'])
                        
                        # Apply boost
                        similarities[idx]['similarity'] += hybrid_boost_factor * (2 - similarities[idx]['similarity'])
                        
                        # Add verification metadata
                        similarities[idx]['hybrid_verified'] = True
                        similarities[idx]['llm_confidence'] = confidence
                        similarities[idx]['llm_reasoning'] = reasoning
                        similarities[idx]['cross_encoder_score'] = cross_encoder_score
                        break
                
                # Re-sort based on adjusted similarities
                similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
                
                # Report changes
                print(f"\nHybrid Verification Results:")
                print(f"Selected: {selected_code} with confidence {confidence:.2f}")
                print(f"Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"Reasoning: {reasoning}")
                
                if original_top != similarities[0]['usda_code']:
                    print(f"Hybrid changed ranking: {original_top} → {similarities[0]['usda_code']}")
            else:
                print(f"\nLLM verification failed or had zero confidence. Keeping cross-encoder rankings.")
        except Exception as e:
            print(f"\nError parsing LLM response: {e}")
            print(f"Raw response: {response_text[:100]}..." if len(response_text) > 100 else response_text)
    except Exception as e:
        print(f"Error in LLM verification step: {e}")
        print("Using cross-encoder rankings only.")
    
    return similarities

def calculate_metrics(test_samples, similarities_list):
    """Calculate performance metrics for the analysis.
    
    Args:
        test_samples: DataFrame with test samples
        similarities_list: List of similarity results for each test sample
    
    Returns:
        dict: Dictionary with performance metrics
    """
    metrics = {
        'total_samples': len(test_samples),
        'correct_top1': 0,
        'correct_top3': 0,
        'correct_top5': 0,
        'avg_rank_of_correct': 0,
        'samples_with_found_correct': 0,
        'mrr': 0  # Mean Reciprocal Rank
    }
    
    # Process each sample's results
    reciprocal_ranks = []
    ranks_of_correct = []
    
    for i, (idx, sample) in enumerate(test_samples.iterrows()):
        # Get true USDA code for this sample
        true_usda = sample['usda_code']
        similarities = similarities_list[i]
        
        # Find rank of correct match (if present)
        correct_rank = None
        for rank, match in enumerate(similarities):
            if match['usda_code'] == true_usda:
                correct_rank = rank + 1  # Convert to 1-based rank
                break
        
        # Update metrics based on rank
        if correct_rank is not None:
            # Found the correct match
            metrics['samples_with_found_correct'] += 1
            ranks_of_correct.append(correct_rank)
            reciprocal_ranks.append(1.0 / correct_rank)
            
            # Check if in top-N
            if correct_rank == 1:
                metrics['correct_top1'] += 1
            if correct_rank <= 3:
                metrics['correct_top3'] += 1
            if correct_rank <= 5:
                metrics['correct_top5'] += 1
        else:
            # Correct match not found
            reciprocal_ranks.append(0.0)
    
    # Calculate averages
    if ranks_of_correct:
        metrics['avg_rank_of_correct'] = sum(ranks_of_correct) / len(ranks_of_correct)
    
    # Calculate MRR (Mean Reciprocal Rank)
    metrics['mrr'] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    
    return metrics


def format_metrics_output(metrics):
    """Format metrics for output display.
    
    Args:
        metrics: Dictionary with performance metrics
    
    Returns:
        str: Formatted metrics string
    """
    output = f"\n===== PERFORMANCE METRICS =====\n"
    output += f"Total test samples: {metrics['total_samples']}\n"
    output += f"Correct matches found: {metrics['samples_with_found_correct']} ({metrics['samples_with_found_correct']/metrics['total_samples']*100:.1f}%)\n\n"
    
    output += f"Accuracy (Top-1): {metrics['correct_top1']/metrics['total_samples']*100:.1f}%\n"
    output += f"Accuracy (Top-3): {metrics['correct_top3']/metrics['total_samples']*100:.1f}%\n"
    output += f"Accuracy (Top-5): {metrics['correct_top5']/metrics['total_samples']*100:.1f}%\n\n"
    
    output += f"Mean Reciprocal Rank (MRR): {metrics['mrr']:.3f}\n"
    if metrics['samples_with_found_correct'] > 0:
        output += f"Average rank of correct match (when found): {metrics['avg_rank_of_correct']:.1f}\n"
    
    return output


# Main analysis function
def run_analysis(
    embedding_type: str, 
    data_scope: str, 
    limit: Optional[int] = None,
    use_llm_verify: bool = False,
    use_cross_encoder: bool = False,
    use_hybrid: bool = False,
    k_samples: int = 5,
    top_matches: int = 20,
    reverse_direction: bool = False
):
    """
    Run vector database analysis with the specified options.
    
    Args:
        embedding_type: 'openai' or 'local'
        data_scope: 'limited', 'usda_only', or 'all'
        reverse_direction: If True, find USDA codes for products. If False (default), find products for USDA codes.
        limit: Maximum number of products to process (for 'limited' scope)
        use_llm_verify: Whether to use LLM verification
        use_cross_encoder: Whether to use cross-encoder reranking
        use_hybrid: Whether to use hybrid approach (cross-encoder + LLM)
        k_samples: Number of samples to use for LLM verification
        top_matches: Number of top matches to consider for reranking
    """

    print(f"\n{'='*80}")
    direction = "Products→USDA" if reverse_direction else "USDA→Products"
    print(f"Running analysis with: Embedding={embedding_type}, Scope={data_scope}, Direction={direction}, LLM Verify={use_llm_verify}")
    
    # 1. Load and process transaction data
    unique_products_df = load_transaction_data_with_usda_codes()
    if unique_products_df is None:
        return
    
    # 2. Apply data scope filters
    filtered_df = filter_data_by_scope(unique_products_df, data_scope, limit)
    
    # 3. Initialize embedder
    embedder = initialize_embedder(embedding_type)
    if embedder is None:
        return
    
    # 4. Generate embeddings
    filtered_df = generate_embeddings(embedder, filtered_df)
    if filtered_df is None:
        return
    
    # 5. Perform sample similarity search
    print("\n5. Performing sample similarity searches...")
    
    # Determine number of samples based on data scope
    if data_scope == 'usda_only':
        # Use more samples for usda_only scope to get better statistics
        num_samples = min(100, len(filtered_df))
    else:
        # Use fewer samples for other scopes
        num_samples = min(5, len(filtered_df))
        
    print(f"Testing on {num_samples} sample products...")
    
    # Select samples for testing
    test_samples = filtered_df.sample(num_samples, random_state=42)
    
    # For the reverse direction (finding USDA codes for products), first get all unique USDA codes
    usda_embeddings = {}
    if reverse_direction:
        # Get all unique USDA codes (excluding NOT_FOUND)
        unique_usda_codes = filtered_df['usda_code'].unique().tolist()
        if 'NOT_FOUND' in unique_usda_codes:
            unique_usda_codes.remove('NOT_FOUND')
        print(f"Found {len(unique_usda_codes)} unique USDA codes for analysis")
        
        # Generate embeddings for USDA codes
        usda_embeddings = generate_usda_code_embeddings(embedding_type, unique_usda_codes)
    # Process each sample and collect results
    all_similarities = []
    
    for idx, sample in test_samples.iterrows():
        query = sample['product_description']
        query_embedding = sample['embedding']
        query_usda = sample['usda_code']
        
        print(f"\nTest query: '{query}'")
        print(f"USDA code: {query_usda}")
        
        # Calculate similarities based on direction
        if reverse_direction:
            # Reverse direction: find USDA codes for products
            similarities = find_similar_usda_codes(query, query_embedding, usda_embeddings)
        else:
            # Original direction: find similar products
            similarities = find_similar_products(query_embedding, filtered_df, idx)
        
        # Apply hybrid approach: cross-encoder + LLM verification
        if use_hybrid:
            print("\nApplying HYBRID approach (Cross-encoder + LLM)...")
            
            # Step 1: Cross-encoder reranking
            similarities = apply_cross_encoder_reranking(query, similarities, top_matches)
            
            # Step 2: LLM verification
            similarities = apply_llm_verification(query, similarities, k_samples)
            
        # Apply cross-encoder reranking only (if not using hybrid approach)
        elif use_cross_encoder:
            print("\nApplying cross-encoder re-ranking...")
            similarities = apply_cross_encoder_reranking(query, similarities, top_matches)
        
        # Store results for this sample
        all_similarities.append(similarities)
    
    # Calculate and display performance metrics
    metrics = calculate_metrics(test_samples, all_similarities)
    metrics_output = format_metrics_output(metrics)
    print(metrics_output)
    
    # Display top matches for each sample
    print("\n===== TOP MATCHES FOR EACH SAMPLE =====")
    for i, (idx, sample) in enumerate(test_samples.iterrows()):
        similarities = all_similarities[i]
        true_usda = sample['usda_code']
        query = sample['product_description']
        
        print(f"\nSample {i+1}: '{query}'")
        print(f"True USDA Code: {true_usda}")
        print("Top 3 matches:")
        
        for j, match in enumerate(similarities[:3]):
            matched_text = "" if match['usda_code'] != true_usda else " [CORRECT MATCH]" 
            print(f"  {j+1}. {match['usda_code']} (score: {match['similarity']:.4f}){matched_text}")
    
    return metrics


def get_interactive_options():
    """Get analysis options interactively from user input."""
    # 1. Choose embedding type
    print("Select embedding type:")
    print("1) OpenAI embedding")
    print("2) Local embedding (sentence-transformers)")
    
    while True:
        try:
            embedding_choice = input("\nEnter choice (1-2): ")
            if embedding_choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except Exception:
            print("Please enter a valid number.")
    
    embedding_type = 'openai' if embedding_choice == '1' else 'local'
    
    # 2. Choose data scope
    print("\nSelect data scope:")
    print("1) Limited number of test products (with valid USDA codes)")
    print("2) Only products with matching USDA code")
    print("3) All products")
    
    while True:
        try:
            scope_choice = input("\nEnter choice (1-3): ")
            if scope_choice in ['1', '2', '3']:
                break
            print("Invalid choice. Please enter a number between 1 and 3.")
        except Exception:
            print("Please enter a valid number.")
    
    # Map choice to scope
    scope_map = {'1': 'limited', '2': 'usda_only', '3': 'all'}
    data_scope = scope_map[scope_choice]
    
    # 3. Choose analysis direction
    print("\nSelect analysis direction:")
    print("1) Find products for USDA codes (original approach)")
    print("2) Find USDA codes for products (reversed approach)")
    
    while True:
        try:
            direction_choice = input("\nEnter choice (1-2): ")
            if direction_choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except Exception:
            print("Please enter a valid number.")
    
    reverse_direction = (direction_choice == '2')
    
    # 4. Get limit if applicable
    limit = None
    if scope_choice == '1':
        while True:
            try:
                limit_input = input("\nEnter product limit (e.g., 100): ")
                limit = int(limit_input)
                if limit > 0:
                    break
                print("Limit must be greater than 0.")
            except ValueError:
                print("Please enter a valid number.")
    
    # 5. Use cross-encoder for re-ranking?
    print("\nUse cross-encoder for re-ranking?")
    print("1) Yes")
    print("2) No")
    
    while True:
        try:
            cross_encoder_choice = input("\nEnter choice (1-2): ")
            if cross_encoder_choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except Exception:
            print("Please enter a valid number.")
    
    use_cross_encoder = (cross_encoder_choice == '1')
    
    # 6. Use LLM verify?
    print("\nUse LLM verification?")
    print("1) Yes")
    print("2) No")
    
    while True:
        try:
            llm_choice = input("\nEnter choice (1-2): ")
            if llm_choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except Exception:
            print("Please enter a valid number.")
    
    use_llm = (llm_choice == '1')
    
    # 7. Set k_samples if using LLM
    k_samples = 5  # Default
    if use_llm:
        while True:
            try:
                k_input = input("\nEnter number of top samples to verify (default 5): ")
                if not k_input.strip():  # User pressed Enter without typing
                    break
                k_samples = int(k_input)
                if k_samples > 0:
                    break
                print("Number must be greater than 0.")
            except ValueError:
                print("Please enter a valid number.")
    
    return embedding_type, data_scope, reverse_direction, limit, use_llm, k_samples, use_cross_encoder

def main():
    """Main function to get options from command-line arguments or interactive input."""
    import argparse
    import sys
    
    print("\n=== VectorDB Analysis Tool ===\n")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run analysis on VectorDB product matching')
    
    # Embedding type
    parser.add_argument('--embedding', '-e', type=str, choices=['openai', 'local'],
                        help='Embedding type (openai or local)')
    
    # Data scope
    parser.add_argument('--scope', '-s', type=str, choices=['limited', 'usda_only', 'all'],
                        help='Data scope (limited, usda_only, or all)')
    
    # Analysis direction - new option
    parser.add_argument('--reverse', '-r', action='store_true',
                        help='Reverse direction: find USDA codes for products')
    
    # Product limit
    parser.add_argument('--limit', '-l', type=int,
                        help='Maximum number of products to process')
    
    # LLM verification
    parser.add_argument('--llm', action='store_true',
                        help='Use LLM verification')

    # K samples for LLM
    parser.add_argument('--k-samples', '-k', type=int,
                        help='Number of samples to use for LLM verification',
                        default=5)

    # Cross-encoder for re-ranking candidates
    parser.add_argument('--cross-encoder', '-c', action='store_true',
                        help='Use cross-encoder for re-ranking candidates')
    
    # Hybrid approach
    parser.add_argument('--hybrid', action='store_true',
                        help='Use hybrid approach: cross-encoder + LLM verification')
    
    # Top matches for hybrid approach
    parser.add_argument('--top-matches', type=int, default=20,
                        help='Number of top matches to consider for re-ranking (default: 20)')
    
    # Interactive mode flag
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode with prompts')

    # Parse arguments
    args = parser.parse_args()

    # Determine if we should use interactive mode
    use_interactive = args.interactive or (
        args.embedding is None and
        args.scope is None and
        args.limit is None and
        not args.llm and
        not args.cross_encoder and
        args.k_samples == 5 and  # Default value
        not args.reverse and
        len(sys.argv) == 1  # Only the script name was provided
    )
    
    # Get options either interactively or from command line
    if use_interactive:
        print("Running in interactive mode...")
        embedding_type, data_scope, reverse_direction, limit, use_llm, k_samples, use_cross_encoder = get_interactive_options()
        use_hybrid = False  # Interactive mode doesn't support hybrid yet
    else:
        # Use command line arguments
        embedding_type = args.embedding or 'openai'
        data_scope = args.scope or 'limited'
        reverse_direction = args.reverse
        limit = args.limit if args.limit is not None else 100
        use_llm = args.llm
        k_samples = args.k_samples if args.k_samples is not None else 5
        use_cross_encoder = args.cross_encoder
        use_hybrid = args.hybrid
    
    # Print summary of options
    print("\nSelected options:")
    print(f"  - Embedding type: {embedding_type}")
    print(f"  - Data scope: {data_scope}")
    print(f"  - Analysis direction: {'Products→USDA' if reverse_direction else 'USDA→Products'}")
    if data_scope == 'limited':
        print(f"  - Product limit: {limit}")
    
    if use_hybrid:
        print(f"  - Mode: HYBRID (Cross-encoder + LLM)")
        print(f"  - Top matches considered: {args.top_matches}")
        print(f"  - K samples for LLM: {k_samples}")
    else:
        print(f"  - Cross-encoder re-ranking: {'Yes' if use_cross_encoder else 'No'}")
        print(f"  - LLM verification: {'Yes' if use_llm else 'No'}")
        if use_llm:
            print(f"  - K samples: {k_samples}")
    
    # Run the analysis with selected options
    print("\nStarting analysis with selected options...")
    run_analysis(
        embedding_type=embedding_type,
        data_scope=data_scope,
        reverse_direction=reverse_direction,
        limit=limit if data_scope == 'limited' else None,
        use_llm_verify=use_llm and not args.hybrid,  # Only use LLM verify by itself if not in hybrid mode
        use_cross_encoder=use_cross_encoder and not args.hybrid,  # Only use cross-encoder by itself if not in hybrid mode
        use_hybrid=args.hybrid,
        k_samples=k_samples,
        top_matches=args.top_matches
    )

if __name__ == "__main__":
    main()
