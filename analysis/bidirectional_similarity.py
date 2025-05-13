import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # For progress bars

# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import config for default parameters
from src import config

# Helper function to normalize descriptions (used throughout this module)
def normalize_description(desc: str) -> str:
    """Normalizes a text description by lowercasing and stripping whitespace.
    
    Args:
        desc: The description text to normalize
        
    Returns:
        Normalized string (lowercase, whitespace stripped)
    """
    if isinstance(desc, str):
        return desc.lower().strip()
    return str(desc).strip()

# Attempt to import from src, assuming 'VectorDB' is the project root and in PYTHONPATH
# Or that the script is run from the 'analysis' directory and 'src' is discoverable
try:
    from src.vectordb import ProductVectorDB, find_similar_products
except ImportError:
    # Fallback for cases where 'src' is not directly in path, adjust as needed
    # This might happen if running directly from 'analysis' without 'VectorDB' in PYTHONPATH
    import sys
    import os
    # Add the parent directory of 'src' to sys.path
    # Assumes 'analysis' and 'src' are siblings under 'VectorDB'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    sys.path.insert(0, project_root)
    from src.vectordb import ProductVectorDB, find_similar_products

def calculate_bidirectional_similarity(
    similarity_ab: float, 
    similarity_ba: float, 
    threshold_forward: float = 0.5,
    threshold_backward: float = 0.5
) -> bool:
    """
    Determines if similarity is bidirectional based on given thresholds.

    Args:
        similarity_ab: Similarity score from product A to product B.
        similarity_ba: Similarity score from product B to product A.
        threshold_forward: Minimum similarity score for A -> B.
        threshold_backward: Minimum similarity score for B -> A.

    Returns:
        True if similarity is bidirectional, False otherwise.
    """
    return similarity_ab >= threshold_forward and similarity_ba >= threshold_backward

def find_similar_products_bidirectional(
    query_description: str,
    vector_db: ProductVectorDB,
    n_results_initial: int = 10,
    n_results_final: int = 5,
    similarity_threshold_forward: float = 0.5,
    similarity_threshold_backward: float = 0.5,
    verbose: bool = True,
    embedder: Any = None # If not provided, will use vector_db.embedder
) -> List[Dict[str, Any]]:
    """
    Finds products that are bidirectionally similar to the query_description.

    Args:
        query_description: The product description to search for.
        vector_db: An initialized ProductVectorDB instance.
        n_results_initial: Number of initial candidates from the forward search.
        n_results_final: Maximum number of bidirectional matches to return.
        similarity_threshold_forward: Threshold for query -> candidate similarity.
        similarity_threshold_backward: Threshold for candidate -> query similarity.
        verbose: Whether to print detailed logs.
        embedder: Optional embedder instance. If None, will use vector_db.embedder.

    Returns:
        List of dictionaries containing bidirectional matches with metadata.
    """
    if verbose:
        print(f"\nFinding products similar to: '{query_description}'")
        print(f"Querying collection '{vector_db.collection_name}' for {n_results_initial} results...")
        print(f"Applying similarity threshold: > {similarity_threshold_forward}")

    # Use embedder from arguments or fallback to vector_db's embedder
    actual_embedder = embedder if embedder is not None else vector_db.embedder
    if actual_embedder is None:
        if verbose:
            print("Error: No embedder available. Cannot perform bidirectional search.")
        return []

    # 1. First query for candidate matches (forward search, A->B)
    query_embedding = actual_embedder.embed_query(query_description)
    query_embeddings_list = [query_embedding.tolist()]
    
    try:
        results = vector_db.collection.query(
            query_embeddings=query_embeddings_list,
            n_results=n_results_initial,
            include=['metadatas', 'distances', 'embeddings', 'documents']
        )
    except Exception as e:
        if verbose:
            print(f"Error querying collection: {e}")
        return []

    if not results or not results.get('ids'):
        if verbose:
            print("No results found in forward search.")
        return []

    # Process the results
    ids = results['ids'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]  
    embeddings = results['embeddings'][0]
    documents = results.get('documents', [[]])[0]  # May not always be included

    if verbose:
        print(f"\n--- Similarity Search Results ---")
        for i in range(min(10, len(ids))):  # Show first 10 results only for verbosity
            similarity = 1 - distances[i]  # Convert distance to similarity
            print(f"{i:>3}. {ids[i]:>10}  {similarity:.6f}  {distances[i]:.6f}  {documents[i] if i < len(documents) else ''}")

    # 2. Convert the original query description to normalized text format
    # This will be used for calculating backward (B->A) similarity
    query_desc_norm = normalize_description(query_description)
    query_embedding_text_only = actual_embedder.embed_query(query_desc_norm)

    # 3. List to store bidirectional matches
    bidirectional_matches = []

    # 4. Process each candidate from forward search
    for i, candidate_id in enumerate(ids):
        forward_distance = distances[i]
        forward_similarity = 1 - forward_distance  # Convert distance to similarity
        
        # Skip if forward similarity is below threshold
        if forward_similarity < similarity_threshold_forward:
            continue
            
        # Get candidate data
        candidate_metadata = metadatas[i]
        candidate_embedding = embeddings[i]
        candidate_description = candidate_metadata.get('product_description', documents[i] if i < len(documents) else 'Unknown')
        
        # Skip self-matches
        if normalize_description(candidate_description) == query_desc_norm:
            if verbose:
                print(f"Skipping self-match for '{candidate_description}'")
            continue
            
        # Calculate backward similarity (B->A)
        backward_similarity = 0.0
        try:
            # Use the embedding from the results
            backward_similarity = cosine_similarity(
                np.array(candidate_embedding).reshape(1, -1),
                query_embedding_text_only.reshape(1, -1)
            )[0][0]
        except Exception as e:
            if verbose:
                print(f"Error calculating backward similarity for '{candidate_description}': {e}")
                
        # Check bidirectional threshold
        if forward_similarity >= similarity_threshold_forward and backward_similarity >= similarity_threshold_backward:
            # It's a bidirectional match
            avg_similarity = (forward_similarity + backward_similarity) / 2
            match_info = {
                'query_description': query_description,
                'candidate_description': candidate_description,
                'forward_similarity': forward_similarity,
                'backward_similarity': backward_similarity,
                'avg_similarity': avg_similarity,
                'metadata': candidate_metadata
            }
            bidirectional_matches.append(match_info)
            
    # 5. Sort matches by average similarity and trim to requested size
    bidirectional_matches.sort(key=lambda x: x['avg_similarity'], reverse=True)
    final_matches = bidirectional_matches[:n_results_final]
    
    if verbose:
        print(f"\nFound {len(bidirectional_matches)} bidirectional matches (forward threshold={similarity_threshold_forward}, backward threshold={similarity_threshold_backward})")
        print(f"Returning top {min(n_results_final, len(bidirectional_matches))} matches sorted by average similarity")
        
    return final_matches

def test_bidirectional_similarity(
    vector_db: ProductVectorDB,
    queries_df: pd.DataFrame, # DataFrame with 'product_description' and 'product_code'
    forward_threshold: float = config.SIMILARITY_THRESHOLD_FORWARD,
    backward_threshold: float = config.SIMILARITY_THRESHOLD_BACKWARD,
    n_final_results: int = config.N_RESULTS_FINAL
) -> Tuple[float, float, float, pd.DataFrame]: # Return metrics and results DF
    """
    Tests the bidirectional similarity search against ground truth USDA codes.
    Leverages USDA codes stored in ChromaDB metadata. Iterates through provided queries.

    Args:
        vector_db: Initialized ProductVectorDB instance.
        queries_df: DataFrame containing query products, MUST include
                    'product_description' and 'product_code' columns.
        forward_threshold: Forward similarity threshold (A->B).
        backward_threshold: Backward similarity threshold (B->A_text_only).
        n_final_results: Max number of results to consider per query.

    Returns:
        Tuple containing (precision, recall, f1_score, detailed_results_df).
    """
    print("\n--- Bidirectional Similarity Evaluation ---")
    if not isinstance(vector_db, ProductVectorDB):
        print("Error: Invalid ProductVectorDB instance provided.")
        return 0.0, 0.0, 0.0, pd.DataFrame() # Return default values on error

    if 'product_description' not in queries_df.columns or 'product_code' not in queries_df.columns:
         print("Error: queries_df must contain 'product_description' and 'product_code' columns.")
         return 0.0, 0.0, 0.0, pd.DataFrame()

    total_queries = len(queries_df)
    if total_queries == 0:
        print("No queries provided for evaluation.")
        return 0.0, 0.0, 0.0, pd.DataFrame()

    all_results = []
    true_positives = 0
    total_predicted_positive = 0 # Total items returned across all queries after bidirectional filter
    total_actual_positive = 0 # Total queries that SHOULD have found a match (based on having a valid USDA code)

    print(f"Evaluating {total_queries} queries...")

    # Iterate through each query product
    for index, query_row in tqdm(queries_df.iterrows(), total=total_queries, desc="Evaluating Queries"):
        query_desc = query_row['product_description']
        query_code = query_row['product_code']
        
        # --- Find the TRUE USDA Code for the QUERY item --- 
        # Use the lookup mechanism built into vector_db
        true_usda_code_query = vector_db._get_usda_code(query_code)

        if true_usda_code_query == 'NOT_FOUND':
            # print(f"Warning: Skipping query '{query_desc}' (Code: {query_code}) as its true USDA code was not found in the mapping.")
            continue # Skip queries without a ground truth USDA code
            
        total_actual_positive += 1 # This query has a valid ground truth
        # print(f"\nQuery: '{query_desc}' (Code: {query_code}) -> True USDA: {true_usda_code_query}")

        # Perform bidirectional search for the current query description
        bidirectional_matches = find_similar_products_bidirectional(
            query_description=query_desc,
            vector_db=vector_db,
            n_results_initial=config.N_RESULTS_INITIAL_SEARCH, # Use config defaults or pass explicitly
            n_results_final=n_final_results,
            similarity_threshold_forward=forward_threshold,
            similarity_threshold_backward=backward_threshold,
            verbose=False, # No need for verbose output during batch evaluation
            embedder=vector_db.embedder # Pass the embedder from the vector_db instance
        )
        
        total_predicted_positive += len(bidirectional_matches)

        # --- Evaluate the results for this query --- 
        query_matched_correctly = False # Flag if any returned item matches the true USDA code
        if not bidirectional_matches:
             # Log a placeholder if no matches were returned for this query
              all_results.append({
                'query_description': query_desc,
                'query_code': query_code,
                'true_usda_code': true_usda_code_query,
                'candidate_description': None,
                'candidate_code': None,
                'candidate_usda_code': None,
                'forward_similarity': None,
                'backward_similarity': None,
                'is_correct_match': False # No match returned, so it's not correct
            })
        else:
            for match in bidirectional_matches:
                candidate_metadata = match['metadata']
                candidate_usda_code = candidate_metadata.get('usda_code', 'MISSING_IN_METADATA')
                candidate_desc = candidate_metadata.get('product_description', 'N/A')
                is_correct = (
                    true_usda_code_query != 'NOT_FOUND' and
                    candidate_usda_code != 'NOT_FOUND' and
                    candidate_usda_code != 'MISSING_IN_METADATA' and
                    candidate_usda_code == true_usda_code_query
                )

                # Store result details for analysis
                all_results.append({
                    'query_description': query_desc,
                    'query_code': query_code,
                    'true_usda_code': true_usda_code_query,
                    'candidate_description': candidate_desc,
                    'candidate_code': candidate_metadata.get('product_code', 'N/A'),
                    'candidate_usda_code': candidate_usda_code,
                    'forward_similarity': match['forward_similarity'],
                    'backward_similarity': match['backward_similarity'],
                    'is_correct_match': is_correct
                })

                # Check if this candidate's USDA code matches the query's true USDA code
                if is_correct:
                    # print(f"  - Correct Match Found: '{candidate_desc}' (USDA: {candidate_usda_code})")
                    true_positives += 1 # Count each correct prediction as a TP
                    query_matched_correctly = True
                    # Depending on definition: Break here if only want to know *if* a match was found?
                    # Current: Counts all correct matches within top N.

    # --- Calculate Final Metrics ---
    print("\n--- Evaluation Summary ---")
    print(f"Total Queries Evaluated (with valid ground truth USDA code): {total_actual_positive}")
    print(f"Total Bidirectional Matches Returned (Predicted Positive): {total_predicted_positive}")
    print(f"Total Correct Matches Found (True Positives): {true_positives}")

    # Precision = TP / (TP + FP) = TP / Total Predicted Positive
    precision = true_positives / total_predicted_positive if total_predicted_positive > 0 else 0.0

    # Recall = TP / (TP + FN) = TP / Total Actual Positive (queries with ground truth)
    recall = true_positives / total_actual_positive if total_actual_positive > 0 else 0.0

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")

    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame() # Handle case with no results
    return precision, recall, f1_score, results_df
