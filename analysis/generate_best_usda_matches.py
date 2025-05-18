#!/usr/bin/env python3
"""
USDA Best Match Generator

This script generates a report showing each product and its best matching
USDA code along with the similarity score, regardless of threshold.
Enhanced with GPT-4 selector for improved matching accuracy.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# --- Path Setup ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
analysis_dir = current_script_path.parent

# Add paths for imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))

# Import project modules
from src import config
try:
    from src.vectordb import create_product_vector_db, ProductVectorDB
    # Import GPT-4 selector - handle gracefully if not available
    try:
        from src.llm_selector import GPT4Selector
        gpt4_available = True
    except ImportError as e:
        print(f"Warning: GPT-4 selector not available: {e}")
        print("Falling back to embedding-only matching.")
        gpt4_available = False
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import necessary modules: {e}")
    sys.exit(1)


def load_ground_truth_mapping() -> pd.DataFrame:
    """
    Load ground truth mapping from the configured file.
    Returns a DataFrame with the mapping data.
    """
    try:
        mapping_df = pd.read_excel(
            config.GROUND_TRUTH_FILE,
            sheet_name=config.GROUND_TRUTH_SHEET_NAME
        )
        
        # Convert all ID columns to strings for consistent comparison
        for id_col in config.GROUND_TRUTH_ID_COLS:
            if id_col in mapping_df.columns:
                mapping_df[id_col] = mapping_df[id_col].astype(str)
        
        print(f"Loaded {len(mapping_df)} rows from ground truth mapping file.")
        return mapping_df
    except Exception as e:
        print(f"Error loading ground truth mapping: {e}")
        return pd.DataFrame()


def get_unique_usda_codes(mapping_df: pd.DataFrame) -> List[str]:
    """
    Extract all unique USDA codes from the mapping file.
    """
    if mapping_df.empty or config.GROUND_TRUTH_USDA_COL not in mapping_df.columns:
        return []
    
    unique_codes = mapping_df[config.GROUND_TRUTH_USDA_COL].dropna().unique().tolist()
    print(f"Found {len(unique_codes)} unique USDA codes in mapping file.")
    return unique_codes


def load_transaction_data() -> pd.DataFrame:
    """
    Load transaction data from the configured file.
    Returns a DataFrame with the transaction data.
    """
    try:
        transaction_df = pd.read_excel(
            config.TRANSACTION_REPORT_FILE,
            sheet_name=config.TRANSACTION_SHEET_NAME
        )
        print(f"Loaded {len(transaction_df)} rows from transaction data file.")
        return transaction_df
    except Exception as e:
        print(f"Error loading transaction data: {e}")
        return pd.DataFrame()


def create_product_to_usda_mapping(
    transaction_df: pd.DataFrame,
    mapping_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Create a dictionary mapping each product code to its known USDA code.
    """
    product_to_usda = {}
    
    # Get all product codes from transactions as strings
    transaction_product_codes = set(transaction_df[config.TRANSACTION_PRODUCT_CODE_COL].astype(str))
    
    # For each USDA code, find all products that match it in the ground truth
    for usda_code in get_unique_usda_codes(mapping_df):
        # Find all product codes in the mapping for this USDA code
        for _, row in mapping_df[mapping_df[config.GROUND_TRUTH_USDA_COL] == usda_code].iterrows():
            for id_col in config.GROUND_TRUTH_ID_COLS:
                if id_col in row and pd.notna(row[id_col]):
                    # Convert to string to ensure type consistency
                    normalized_id = str(row[id_col]).strip()
                    
                    # Only include products that exist in transaction data
                    if normalized_id in transaction_product_codes:
                        product_to_usda[normalized_id] = usda_code
    
    print(f"Created product-to-USDA mapping with {len(product_to_usda)} entries")
    return product_to_usda


def find_top_k_usda_matches(
    product_embedding: np.ndarray,
    usda_embeddings: Dict[str, np.ndarray],
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    Finds the top k USDA codes with highest similarity to the product embedding.
    
    Args:
        product_embedding: Embedding of the product description
        usda_embeddings: Dictionary mapping USDA codes to their embeddings
        k: Number of top matches to return
        
    Returns:
        List of (usda_code, similarity_score) tuples, sorted by similarity (highest first)
    """
    similarities = []
    
    for usda_code, usda_embedding in usda_embeddings.items():
        # Calculate cosine similarity
        similarity = np.dot(product_embedding, usda_embedding) / (
            np.linalg.norm(product_embedding) * np.linalg.norm(usda_embedding)
        )
        similarities.append((usda_code, float(similarity)))
    
    # Sort by similarity (highest first) and take top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def find_best_usda_match(
    product_embedding: np.ndarray,
    usda_embeddings: Dict[str, np.ndarray]
) -> Tuple[str, float]:
    """
    Find the USDA code with the highest similarity to the product.
    
    Args:
        product_embedding: The embedding of the product description
        usda_embeddings: Dictionary mapping USDA codes to their embeddings
        
    Returns:
        Tuple of (best_matching_usda_code, similarity_score)
    """
    # Get the top match using the find_top_k_usda_matches function
    top_matches = find_top_k_usda_matches(product_embedding, usda_embeddings, k=1)
    if top_matches:
        return top_matches[0]  # Return the top match (usda_code, similarity)
    return None, 0.0  # Return default values if no matches found


def generate_best_usda_matches(
    vector_db: ProductVectorDB,
    mapping_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    usda_embeddings: Dict[str, np.ndarray],
    product_to_usda: Dict[str, str],
    use_gpt4_selector: bool = False,
    top_k_candidates: int = 5,
    test_limit: int = 0
) -> pd.DataFrame:
    """
    Generate a DataFrame showing each product's best matching USDA code.
    Can use GPT-4 to select the best match from top embedding candidates.
    
    Args:
        vector_db: Initialized vector database
        mapping_df: DataFrame with ground truth mappings
        transaction_df: DataFrame with transaction data
        usda_embeddings: Dictionary mapping USDA codes to their embeddings
        product_to_usda: Dictionary mapping product codes to their known USDA codes
        use_gpt4_selector: Whether to use GPT-4 to select from top matches
        top_k_candidates: Number of top candidates to consider for GPT-4 selection
        
    Returns:
        DataFrame with products and their best matching USDA codes
    """
    # Using the usda_embeddings and product_to_usda passed as parameters
    print(f"Using {len(usda_embeddings)} USDA code embeddings for matching")
    print(f"Using {len(product_to_usda)} products with known USDA codes for verification")
    
    # Get unique products from transaction data
    unique_products = transaction_df.drop_duplicates(subset=[config.TRANSACTION_PRODUCT_CODE_COL, config.TRANSACTION_DESC_COL])
    
    # Filter to only include products with known USDA codes for testing
    known_products = []
    for _, row in unique_products.iterrows():
        product_code = str(row[config.TRANSACTION_PRODUCT_CODE_COL])
        if product_code in product_to_usda:
            known_products.append(row)
    
    # Convert known products to DataFrame
    if known_products:
        unique_products = pd.DataFrame(known_products)
        print(f"Filtered to {len(unique_products)} products with known USDA codes for testing")
        
        # Apply test limit if specified
        if test_limit > 0 and len(unique_products) > test_limit:
            unique_products = unique_products.sample(test_limit, random_state=42)  # Use fixed seed for reproducibility
            print(f"Limited test to {len(unique_products)} randomly selected products")
    else:
        print("Warning: No products with known USDA codes found!")
        return pd.DataFrame()  # Return empty DataFrame if no known products
    
    # Find best USDA match for each product
    results = []
    
    for _, row in tqdm(unique_products.iterrows(), total=len(unique_products), desc="Finding best USDA matches"):
        product_code = str(row[config.TRANSACTION_PRODUCT_CODE_COL])
        product_desc = row[config.TRANSACTION_DESC_COL]
        
        # Get product embedding from the vector DB
        # First try to find it by ID convention
        product_id = f"item_{product_code}" # This is how IDs are formatted in the DB
        try:
            query_result = vector_db.collection.get(
                ids=[product_id],
                include=["embeddings", "metadatas"]
            )
            
            # If not found by ID, try with a query embedding
            if not query_result or len(query_result["ids"]) == 0:
                # Create an embedding for the product description
                product_query_embedding = vector_db.embedder.embed_query(product_desc)
                
                # Query the most similar item
                query_result = vector_db.collection.query(
                    query_embeddings=[product_query_embedding.tolist()],
                    n_results=1,
                    include=["embeddings", "metadatas"]
                )
        except Exception as e:
            print(f"Error querying for product {product_code}: {e}")
            query_result = None
        
        # If product is found in the vector DB
        if query_result and len(query_result["ids"]) > 0:
            # Handle different return formats between get() and query()
            if "embeddings" in query_result:
                # Try to extract the embedding, handling different ChromaDB versions
                try:
                    if isinstance(query_result["embeddings"], list):
                        if len(query_result["embeddings"]) > 0:
                            # Handle nested lists from query() result
                            if isinstance(query_result["embeddings"][0], list):
                                if len(query_result["embeddings"][0]) > 0:
                                    # New ChromaDB format with nested lists
                                    product_embedding = np.array(query_result["embeddings"][0][0])
                                    product_id = query_result["ids"][0][0] if isinstance(query_result["ids"][0], list) else query_result["ids"][0]
                                else:
                                    print(f"Empty nested embeddings for product {product_code}")
                                    continue
                            else:
                                # Direct get() result in older ChromaDB
                                product_embedding = np.array(query_result["embeddings"][0])
                                product_id = query_result["ids"][0]
                        else:
                            print(f"Empty embeddings for product {product_code}")
                            continue
                    else:
                        print(f"Unexpected embeddings format for product {product_code}")
                        continue
                except Exception as e:
                    print(f"Error processing embeddings for product {product_code}: {e}")
                    # Fallback approach - try to get the embedding via data key in newer ChromaDB versions
                    if "data" in query_result and len(query_result["data"]) > 0:
                        try:
                            if isinstance(query_result["data"][0], dict) and "embedding" in query_result["data"][0]:
                                product_embedding = np.array(query_result["data"][0]["embedding"])
                                product_id = query_result["ids"][0]
                            else:
                                print(f"Could not find embedding in data for product {product_code}")
                                continue
                        except Exception as e2:
                            print(f"Error extracting embedding from data for product {product_code}: {e2}")
                            continue
                    else:
                        continue
            else:
                # For newer ChromaDB versions that include embedding in 'data'
                if "data" in query_result and len(query_result["data"]) > 0:
                    try:
                        if isinstance(query_result["data"][0], dict) and "embedding" in query_result["data"][0]:
                            product_embedding = np.array(query_result["data"][0]["embedding"])
                            product_id = query_result["ids"][0]
                        else:
                            print(f"Could not find embedding in data for product {product_code}")
                            continue
                    except Exception as e:
                        print(f"Error extracting embedding from data for product {product_code}: {e}")
                        continue
                else:
                    print(f"No embeddings or data found for product {product_code}: {query_result.keys()}")
                    continue
            
            # Extra fields for result dictionary
            extra_fields = {}
            
            if use_gpt4_selector and gpt4_available:
                # Get top k matches instead of just the best match
                top_k_matches = find_top_k_usda_matches(product_embedding, usda_embeddings, k=top_k_candidates)
                
                # Initialize GPT-4 selector if not already initialized
                if 'gpt4_selector' not in globals():
                    global gpt4_selector
                    try:
                        api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
                        if not api_key:
                            print("Warning: No OpenAI API key found. Falling back to embedding-only matching.")
                            gpt4_selector = None
                        else:
                            gpt4_selector = GPT4Selector(api_key=api_key)
                    except Exception as e:
                        print(f"Error initializing GPT-4 selector: {e}")
                        print("Falling back to embedding-only matching.")
                        gpt4_selector = None
                        use_gpt4_selector = False
                
                if gpt4_selector is not None and top_k_matches:
                    # Use GPT-4 to select the best match
                    try:
                        best_usda_code, confidence, reasoning = gpt4_selector.select_best_match(
                            product_description=product_desc,
                            candidate_usda_codes=top_k_matches
                        )
                        
                        # Find the similarity score for the selected code
                        similarity = next((score for code, score in top_k_matches if code == best_usda_code), 0.0)
                        
                        # Add GPT-4 specific fields
                        extra_fields = {
                            "gpt4_confidence": confidence,
                            "gpt4_reasoning": reasoning,
                            "candidate_codes": ",".join([code for code, _ in top_k_matches]),
                            "candidate_scores": ",".join([f"{score:.4f}" for _, score in top_k_matches])
                        }
                        
                        print(f"GPT-4 selected USDA code: {best_usda_code} for product: {product_desc[:50]}...")
                    except Exception as e:
                        print(f"Error using GPT-4 selector: {e}")
                        # Fall back to regular embedding match
                        best_usda_code, similarity = find_best_usda_match(product_embedding, usda_embeddings)
                else:
                    # Fall back to regular embedding match
                    best_usda_code, similarity = find_best_usda_match(product_embedding, usda_embeddings)
            else:
                # Regular embedding-based matching
                best_usda_code, similarity = find_best_usda_match(product_embedding, usda_embeddings)
            
            # Convert similarity from numpy array to float if needed
            if hasattr(similarity, 'item'):
                similarity = similarity.item()
            elif isinstance(similarity, (list, np.ndarray)):
                similarity = float(similarity[0]) if len(similarity) > 0 else 0.0
            
            # Check if this is a known match in the ground truth
            known_usda_code = product_to_usda.get(product_code, None)
            is_correct_match = (known_usda_code == best_usda_code) if known_usda_code else None
            
            result_dict = {
                "product_id": product_id,
                "product_code": product_code,
                "product_description": product_desc,
                "best_matching_usda_code": best_usda_code,
                "similarity_score": similarity,  # Now a plain float, not array
                "known_usda_code": known_usda_code,
                "is_correct_match": is_correct_match
            }
            
            # Add any extra fields from GPT-4 selection
            result_dict.update(extra_fields)
            
            results.append(result_dict)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate detailed accuracy statistics for products with known USDA codes
    known_usda_products = results_df.dropna(subset=["known_usda_code"])
    unknown_usda_products = results_df[results_df["known_usda_code"].isna()]
    
    total_products = len(results_df)
    known_count = len(known_usda_products)
    unknown_count = len(unknown_usda_products)
    
    print("\n--- USDA Matching Analysis ---")
    print(f"Total products analyzed: {total_products}")
    
    if known_count > 0:
        # Count of products that matched and didn't match
        correct_matches = known_usda_products[known_usda_products["is_correct_match"] == True]
        incorrect_matches = known_usda_products[known_usda_products["is_correct_match"] == False]
        
        correct_count = len(correct_matches)
        incorrect_count = len(incorrect_matches)
        
        # Calculate percentages
        correct_pct = (correct_count / known_count) * 100
        incorrect_pct = (incorrect_count / known_count) * 100
        known_pct = (known_count / total_products) * 100
        unknown_pct = (unknown_count / total_products) * 100
        
        print(f"\nProducts with known USDA codes: {known_count} ({known_pct:.2f}% of total)")
        print(f"  - Correctly matched: {correct_count} ({correct_pct:.2f}% of known products)")
        print(f"  - Incorrectly matched: {incorrect_count} ({incorrect_pct:.2f}% of known products)")
        print(f"\nProducts without known USDA codes: {unknown_count} ({unknown_pct:.2f}% of total)")
        
        # Overall accuracy
        accuracy = correct_count / known_count if known_count > 0 else 0
        print(f"\nOverall accuracy: {accuracy:.4f} ({correct_count}/{known_count})")
    else:
        print("No products with known USDA codes found for accuracy analysis.")
    
    return results_df


def main(use_gpt4_selector: bool = False, top_k_candidates: int = 5, test_limit: int = 0):
    """
    Main function to generate best USDA matches report.
    
    Args:
        use_gpt4_selector: Whether to use GPT-4 to select from top embedding matches
        top_k_candidates: Number of top candidates to consider for GPT-4 selection
        test_limit: Limit the number of products to test (0 = no limit)
        
    Returns:
        Tuple of (accuracy, correct_count, total_count) for metrics reporting
    """
    # Set up the project paths
    project_root = Path(config.PROJECT_ROOT)
    results_dir = project_root / "analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    # Display GPT-4 selector status
    if use_gpt4_selector:
        if gpt4_available:
            print(f"Using GPT-4 to select best match from top {top_k_candidates} embedding candidates")
        else:
            print("GPT-4 selector requested but not available. Falling back to embedding-only matching.")
            use_gpt4_selector = False
    
    print("Loading ground truth mapping data...")
    mapping_df = load_ground_truth_mapping()
    if mapping_df is None or mapping_df.empty:
        print("Error: Could not load mapping data.")
        return
    
    print("\nLoading transaction data...")
    transaction_df = load_transaction_data()
    if transaction_df is None or transaction_df.empty:
        print("Error: Could not load transaction data.")
        return
        
    print("\nExtracting unique USDA codes from mapping...")
    unique_usda_codes = get_unique_usda_codes(mapping_df)
    print(f"Found {len(unique_usda_codes)} unique USDA codes in mapping.")
    
    # Create mapping from product codes to known USDA codes
    print("\nCreating product to USDA mapping...")
    product_to_usda = create_product_to_usda_mapping(transaction_df, mapping_df)
    print(f"Found {len(product_to_usda)} products with known USDA codes.")
    
    print("\nInitializing vector database...")
    try:
        vector_db = ProductVectorDB(
            persist_directory=str(config.CHROMA_DB_PATH),
            collection_name=config.COLLECTION_NAME,
            embedding_model_name=config.EMBEDDING_MODEL
        )
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        return
    
    # USDA code embedding dictionary for similarity search
    print("\nGenerating USDA code embeddings...")
    usda_embeddings = {}
    for usda_code in tqdm(unique_usda_codes, desc="Embedding USDA Codes"):
        try:
            embedding = vector_db.embedder.embed_query(usda_code)
            usda_embeddings[usda_code] = embedding
        except Exception as e:
            print(f"Error embedding USDA code '{usda_code}': {e}")
    print(f"Generated embeddings for {len(usda_embeddings)} USDA codes.")
    
    print("\nGenerating best USDA matches report...")
    try:
        results_df = generate_best_usda_matches(
            vector_db=vector_db,
            mapping_df=mapping_df,
            transaction_df=transaction_df,
            usda_embeddings=usda_embeddings,
            product_to_usda=product_to_usda,
            use_gpt4_selector=use_gpt4_selector,
            top_k_candidates=top_k_candidates,
            test_limit=test_limit
        )
        
        # Print results to terminal instead of saving to files
        if not results_df.empty:
            print("\n--- USDA Matching Results ---")
            print("Format: product_id, product_code, product_description, best_matching_usda_code, similarity_score, known_usda_code, is_correct_match")
            
            # Print header row
            print("\n" + ",".join(results_df.columns.tolist()))
            
            # Print data rows (limited to prevent overwhelming terminal)
            max_rows_to_print = 50  # Adjust this number as needed
            for idx, row in results_df.head(max_rows_to_print).iterrows():
                row_values = [str(row[col]) for col in results_df.columns]
                print(",".join(row_values))
                
            if len(results_df) > max_rows_to_print:
                print(f"\n... and {len(results_df) - max_rows_to_print} more rows (not shown to avoid terminal overflow).")
            
            # Sort by similarity score and print top 10 and bottom 10
            print("\n--- Top 10 Highest Similarity Matches ---")
            top10 = results_df.sort_values('similarity_score', ascending=False).head(10)
            for _, row in top10.iterrows():
                print(f"{row['product_description']} -> {row['best_matching_usda_code']}: {row['similarity_score']:.4f}")
                
            print("\n--- Bottom 10 Lowest Similarity Matches ---")
            bottom10 = results_df.sort_values('similarity_score', ascending=True).head(10)
            for _, row in bottom10.iterrows():
                print(f"{row['product_description']} -> {row['best_matching_usda_code']}: {row['similarity_score']:.4f}")
            
            # Get accuracy statistics
            known_usda_products = results_df.dropna(subset=["known_usda_code"])
            unknown_usda_products = results_df[results_df["known_usda_code"].isna()]
            
            total_products = len(results_df)
            known_count = len(known_usda_products)
            unknown_count = len(unknown_usda_products)
            
            if known_count > 0:
                # Count products that matched and didn't match
                correct_matches = known_usda_products[known_usda_products["is_correct_match"] == True]
                incorrect_matches = known_usda_products[known_usda_products["is_correct_match"] == False]
                
                correct_count = len(correct_matches)
                incorrect_count = len(incorrect_matches)
                
                # Calculate percentages
                correct_pct = (correct_count / known_count) * 100
                incorrect_pct = (incorrect_count / known_count) * 100
                known_pct = (known_count / total_products) * 100
                unknown_pct = (unknown_count / total_products) * 100
                accuracy = correct_count / known_count

                # Print detailed statistics to terminal
                print("\n--- USDA Matching Statistics ---")
                print(f"Total products analyzed: {total_products}")
                print(f"Products with known USDA codes: {known_count} ({known_pct:.2f}%)")
                print(f"Products with unknown USDA codes: {unknown_count} ({unknown_pct:.2f}%)")
                print(f"\nFor products with known USDA codes:")
                print(f"Correct matches: {correct_count} ({correct_pct:.2f}%)")
                print(f"Incorrect matches: {incorrect_count} ({incorrect_pct:.2f}%)")
                print(f"\nOverall accuracy: {accuracy:.4f}")

                if correct_count > 0:
                    # Display the average similarity for correct matches
                    avg_similarity_correct = correct_matches["similarity_score"].mean()
                    print(f"Average similarity score for correct matches: {avg_similarity_correct:.4f}")

                if incorrect_count > 0:
                    # Display the average similarity for incorrect matches
                    avg_similarity_incorrect = incorrect_matches["similarity_score"].mean()
                    print(f"Average similarity score for incorrect matches: {avg_similarity_incorrect:.4f}")

                # Return metrics for external use (for compare_embeddings.py)
                return accuracy, correct_count, known_count
        else:
            print("No results generated.")
            return 0.0, 0, 0
    except Exception as e:
        print(f"Error generating report: {e}")
        return 0.0, 0, 0
        
    print("\n--- Report Generation Complete ---")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate best USDA matches report')
    parser.add_argument('--use-gpt4', action='store_true', help='Use GPT-4 to select the best match from top candidates')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top candidates to consider for GPT-4 selection')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of products to test (0 = no limit)')
    args = parser.parse_args()
    
    # Run the main function with arguments
    main(use_gpt4_selector=args.use_gpt4, top_k_candidates=args.top_k, test_limit=args.limit)
