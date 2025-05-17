#!/usr/bin/env python3
"""
USDA Best Match Generator

This script generates a report showing each product and its best matching
USDA code along with the similarity score, regardless of threshold.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

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
    best_usda_code = None
    best_similarity = -1.0
    
    for usda_code, usda_embedding in usda_embeddings.items():
        # Calculate cosine similarity
        similarity = np.dot(product_embedding, usda_embedding) / (
            np.linalg.norm(product_embedding) * np.linalg.norm(usda_embedding)
        )
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_usda_code = usda_code
    
    return best_usda_code, best_similarity


def generate_best_usda_matches(
    vector_db: ProductVectorDB,
    mapping_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    usda_embeddings: Dict[str, np.ndarray],
    product_to_usda: Dict[str, str]
) -> pd.DataFrame:
    """
    Generate a DataFrame showing each product's best matching USDA code.
    
    Args:
        vector_db: Initialized vector database
        mapping_df: DataFrame with ground truth mappings
        transaction_df: DataFrame with transaction data
        usda_embeddings: Dictionary mapping USDA codes to their embeddings
        product_to_usda: Dictionary mapping product codes to their known USDA codes
        
    Returns:
        DataFrame with products and their best matching USDA codes
    """
    # Using the usda_embeddings and product_to_usda passed as parameters
    print(f"Using {len(usda_embeddings)} USDA code embeddings for matching")
    print(f"Using {len(product_to_usda)} products with known USDA codes for verification")
    
    # Get unique products from transaction data
    unique_products = transaction_df.drop_duplicates(subset=[config.TRANSACTION_PRODUCT_CODE_COL, config.TRANSACTION_DESC_COL])
    print(f"Processing {len(unique_products)} unique products...")
    
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
            
            # Find best matching USDA code
            best_usda_code, similarity = find_best_usda_match(product_embedding, usda_embeddings)
            
            # Convert similarity from numpy array to float if needed
            if hasattr(similarity, 'item'):
                similarity = similarity.item()
            elif isinstance(similarity, (list, np.ndarray)):
                similarity = float(similarity[0]) if len(similarity) > 0 else 0.0
            
            # Check if this is a known match in the ground truth
            known_usda_code = product_to_usda.get(product_code, None)
            is_correct_match = (known_usda_code == best_usda_code) if known_usda_code else None
            
            results.append({
                "product_id": product_id,
                "product_code": product_code,
                "product_description": product_desc,
                "best_matching_usda_code": best_usda_code,
                "similarity_score": similarity,  # Now a plain float, not array
                "known_usda_code": known_usda_code,
                "is_correct_match": is_correct_match
            })
    
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


def main():
    # Set up the project paths
    project_root = Path(config.PROJECT_ROOT)
    results_dir = project_root / "analysis_results"
    results_dir.mkdir(exist_ok=True)
    
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
            product_to_usda=product_to_usda
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
                print("\n--- Detailed USDA Matching Statistics ---")
                print(f"{'Metric':<30} {'Count':<10} {'Percentage':<15}")
                print("-" * 55)
                print(f"{'Total Products':<30} {total_products:<10} {'':15}")
                print(f"{'Products with Known USDA':<30} {known_count:<10} {known_pct:.2f}%")
                print(f"{'Products with Unknown USDA':<30} {unknown_count:<10} {unknown_pct:.2f}%")
                print(f"{'Correct Matches':<30} {correct_count:<10} {correct_pct:.2f}%")
                print(f"{'Incorrect Matches':<30} {incorrect_count:<10} {incorrect_pct:.2f}%")
                print(f"{'Accuracy':<30} {'':10} {accuracy:.4f}")
                print("-" * 55)
            
            # Product 1040948 special analysis
            special_product = results_df[results_df["product_code"] == "1040948"]
            if not special_product.empty:
                print("\n--- Special Analysis for Product Code 1040948 ---")
                for _, row in special_product.iterrows():
                    print(f"Product Description: {row['product_description']}")
                    print(f"Best Matching USDA Code: {row['best_matching_usda_code']}")
                    print(f"Similarity Score: {row['similarity_score']:.4f}")
                    print(f"Known USDA Code: {row['known_usda_code']}")
                    print(f"Is Correct Match: {row['is_correct_match']}")
            else:
                print("\n--- Product Code 1040948 Not Found in Results ---")
    except Exception as e:
        print(f"An error occurred during report generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\n--- Report Generation Complete ---")


if __name__ == "__main__":
    main()
