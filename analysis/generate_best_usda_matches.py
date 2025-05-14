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
    transaction_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a DataFrame showing each product's best matching USDA code.
    
    Args:
        vector_db: Initialized vector database
        mapping_df: DataFrame with ground truth mappings
        transaction_df: DataFrame with transaction data
        
    Returns:
        DataFrame with products and their best matching USDA codes
    """
    # Create embeddings for all USDA codes
    usda_embeddings = {}
    for usda_code in get_unique_usda_codes(mapping_df):
        usda_embedding = vector_db.embedder.embed_query(usda_code)
        usda_embeddings[usda_code] = usda_embedding
    
    # Create ground truth mapping for verification
    product_to_usda = create_product_to_usda_mapping(transaction_df, mapping_df)
    
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
            if "embeddings" in query_result and isinstance(query_result["embeddings"], list):
                if len(query_result["embeddings"]) > 0:
                    # Check if we have a nested list (from query() result)
                    if isinstance(query_result["embeddings"][0], list):
                        # query() result with nested lists
                        product_embedding = np.array(query_result["embeddings"][0][0])
                        product_id = query_result["ids"][0][0]
                    else:
                        # Direct get() result
                        product_embedding = np.array(query_result["embeddings"][0])
                        product_id = query_result["ids"][0]
                else:
                    print(f"Empty embeddings for product {product_code}")
                    continue
            else:
                print(f"Unexpected query_result format for product {product_code}: {query_result.keys()}")
                continue
            
            # Find best matching USDA code
            best_usda_code, similarity = find_best_usda_match(product_embedding, usda_embeddings)
            
            # Check if this is a known match in the ground truth
            known_usda_code = product_to_usda.get(product_code, None)
            is_correct_match = (known_usda_code == best_usda_code) if known_usda_code else None
            
            results.append({
                "product_id": product_id,
                "product_code": product_code,
                "product_description": product_desc,
                "best_matching_usda_code": best_usda_code,
                "similarity_score": similarity,
                "known_usda_code": known_usda_code,
                "is_correct_match": is_correct_match
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy statistics for products with known USDA codes
    known_usda_products = results_df.dropna(subset=["known_usda_code"])
    if not known_usda_products.empty:
        accuracy = known_usda_products["is_correct_match"].mean()
        print(f"\nAccuracy on {len(known_usda_products)} products with known USDA codes: {accuracy:.4f}")
    
    return results_df


def main():
    """Main function to generate best USDA matches report."""
    print("--- Starting Best USDA Matches Generator ---")
    
    # Initialize vector database
    recreate_db = False  # Set to True to force rebuild
    print(f"Initializing Vector DB (recreate={recreate_db})...")
    
    try:
        vector_db_instance, unique_products_df = create_product_vector_db(recreate=recreate_db)
        if vector_db_instance is None:
            print("Error: Failed to initialize vector database.")
            sys.exit(1)
            
        print(f"Vector DB initialized with {len(unique_products_df)} unique products.")
        
        # Load ground truth mapping
        mapping_df = load_ground_truth_mapping()
        if mapping_df.empty:
            print("Error: Ground truth mapping is empty or failed to load.")
            sys.exit(1)
            
        # Load transaction data
        transaction_df = load_transaction_data()
        if transaction_df.empty:
            print("Error: Transaction data is empty or failed to load.")
            sys.exit(1)
        
        # Generate best USDA matches report
        print("\nGenerating best USDA matches report...")
        results_df = generate_best_usda_matches(
            vector_db=vector_db_instance,
            mapping_df=mapping_df,
            transaction_df=transaction_df
        )
        
        # Save results
        results_dir = project_root / "analysis_results"
        results_dir.mkdir(exist_ok=True)
        
        if not results_df.empty:
            results_file = results_dir / "best_usda_matches.xlsx"
            results_df.to_excel(results_file, index=False)
            print(f"Results saved to: {results_file}")
            
            # Also save as CSV for easier viewing
            csv_file = results_dir / "best_usda_matches.csv"
            results_df.to_csv(csv_file, index=False)
            print(f"Results also saved as CSV to: {csv_file}")
            
            # Sort by similarity score and print top 10 and bottom 10
            print("\n--- Top 10 Highest Similarity Matches ---")
            top10 = results_df.sort_values('similarity_score', ascending=False).head(10)
            for _, row in top10.iterrows():
                sim_score = float(row['similarity_score']) if hasattr(row['similarity_score'], 'item') else row['similarity_score']
                print(f"{row['product_description']} -> {row['best_matching_usda_code']}: {sim_score:.4f}")
                
            print("\n--- Bottom 10 Lowest Similarity Matches ---")
            bottom10 = results_df.sort_values('similarity_score', ascending=True).head(10)
            for _, row in bottom10.iterrows():
                sim_score = float(row['similarity_score']) if hasattr(row['similarity_score'], 'item') else row['similarity_score']
                print(f"{row['product_description']} -> {row['best_matching_usda_code']}: {sim_score:.4f}")
            
            # Product 1040948 special analysis
            special_product = results_df[results_df["product_code"] == "1040948"]
            if not special_product.empty:
                print("\n--- Special Analysis for Product Code 1040948 ---")
                for _, row in special_product.iterrows():
                    sim_score = float(row['similarity_score']) if hasattr(row['similarity_score'], 'item') else row['similarity_score']
                    print(f"Product Description: {row['product_description']}")
                    print(f"Best Matching USDA Code: {row['best_matching_usda_code']}")
                    print(f"Similarity Score: {sim_score:.4f}")
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
