#!/usr/bin/env python3
"""
USDA Code Classification Test Script

This script implements a direct USDA code classification approach:
1. Uses USDA codes as the query (rather than product descriptions)
2. Finds products with similarity above a threshold to each USDA code
3. Evaluates precision/recall against ground truth
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Set

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


def create_ground_truth_dict(mapping_df: pd.DataFrame, transaction_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Create a dictionary mapping each USDA code to a set of product codes that should match it.
    Only includes product codes that exist in both the mapping and transaction data.
    
    Args:
        mapping_df: DataFrame with ground truth mappings
        transaction_df: DataFrame with transaction data
        
    Returns:
        Dictionary mapping USDA codes to sets of product codes
    """
    usda_to_product_codes = {}
    
    # Get all product codes from transactions as strings
    transaction_product_codes = set(transaction_df[config.TRANSACTION_PRODUCT_CODE_COL].astype(str))
    print(f"Found {len(transaction_product_codes)} unique product codes in transaction data")
    
    # For each USDA code, find all products that should match it
    for usda_code in get_unique_usda_codes(mapping_df):
        product_codes = set()
        
        # Find all product codes in the mapping for this USDA code
        for _, row in mapping_df[mapping_df[config.GROUND_TRUTH_USDA_COL] == usda_code].iterrows():
            for id_col in config.GROUND_TRUTH_ID_COLS:
                if id_col in row and pd.notna(row[id_col]):
                    # Convert to string to ensure type consistency
                    normalized_id = str(row[id_col]).strip()
                    
                    # Only include products that exist in transaction data
                    if normalized_id in transaction_product_codes:
                        product_codes.add(normalized_id)
        
        if product_codes:
            usda_to_product_codes[usda_code] = product_codes
            
    # Print summary of ground truth data
    total_products_with_usda = sum(len(codes) for codes in usda_to_product_codes.values())
    print(f"Created ground truth with {len(usda_to_product_codes)} USDA codes and {total_products_with_usda} total products")
    
    return usda_to_product_codes


def classify_products_by_usda_code(
    vector_db: ProductVectorDB,
    usda_code: str,
    usda_description: str,
    similarity_threshold: float,
    max_results: int = 100
) -> List[Dict]:
    """
    Find all products that match a given USDA code based on similarity threshold.
    
    Args:
        vector_db: Initialized vector database instance
        usda_code: The USDA code to match against
        usda_description: The description of the USDA code to embed
        similarity_threshold: Minimum similarity score to consider a match
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries with matched products and similarity scores
    """
    # Embed the USDA description
    usda_embedding = vector_db.embedder.embed_query(usda_description)
    
    # Query the vector database
    results = vector_db.collection.query(
        query_embeddings=[usda_embedding.tolist()],
        n_results=max_results,
        include=["metadatas", "distances"]
    )
    
    # Process results
    matched_products = []
    if results and len(results["ids"][0]) > 0:
        for i, product_id in enumerate(results["ids"][0]):
            similarity = 1 - results["distances"][0][i]  # Convert distance to similarity
            metadata = results["metadatas"][0][i]
            
            if similarity >= similarity_threshold:
                matched_products.append({
                    "product_id": product_id,
                    "product_code": metadata.get("product_code", "N/A"),
                    "product_description": metadata.get("product_description", "N/A"),
                    "similarity": similarity,
                    "usda_code": usda_code
                })
    
    return matched_products


def evaluate_usda_classification(
    vector_db: ProductVectorDB,
    mapping_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    similarity_threshold: float = 0.7,
    max_results: int = 100
) -> Tuple[float, float, float, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate classification accuracy by USDA code, focusing only on products 
    that have known USDA codes according to the ground truth mapping.
    
    Args:
        vector_db: Initialized vector database
        mapping_df: DataFrame with ground truth mappings
        transaction_df: DataFrame with transaction data
        similarity_threshold: Threshold for classification
        max_results: Maximum results per USDA code query
        
    Returns:
        precision, recall, f1_score, detailed_results_df, usda_metrics_df
    """
    # Get ground truth mappings (only including products in transaction data)
    ground_truth = create_ground_truth_dict(mapping_df, transaction_df)
    
    if not ground_truth:
        print("Error: No valid ground truth mappings found")
        return 0.0, 0.0, 0.0, pd.DataFrame(), pd.DataFrame()
    
    # For each USDA code, find products that match it by similarity
    all_matches = []
    
    # Track statistics
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    usda_metrics = []
    
    # Process each USDA code with ground truth data
    for usda_code, true_product_codes in tqdm(ground_truth.items(), desc="Evaluating USDA Codes"):
        n_true_products = len(true_product_codes)
        
        # Get the description for this USDA code from the mapping file
        usda_descriptions = mapping_df[mapping_df[config.GROUND_TRUTH_USDA_COL] == usda_code]
        if usda_descriptions.empty:
            continue
            
        # Use the first description (we could enhance this to use all descriptions)
        usda_description = usda_descriptions.iloc[0][config.GROUND_TRUTH_USDA_COL]
        
        # Find products that match this USDA code by similarity
        matched_products = classify_products_by_usda_code(
            vector_db=vector_db,
            usda_code=usda_code,
            usda_description=usda_description,
            similarity_threshold=similarity_threshold,
            max_results=max_results
        )
        
        # Get product codes of matches (as strings for consistent comparison)
        predicted_product_codes = {str(p["product_code"]) for p in matched_products}
        
        # Calculate metrics
        true_positives = len(true_product_codes.intersection(predicted_product_codes))
        false_positives = len(predicted_product_codes - true_product_codes)
        false_negatives = len(true_product_codes - predicted_product_codes)
        
        # Update totals
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        
        # Calculate per-USDA code metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store USDA code metrics
        usda_metrics.append({
            "usda_code": usda_code,
            "description": usda_description,
            "n_true_products": n_true_products,
            "n_predicted": len(predicted_product_codes),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        # Store detailed match results
        for product in matched_products:
            product["is_correct"] = str(product["product_code"]) in true_product_codes
            all_matches.append(product)
    
    # Calculate overall metrics
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Convert results to DataFrames
    detailed_df = pd.DataFrame(all_matches) if all_matches else pd.DataFrame()
    usda_metrics_df = pd.DataFrame(usda_metrics) if usda_metrics else pd.DataFrame()
    
    return precision, recall, f1, detailed_df, usda_metrics_df


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


def main():
    """Main function to run USDA code classification evaluation."""
    print("--- Starting USDA Code Classification Test ---")
    
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
            
        # Load transaction data for product code validation
        transaction_df = load_transaction_data()
        if transaction_df.empty:
            print("Error: Transaction data is empty or failed to load.")
            sys.exit(1)
        
        # Run USDA classification evaluation
        print("\nRunning USDA code classification evaluation...")
        
        # Allow configuring threshold from command line
        similarity_threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
        print(f"Using similarity threshold: {similarity_threshold}")
        
        precision, recall, f1, detailed_results, usda_metrics = evaluate_usda_classification(
            vector_db=vector_db_instance,
            mapping_df=mapping_df,
            transaction_df=transaction_df,
            similarity_threshold=similarity_threshold
        )
        
        # Display results
        print("\n--- Evaluation Results ---")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall:    {recall:.4f}")
        print(f"Overall F1 Score:  {f1:.4f}")
        
        # Save results
        results_dir = project_root / "analysis_results"
        results_dir.mkdir(exist_ok=True)
        
        if not detailed_results.empty:
            detailed_file = results_dir / f"usda_classification_results_{similarity_threshold:.2f}.csv"
            detailed_results.to_csv(detailed_file, index=False)
            print(f"Detailed results saved to: {detailed_file}")
            
        if not usda_metrics.empty:
            metrics_file = results_dir / f"usda_metrics_{similarity_threshold:.2f}.csv"
            usda_metrics.to_csv(metrics_file, index=False)
            print(f"Per-USDA code metrics saved to: {metrics_file}")
            
            # Show top and bottom performing USDA codes
            if len(usda_metrics) > 5:
                print("\n--- Top 5 Best Performing USDA Codes ---")
                top5 = usda_metrics.sort_values('f1', ascending=False).head(5)
                for _, row in top5.iterrows():
                    print(f"{row['usda_code']}: F1={row['f1']:.4f}, Precision={row['precision']:.4f}, Recall={row['recall']:.4f}")
                    
                print("\n--- Bottom 5 Worst Performing USDA Codes ---")
                bottom5 = usda_metrics.sort_values('f1', ascending=True).head(5)
                for _, row in bottom5.iterrows():
                    print(f"{row['usda_code']}: F1={row['f1']:.4f}, Precision={row['precision']:.4f}, Recall={row['recall']:.4f}")
            
            # Generate summary of products with USDA codes
            print("\n--- Products with USDA Codes Summary ---")
            total_products_with_usda = sum(row['n_true_products'] for _, row in usda_metrics.iterrows())
            print(f"Total Products with Known USDA Codes: {total_products_with_usda}")
            print(f"Total Unique USDA Codes: {len(usda_metrics)}")
            print(f"Average Products per USDA Code: {total_products_with_usda / len(usda_metrics):.2f}")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
