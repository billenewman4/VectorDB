#!/usr/bin/env python3
"""
Simplified script to run and test bidirectional similarity analysis.
Initializes the vector database and then calls the main test function.
"""
import sys
import pandas as pd
from pathlib import Path

# --- Path Setup ---
# Get the project root directory (VectorDB)
# Assumes this script (run_bidirectional_test.py) is in VectorDB/analysis/
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

# Add project root to sys.path to allow imports like 'from src.module'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add the 'analysis' directory itself to sys.path to allow 'from bidirectional_similarity'
# This helps if 'analysis' is not a package or script is run from project root.
analysis_dir = current_script_path.parent
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))
# --- End Path Setup ---

# Now import from src and analysis
from src import config
try:
    from src.vectordb import create_product_vector_db, ProductVectorDB
    # This import should now work whether script is run from 'analysis' dir
    # or from project root 'VectorDB' dir.
    from analysis.bidirectional_similarity import test_bidirectional_similarity
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import necessary modules.")
    print(f"Details: {e}")
    print(f"Attempted Project Root: {project_root}")
    print(f"Attempted Analysis Dir: {analysis_dir}")
    print(f"Current sys.path: {sys.path}")
    print("Ensure 'src.vectordb' is a package and 'bidirectional_similarity.py' is in the 'analysis' directory.")
    sys.exit(1)

def main():
    """
    Main function to run the bidirectional similarity test.
    """
    print("--- Starting Bidirectional Similarity Test Run ---")

    # --- 1. Initialize Vector Database --- 
    # This now loads actual transaction data, processes it, builds the USDA lookup,
    # embeds products, and stores metadata (including usda_code).
    # Set recreate=False to use existing DB if available, True to rebuild.
    recreate_db = False # Set to True to force rebuild on first run or after changes
    print(f"Initializing Product Vector DB (recreate={recreate_db})...")
    try:
        vector_db_instance, unique_products_df = create_product_vector_db(recreate=recreate_db)
        
        if vector_db_instance is None or unique_products_df is None:
            print("Error: Failed to initialize vector database or process unique products.")
            sys.exit(1)
            
        if unique_products_df.empty:
             print("Warning: No unique products were processed. Evaluation may not run.")
             # Allow continuing, test function might handle empty query df
             
        # Ensure necessary columns are present for the test
        if 'product_description' not in unique_products_df.columns or 'product_code' not in unique_products_df.columns:
             print("Error: Processed unique products DataFrame is missing required 'product_description' or 'product_code' columns.")
             sys.exit(1)

    except Exception as e:
        print(f"Error during Vector DB initialization: {e}")
        import traceback
        traceback.print_exc() # Print detailed error
        sys.exit(1)

    print("Vector DB Initialized successfully.")
    print(f"Total unique products processed for potential querying: {len(unique_products_df)}")

    # --- 2. Prepare Queries for Evaluation --- 
    # We'll use ALL unique products as queries for comprehensive evaluation.
    # The test function requires a DataFrame with 'product_description' and 'product_code'.
    if len(unique_products_df) > 0:
        queries_df = unique_products_df.copy()  # Use all products (no sampling)
        print(f"Using all {len(queries_df)} unique products as queries for comprehensive evaluation.")
    else:
        print("No unique products available to use as queries. Skipping evaluation.")
        sys.exit(0) # Exit cleanly if no queries
        
    # --- 3. Run Bidirectional Similarity Test --- 
    # The test function now uses the USDA code stored in metadata.
    # We pass the DB instance and the DataFrame of queries.
    print("\nRunning bidirectional similarity evaluation...")
    try:
        # Use parameters from config or override here
        precision, recall, f1, detailed_results = test_bidirectional_similarity(
            vector_db=vector_db_instance,
            queries_df=queries_df, # Pass the DataFrame of queries
            forward_threshold=config.SIMILARITY_THRESHOLD_FORWARD,
            backward_threshold=config.SIMILARITY_THRESHOLD_BACKWARD,
            n_final_results=config.N_RESULTS_FINAL
        )
        
        print("\n--- Evaluation Complete --- ")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Optional: Save detailed results to a file
        results_dir = project_root / "analysis_results"
        results_dir.mkdir(exist_ok=True) # Ensure directory exists
        results_file = results_dir / "bidirectional_evaluation_results.csv"
        detailed_results.to_csv(results_file, index=False)
        print(f"Detailed evaluation results saved to: {results_file}")

    except Exception as e:
        print(f"An error occurred during the similarity test: {e}")
        import traceback
        traceback.print_exc() # Print detailed error
        sys.exit(1)

    print("\n--- Test Run Finished Successfully ---")

if __name__ == "__main__":
    main()
