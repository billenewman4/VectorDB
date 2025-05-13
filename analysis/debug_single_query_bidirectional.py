import pandas as pd
import sys
import os

# Adjust path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.vectordb import ProductVectorDB, ProductEmbedder, find_similar_products, create_product_vector_db

def debug_single_query(original_query_desc: str, vector_db: ProductVectorDB, embedder: ProductEmbedder):
    print(f"\n--- Debugging Bidirectional Search for Query: '{original_query_desc}' ---")

    # --- Parameters for this debug run ---
    n_results_forward = 5
    similarity_threshold_forward = 0.25
    n_results_backward = 100  # Get all products for backward search
    similarity_threshold_backward = 0.0 # See all scores for backward search

    print(f"Forward search params: n_results={n_results_forward}, threshold={similarity_threshold_forward}")
    print(f"Backward search params (for vector_db.find_similar_products): n_results={n_results_backward}, threshold={similarity_threshold_backward}\n")

    # 1. Forward Search (Original Query -> Potential Candidates)
    print(f"Step 1: Forward Search for '{original_query_desc}'")
    forward_results_df = find_similar_products(
        query=original_query_desc,
        n_results=n_results_forward,
        similarity_threshold=similarity_threshold_forward,
        vector_db=vector_db,
        bidirectional=False # Standard search for forward pass
    )

    print("\nFull Forward Search Results (Original Query -> Candidates):")
    if forward_results_df.empty:
        print("  No candidates found in forward search meeting the threshold.")
        return
    else:
        # Ensure pandas doesn't truncate columns/rows for this specific print
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(forward_results_df)

    # 2. Backward Search (Candidate -> Original Query?)
    for idx, row in forward_results_df.iterrows():
        candidate_desc = row['product_description']
        similarity_ab = row['similarity'] # A->B similarity

        if candidate_desc.lower() == original_query_desc.lower():
            print(f"\n--- Skipping self-match candidate: '{candidate_desc}' ---")
            continue

        print(f"\n--- Processing Candidate: '{candidate_desc}' (A->B Similarity: {similarity_ab:.4f}) ---")
        print(f"Step 2: Backward Search FROM Candidate '{candidate_desc}'")

        backward_results_df = find_similar_products(
            query=candidate_desc, # Query is the candidate description
            n_results=n_results_backward, # Get all (or up to 100) products
            similarity_threshold=similarity_threshold_backward, # Effectively no threshold
            vector_db=vector_db,
            bidirectional=False # Standard search for backward pass
        )

        print(f"\nFull Backward Search Results (Candidate '{candidate_desc}' -> All DB Items):")
        if backward_results_df.empty:
            print("  No items found in backward search (this shouldn't happen with threshold 0.0 unless DB is empty).")
        else:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(backward_results_df)
        
        # Check if the original query is in these backward results
        original_query_found_in_backward = False
        similarity_ba = 0.0
        
        # Normalize original query for matching (as done in find_similar_products_bidirectional)
        original_query_desc_norm = ' '.join(original_query_desc.lower().split())

        if not backward_results_df.empty:
            # Normalize product descriptions in backward results for robust matching
            # Create a temporary normalized column for searching
            backward_results_df['normalized_description'] = backward_results_df['product_description'].apply(lambda x: ' '.join(x.lower().split()))
            
            original_query_row = backward_results_df[backward_results_df['normalized_description'] == original_query_desc_norm]
            
            if not original_query_row.empty:
                original_query_found_in_backward = True
                similarity_ba = original_query_row.iloc[0]['similarity']
                print(f"\n  SUCCESS: Original Query '{original_query_desc}' FOUND in backward search for '{candidate_desc}'.")
                print(f"    B->A Similarity ('{candidate_desc}' -> '{original_query_desc}'): {similarity_ba:.4f}")
                print(f"    Full row for original query in backward results:")
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                    print(original_query_row[['product_description', 'similarity']]) # Show only relevant columns
            else:
                print(f"\n  FAILURE: Original Query '{original_query_desc}' NOT FOUND in backward search for '{candidate_desc}'.")
                # Let's see if it's there with a slightly different normalization or case
                if original_query_desc in backward_results_df['product_description'].values:
                     print(f"    Note: Original query '{original_query_desc}' IS present if checking exact string match on non-normalized descriptions.")
                else:
                     print(f"    Note: Original query '{original_query_desc}' is not present even with exact string match on non-normalized descriptions.")

        print(f"\nSummary for candidate '{candidate_desc}':")
        print(f"  A->B ('{original_query_desc}' -> '{candidate_desc}') Similarity: {similarity_ab:.4f}")
        print(f"  B->A ('{candidate_desc}' -> '{original_query_desc}') Similarity (from B's search results): {similarity_ba:.4f}")

    print("\n--- End of Debugging Bidirectional Search ---")

if __name__ == "__main__":
    print("=== Initializing Vector Database for Debug Script ===")
    
    # This function handles data loading, DB init, and product adding internally.
    # It uses its own default paths (from src/config.py) and model names.
    # recreate=False will use existing ChromaDB if already populated (faster).
    # recreate=True will rebuild the database from the Excel file.
    product_vector_db = create_product_vector_db(recreate=False) 
    
    # Get the embedder instance from the ProductVectorDB instance returned by create_product_vector_db
    product_embedder = product_vector_db.embedder 

    print("Vector database ready.")

    # --- Queries to Debug ---
    queries_to_debug = [
        "Frozen Jumbo Shrimp",
        "Ground Turkey Frozen",
        "Organic Chicken Breast" # As a control, this one worked
    ]

    for query in queries_to_debug:
        debug_single_query(query, product_vector_db, product_embedder)

    print("\n=== Debug Script Finished ===")
