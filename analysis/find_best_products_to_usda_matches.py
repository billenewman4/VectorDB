#!/usr/bin/env python3
"""
Product to USDA Best Match Generator

This script is a modified version of generate_best_usda_matches.py that reverses the approach:
Instead of using USDA codes to find similar products, it takes product descriptions
and finds the most similar USDA codes.

It maintains all the same options as the original script:
- Support for different embedding models (SentenceTransformer and OpenAI)
- Optional GPT-4 selector for improving match quality
- Detailed accuracy evaluation
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
    return "NOT_FOUND", 0.0  # Return default values if no matches found


def generate_best_products_to_usda_matches(
    embedding_function,
    mapping_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    usda_embeddings: Dict[str, np.ndarray],
    product_to_usda: Dict[str, str],
    use_gpt4_selector: bool = False,
    top_k_candidates: int = 5,
    test_limit: int = 0
) -> pd.DataFrame:
    """
    For each product, find the most similar USDA code.
    This is the reverse of the original approach, which found products for USDA codes.
    
    Args:
        embedding_function: Function to generate embeddings from text
        mapping_df: DataFrame with ground truth mappings
        transaction_df: DataFrame with transaction data
        usda_embeddings: Dictionary mapping USDA codes to their embeddings
        product_to_usda: Dictionary mapping product codes to their known USDA codes
        use_gpt4_selector: Whether to use GPT-4 to select from top matches
        top_k_candidates: Number of top candidates to consider for GPT-4 selection
        test_limit: Limit the number of products to test
        
    Returns:
        DataFrame with products and their best matching USDA codes
    """
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
    
    # Initialize GPT-4 selector if requested
    if use_gpt4_selector and gpt4_available:
        try:
            api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Warning: No OpenAI API key found. Falling back to embedding-only matching.")
                gpt4_selector = None
                use_gpt4_selector = False
            else:
                gpt4_selector = GPT4Selector(api_key=api_key)
        except Exception as e:
            print(f"Error initializing GPT-4 selector: {e}")
            print("Falling back to embedding-only matching.")
            gpt4_selector = None
            use_gpt4_selector = False
    else:
        gpt4_selector = None
    
    # Find best USDA match for each product
    results = []
    
    for _, row in tqdm(unique_products.iterrows(), total=len(unique_products), desc="Finding best USDA matches"):
        product_code = str(row[config.TRANSACTION_PRODUCT_CODE_COL])
        product_desc = row[config.TRANSACTION_DESC_COL]
        
        # Generate embedding for the product description
        try:
            product_embedding = embedding_function(product_desc)
        except Exception as e:
            print(f"Error embedding product {product_code}: {e}")
            continue
        
        # Extra fields for result dictionary
        extra_fields = {}
        
        if use_gpt4_selector and gpt4_selector is not None:
            # Get top k matches instead of just the best match
            top_k_matches = find_top_k_usda_matches(product_embedding, usda_embeddings, k=top_k_candidates)
            
            if top_k_matches:
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
                    # Fall back to embedding match
                    best_usda_code, similarity = find_best_usda_match(product_embedding, usda_embeddings)
            else:
                # Fall back to embedding match
                best_usda_code, similarity = "NOT_FOUND", 0.0
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
            "product_code": product_code,
            "product_description": product_desc,
            "best_matching_usda_code": best_usda_code,
            "similarity_score": similarity,
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
    Main function to find the most similar USDA code for each product.
    
    Args:
        use_gpt4_selector: Whether to use GPT-4 to select from top embedding matches
        top_k_candidates: Number of top candidates to consider for GPT-4 selection
        test_limit: Limit the number of products to test (0 = no limit)
        
    Returns:
        Tuple of (accuracy, correct_count, total_count) for metrics reporting
    """
    # Set up the project paths
    project_root = Path(config.PROJECT_ROOT) if hasattr(config, 'PROJECT_ROOT') else current_script_path.parent.parent
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
        return 0.0, 0, 0
    
    print("\nLoading transaction data...")
    transaction_df = load_transaction_data()
    if transaction_df is None or transaction_df.empty:
        print("Error: Could not load transaction data.")
        return 0.0, 0, 0
        
    print("\nExtracting unique USDA codes from mapping...")
    unique_usda_codes = get_unique_usda_codes(mapping_df)
    print(f"Found {len(unique_usda_codes)} unique USDA codes in mapping.")
    
    # Create mapping from product codes to known USDA codes
    print("\nCreating product to USDA mapping...")
    product_to_usda = create_product_to_usda_mapping(transaction_df, mapping_df)
    print(f"Found {len(product_to_usda)} products with known USDA codes.")
    
    # Initialize embedding function based on config
    print("\nInitializing embedding model...")
    if config.EMBEDDING_TYPE == 'openai':
        try:
            import openai
            model_name = config.OPENAI_EMBEDDING_MODEL
            api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OpenAI API key found.")
            
            openai.api_key = api_key
            print(f"Using OpenAI embedding model: {model_name}")
            
            def embedding_function(text):
                response = openai.embeddings.create(
                    model=model_name,
                    input=text
                )
                return np.array(response.data[0].embedding)
        except (ImportError, ValueError) as e:
            print(f"Error setting up OpenAI embeddings: {e}")
            print("Falling back to sentence-transformer embeddings.")
            # Fall back to SentenceTransformer
            from sentence_transformers import SentenceTransformer
            model_name = config.SENTENCE_TRANSFORMER_MODEL
            print(f"Using sentence-transformer model: {model_name}")
            model = SentenceTransformer(model_name)
            
            def embedding_function(text):
                return model.encode(text)
    else:
        # Use SentenceTransformer
        from sentence_transformers import SentenceTransformer
        model_name = config.SENTENCE_TRANSFORMER_MODEL
        print(f"Using sentence-transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        def embedding_function(text):
            return model.encode(text)
    
    # USDA code embedding dictionary for similarity search
    print("\nGenerating USDA code embeddings...")
    usda_embeddings = {}
    for usda_code in tqdm(unique_usda_codes, desc="Embedding USDA Codes"):
        try:
            embedding = embedding_function(usda_code)
            usda_embeddings[usda_code] = embedding
        except Exception as e:
            print(f"Error embedding USDA code '{usda_code}': {e}")
    print(f"Generated embeddings for {len(usda_embeddings)} USDA codes.")
    
    print("\nFinding most similar USDA codes for products...")
    try:
        results_df = generate_best_products_to_usda_matches(
            embedding_function=embedding_function,
            mapping_df=mapping_df,
            transaction_df=transaction_df,
            usda_embeddings=usda_embeddings,
            product_to_usda=product_to_usda,
            use_gpt4_selector=use_gpt4_selector,
            top_k_candidates=top_k_candidates,
            test_limit=test_limit
        )
        
        # Get output file paths for results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = f"_{config.EMBEDDING_TYPE}_{config.SENTENCE_TRANSFORMER_MODEL if config.EMBEDDING_TYPE == 'sentence-transformer' else config.OPENAI_EMBEDDING_MODEL}"
        gpt_suffix = "_with_gpt4" if use_gpt4_selector else ""
        output_csv = results_dir / f"product_to_usda_matches{model_suffix}{gpt_suffix}_{timestamp}.csv"
        
        # Save results to CSV
        if not results_df.empty:
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")
            
            # Print top 10 matches to console
            print("\n--- Example Top 10 Matches ---")
            top10 = results_df.sort_values('similarity_score', ascending=False).head(10)
            for _, row in top10.iterrows():
                match_status = "✓" if row['is_correct_match'] else "✗" if pd.notna(row['is_correct_match']) else "?"
                print(f"{match_status} {row['product_description'][:50]}... -> {row['best_matching_usda_code']} ({row['similarity_score']:.4f})")
            
            # Print top 10 incorrect matches (for analysis)
            incorrect = results_df[results_df['is_correct_match'] == False]
            if len(incorrect) > 0:
                print("\n--- Example Top 10 Incorrect Matches (for analysis) ---")
                top_incorrect = incorrect.sort_values('similarity_score', ascending=False).head(10)
                for _, row in top_incorrect.iterrows():
                    print(f"✗ {row['product_description'][:40]}... -> Predicted: {row['best_matching_usda_code']}, Actual: {row['known_usda_code']} ({row['similarity_score']:.4f})")
            
            # Grab accuracy metrics for return
            known_usda_products = results_df.dropna(subset=["known_usda_code"])
            if len(known_usda_products) > 0:
                correct_matches = known_usda_products[known_usda_products["is_correct_match"] == True]
                correct_count = len(correct_matches)
                known_count = len(known_usda_products)
                accuracy = correct_count / known_count if known_count > 0 else 0.0
                
                # Return metrics for potential comparison
                return accuracy, correct_count, known_count
            else:
                return 0.0, 0, 0
        else:
            print("No results generated.")
            return 0.0, 0, 0
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0, 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find most similar USDA codes for products")
    parser.add_argument("--use-gpt4", action="store_true", help="Use GPT-4 to select the best match from embedding candidates")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to consider for GPT-4 selection")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of products to test (0 = no limit)")
    
    args = parser.parse_args()
    
    main(
        use_gpt4_selector=args.use_gpt4,
        top_k_candidates=args.top_k,
        test_limit=args.limit
    )
