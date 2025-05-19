# Helper functions for working with USDA codes, normalizing IDs, etc.
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from src import config

def preprocess_text_for_matching(text: str) -> str:
    """
    Enhanced preprocessing function for better text matching between product descriptions and USDA codes.
    Normalizes text by expanding abbreviations, removing special characters, and standardizing format.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text optimized for matching
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace common abbreviations
    abbrev_map = {
        "bf": "beef",
        "bn": "bone",
        "bnls": "boneless",
        "bi": "bonein",
        "ch": "choice",
        "sel": "select",
        "os": "outside",
        "lip on": "lipon",
        "inside": "inside",
        "lip-on": "lipon",
        "hvy": "heavy",
        "xt": "extra",
        "trmd": "trimmed",
    }
    
    # Apply abbreviation replacements
    for abbrev, full in abbrev_map.items():
        text = re.sub(r'\b' + abbrev + r'\b', full, text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Standardize white space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Helper function to normalize IDs from the mapping file
def normalize_mapping_id(code):
    if isinstance(code, str):
        # Remove trailing '-<number>' and strip whitespace
        code = re.sub(r'-\d+$', '', code).strip()
        # Also remove any leading company prefix (single digit followed by digits)
        # This will convert patterns like '51040948' to '1040948'
        code = re.sub(r'^\d(\d+)$', r'\1', code)
    else:
        code = str(code).strip() # Convert non-strings to string and strip
    return code

# Helper function to build the lookup map
def build_usda_lookup(mapping_file=config.GROUND_TRUTH_FILE, 
                      sheet_name=config.GROUND_TRUTH_SHEET_NAME, 
                      id_cols=config.GROUND_TRUTH_ID_COLS, 
                      usda_col=config.GROUND_TRUTH_USDA_COL) -> Dict[str, str]:
    """Builds a lookup map from normalized mapping IDs to USDA codes."""
    print(f"Loading ground truth mapping from: {mapping_file}, Sheet: {sheet_name}")
    try:
        df_map = pd.read_excel(mapping_file, sheet_name=sheet_name)
        print(f"Loaded {len(df_map)} rows from mapping file.")
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {mapping_file}")
        return {}
    except Exception as e:
        print(f"Error loading mapping data: {e}")
        return {}

    # Check required columns
    required_cols = id_cols + [usda_col]
    if not all(col in df_map.columns for col in required_cols):
        print(f"Error: Missing one or more required columns in mapping file. Need: {required_cols}. Found: {df_map.columns.tolist()}")
        return {}

    lookup_map = {}
    processed_rows = 0
    skipped_rows = 0
    print(f"Building USDA lookup map using ID columns: {id_cols} -> {usda_col}")
    for _, row in tqdm(df_map.iterrows(), total=len(df_map), desc="Processing mapping rows"):
        # Keep USDA code in original format, only convert to string if needed and handle NaN
        usda_code = str(row[usda_col]) if pd.notna(row[usda_col]) else None
        if not usda_code:
            skipped_rows += 1
            continue # Skip rows with no USDA code
            
        found_id_for_row = False
        for id_col in id_cols:
            raw_id = row[id_col]
            if pd.notna(raw_id):
                normalized_id = normalize_mapping_id(raw_id)
                if normalized_id: # Ensure not empty after normalization
                    # Simple handling: store the USDA code for this normalized ID
                    # If multiple rows map the same normalized ID, the last one wins (or first if checked)
                    # Consider adding warning for conflicts if needed
                    if normalized_id in lookup_map and lookup_map[normalized_id] != usda_code:
                         # Log potential conflict if needed
                         # print(f"Warning: Normalized ID '{normalized_id}' maps to multiple USDA codes ('{lookup_map[normalized_id]}' and '{usda_code}'). Using last found.")
                         pass # Keeping last one for now
                    lookup_map[normalized_id] = usda_code
                    found_id_for_row = True
                    
        if found_id_for_row:
            processed_rows += 1
        else:
            skipped_rows += 1

    print(f"Built USDA lookup map with {len(lookup_map)} unique normalized ID entries.")
    print(f"Processed {processed_rows} rows, skipped {skipped_rows} rows (missing USDA code or all ID cols empty).")
    return lookup_map



def create_product_vector_db(recreate: bool = False, 
                         embedding_type: str = config.EMBEDDING_TYPE,
                         embedding_model_name: Optional[str] = None,
                         api_key: Optional[str] = None) -> Tuple[Any, pd.DataFrame]:
    """
    Create a complete product vector database from the new transaction data structure.
    Uses data_processing functions and stores USDA code.
    
    Args:
        recreate: Whether to delete and recreate existing collections
        embedding_type: Type of embedding model to use ('openai' or 'sentence-transformer')
        embedding_model_name: Optional specific model name (default uses config value)
        api_key: Optional API key for OpenAI embeddings
        
    Returns:
        Tuple of (Initialized vector database instance, DataFrame of unique products processed)
    """
    # Load and process transaction data using new functions
    print("Processing transaction data...")
    raw_transactions_df = load_transaction_data() # Uses paths from config
    unique_products_df = process_transaction_data(raw_transactions_df)

    if unique_products_df is None or unique_products_df.empty:
        print("Error: Failed to process transaction data. Cannot create vector DB.")
        return None, None # Return None tuple to indicate failure

    # Initialize vector database with specified embedding options
    print(f"Initializing vector database with {embedding_type} embeddings...")
    
    if embedding_type == 'openai':
        model_name = embedding_model_name or config.OPENAI_EMBEDDING_MODEL
        print(f"Using OpenAI model: {model_name}")
    else:
        model_name = embedding_model_name or config.SENTENCE_TRANSFORMER_MODEL
        print(f"Using SentenceTransformer model: {model_name}")
    
    # Import here to avoid circular imports
    from src.VectorDB.DB import ProductVectorDB
    vector_db = ProductVectorDB(
        persist_directory=str(config.CHROMA_DB_PATH),
        collection_name=config.COLLECTION_NAME,
        embedding_type=embedding_type,
        embedding_model_name=model_name,
        api_key=api_key
    )
    
    # Add products to database (this now handles USDA lookup and embedding)
    print("Adding products to vector database...")
    try:
        vector_db.add_products_to_db(unique_products_df, recreate=recreate)
    except Exception as e:
        print(f"Error occurred during add_products_to_db: {e}")
        # Depending on severity, you might want to return None here too
        raise # Re-raise for now to make the error visible
    
    # Return the DB instance and the processed unique products DataFrame (now just desc/code)
    return vector_db, unique_products_df


def find_similar_products(query: str, n_results: int = 5, similarity_threshold: Optional[float] = None, 
                           vector_db: Optional[Any] = None, 
                           initial_results: int = config.N_RESULTS_INITIAL_SEARCH,
                           embedding_type: str = config.EMBEDDING_TYPE,
                           embedding_model_name: Optional[str] = None,
                           api_key: Optional[str] = None) -> pd.DataFrame:
    """Helper function to find similar products using an existing or new DB with bi-directional similarity.
    
    Args:
        query: The search query text
        n_results: Number of final results to return
        similarity_threshold: Minimum bi-directional similarity threshold
        vector_db: Optional existing vector database instance
        initial_results: Number of initial results to fetch for bi-directional check
        embedding_type: Type of embedding model to use ('openai' or 'sentence-transformer')
        embedding_model_name: Optional specific model name (default uses config value)
        api_key: Optional API key for OpenAI embeddings
        
    Returns:
        DataFrame with results ranked by bi-directional similarity
    """
    # Create or use existing vector database
    if vector_db is None:
        print("Loading vector database...")
        vector_db, _ = create_product_vector_db(
            recreate=False,
            embedding_type=embedding_type,
            embedding_model_name=embedding_model_name,
            api_key=api_key
        ) # Don't need unique_products_df here
        if vector_db is None:
             print("Failed to load or create vector database.")
             return pd.DataFrame()
    
    # Find similar products using bi-directional similarity
    print(f"Finding products similar to: '{query}' using bi-directional similarity")
    results_df = vector_db.get_similar_products(
        query, 
        n_results=n_results, 
        similarity_threshold=similarity_threshold,
        initial_results=initial_results
    )
    
    print("\n--- Bi-Directional Similarity Search Results ---")
    if not results_df.empty:
        # Display forward, backward, and bi-directional similarities
        display_cols = ['product_description', 'forward_similarity', 'backward_similarity', 'bi_directional_similarity', 'usda_code']
        print(results_df[display_cols].to_string())
    else:
        print("No similar products found.")
        
    return results_df
