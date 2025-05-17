import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import src.config as config
import re # For normalization
from tqdm import tqdm # Progress bar

# Import new data processing functions
from src.data_processing import load_transaction_data, process_transaction_data

class ProductEmbedder:
    """Handles embedding generation for product descriptions."""
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        print(f"Initializing sentence transformer with model: {model_name}")
        self.model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        print("Sentence transformer initialized.")

    def _prepare_metadata(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepares metadata dictionaries for ChromaDB from the DataFrame.
        Now includes product_description, product_code, and usda_code.
        """
        metadata_columns = ['product_description', 'product_code', 'usda_code']
        # Ensure all required columns are present
        for col in metadata_columns:
            if col not in df.columns:
                raise ValueError(f"Metadata preparation error: Column '{col}' not found in DataFrame.")
                
        # Ensure no NaN values in metadata columns (replace with empty string or placeholder)
        df_metadata = df[metadata_columns].fillna('N/A').astype(str)
        
        metadatas = df_metadata.to_dict('records')
        return metadatas

    def embed_products(self, df: pd.DataFrame) -> Tuple[List[str], List[np.ndarray], List[Dict[str, Any]]]:
        """
        Generates embeddings for product descriptions and prepares metadata.
        Args:
            df: DataFrame containing at least 'product_description', 'product_code', 'usda_code'.
        Returns:
            Tuple of (ids, embeddings, metadatas).
        """
        if 'product_description' not in df.columns:
            raise ValueError("DataFrame must contain 'product_description' column for embedding.")
            
        print(f"Preparing to embed {len(df)} product descriptions...")
        descriptions = df['product_description'].tolist()
        
        # Generate embeddings (ChromaDB's helper handles batching etc.)
        # The model() call expects a list of documents
        embeddings = self.model(descriptions)
        print(f"Generated {len(embeddings)} embeddings.")

        # Generate unique IDs for ChromaDB (using index for simplicity)
        ids = [f"item_{i}" for i in range(len(df))]

        # Prepare metadata
        metadatas = self._prepare_metadata(df)
        print(f"Prepared {len(metadatas)} metadata records.")

        return ids, embeddings, metadatas

    def embed_query(self, query: str) -> np.ndarray:
        """Generates embedding for a single query string."""
        # SentenceTransformerEmbeddingFunction expects a list, returns a list of embeddings
        embedding = self.model([query])[0]
        return np.array(embedding)

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

class ProductVectorDB:
    """Manages ChromaDB interactions for product embeddings."""
    def __init__(self, persist_directory: str = str(config.CHROMA_DB_PATH), 
                 collection_name: str = config.COLLECTION_NAME, 
                 embedding_model_name: str = config.EMBEDDING_MODEL):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedder = ProductEmbedder(model_name=embedding_model_name)
        
        # Build the USDA lookup map during initialization
        self.usda_lookup_map = build_usda_lookup()
        
        print(f"Getting or creating ChromaDB collection: {self.collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedder.model, # Pass the embedding function instance
            metadata={"hnsw:space": "cosine"} # Use cosine distance
        )
        print(f"Collection '{self.collection_name}' ready.")

    def _get_usda_code(self, product_code: str) -> str:
        """Looks up the USDA code for a given transaction product code."""
        # The lookup map keys are already normalized IDs from the mapping file.
        # We assume the transaction product_code might directly match one of these normalized keys.
        # If transaction codes also need normalization (e.g., removing '-<number>'), add it here.
        normalized_transaction_code = normalize_mapping_id(product_code) # Apply same normalization
        return self.usda_lookup_map.get(normalized_transaction_code, 'NOT_FOUND')

    def add_products_to_db(self, unique_products_df: pd.DataFrame, recreate: bool = False):
        """
        Adds products to the ChromaDB collection.
        Expects df with 'product_description' and 'product_code'.
        Finds USDA codes and embeds descriptions.
        """
        if recreate:
            print(f"Recreating collection: {self.collection_name}")
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception as e:
                print(f"Warning: Could not delete collection {self.collection_name} (may not exist): {e}")
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedder.model,
                metadata={"hnsw:space": "cosine"} 
            )

        print("Adding products to database...")
        if 'product_description' not in unique_products_df.columns or 'product_code' not in unique_products_df.columns:
            raise ValueError("Input DataFrame must contain 'product_description' and 'product_code' columns.")

        # Add USDA code by looking up product_code
        print("Looking up USDA codes for products...")
        unique_products_df['usda_code'] = unique_products_df['product_code'].apply(self._get_usda_code)
        
        not_found_count = (unique_products_df['usda_code'] == 'NOT_FOUND').sum()
        if not_found_count > 0:
            print(f"Warning: Could not find USDA code mapping for {not_found_count} out of {len(unique_products_df)} products.")

        # Embed products and prepare data for ChromaDB
        ids, embeddings, metadatas = self.embedder.embed_products(unique_products_df)
        
        # Add to collection in batches
        batch_size = 5000 # Define batch size, safely below the typical limit
        total_items = len(ids)
        print(f"Adding {total_items} items to ChromaDB collection '{self.collection_name}' in batches of {batch_size}...")

        for i in tqdm(range(0, total_items, batch_size), desc="Adding Batches to ChromaDB"):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            try:
                self.collection.add(
                    embeddings=[emb.tolist() for emb in batch_embeddings],
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            except Exception as e:
                print(f"\nError adding batch starting at index {i}: {e}")
                # Decide if you want to stop or continue with other batches
                # For now, let's re-raise to halt the process on error
                raise e 
                
        print(f"Successfully added {total_items} items to the collection.")

    def get_similar_products(self, query: str, n_results: int = 5, 
                              similarity_threshold: Optional[float] = None, 
                              where_filter: Optional[Dict[str, Any]] = None,
                              initial_results: int = config.N_RESULTS_INITIAL_SEARCH) -> pd.DataFrame:
        """Finds products similar to a query description using bi-directional similarity.
        
        The bi-directional similarity approach works as follows:
        1. Initial query->database similarity (forward): Get top N items from database similar to the query
        2. Database->query similarity (backward): For each item found, calculate similarity from item to query
        3. Calculate bi-directional similarity: Average of forward and backward similarities
        4. Rank results by bi-directional similarity
        
        Args:
            query: The search query text
            n_results: Number of final results to return
            similarity_threshold: Minimum similarity threshold (applied to bi-directional similarity)
            where_filter: Optional filter for ChromaDB query
            initial_results: Number of initial results to retrieve for bi-directional check
            
        Returns:
            DataFrame with results ranked by bi-directional similarity
        """
        # Embed the query once for reuse
        query_embedding = self.embedder.embed_query(query)
        query_embedding_array = np.array(query_embedding)
        
        # Ensure embedding is in the correct list-of-lists format for ChromaDB
        query_embeddings_list = [query_embedding.tolist()]
        
        # Step 1: Initial forward search (query -> database)
        print(f"Performing initial forward search on '{self.collection_name}' for {initial_results} candidates...")
        forward_results = self.collection.query(
            query_embeddings=query_embeddings_list,
            n_results=initial_results,  # Get more initial results for bi-directional check
            include=['metadatas', 'distances', 'embeddings'],  # We need embeddings for backward similarity
            where=where_filter
        )

        # Process results
        ids = forward_results.get('ids', [[]])[0]
        forward_distances = forward_results.get('distances', [[]])[0]
        metadatas = forward_results.get('metadatas', [[]])[0]
        embeddings = forward_results.get('embeddings', [[]])[0]
        
        if not ids:
            print("Query returned no results.")
            return pd.DataFrame()
        
        # Convert distance to forward similarity
        forward_similarities = [1 - d for d in forward_distances]
        
        # Step 2: Calculate backward similarity (database items -> query)
        print("Calculating backward similarities (database -> query)...")
        results_data = []
        for i, item_id in enumerate(ids):
            # Get item embedding
            item_embedding = np.array(embeddings[i])
            
            # Calculate backward similarity (item -> query)
            # Cosine similarity = dot(A, B) / (norm(A) * norm(B))
            dot_product = np.dot(item_embedding, query_embedding_array)
            norm_item = np.linalg.norm(item_embedding)
            norm_query = np.linalg.norm(query_embedding_array)
            
            backward_similarity = dot_product / (norm_item * norm_query) if norm_item * norm_query != 0 else 0
            
            # Step 3: Calculate bi-directional similarity (average of forward and backward)
            forward_similarity = forward_similarities[i]
            bi_directional_similarity = (forward_similarity + backward_similarity) / 2
            
            # Create result row
            metadata = metadatas[i]
            row = {
                'id': item_id,
                'forward_similarity': forward_similarity,
                'backward_similarity': backward_similarity,
                'bi_directional_similarity': bi_directional_similarity,
                'distance': forward_distances[i],  # Original distance from forward search
                **metadata  # Unpack all metadata fields
            }
            results_data.append(row)
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results_data)
        
        # Step 4: Rank by bi-directional similarity
        results_df = results_df.sort_values('bi_directional_similarity', ascending=False)
        
        # Apply similarity threshold if specified (to bi-directional similarity)
        if similarity_threshold is not None:
            print(f"Applying bi-directional similarity threshold: > {similarity_threshold}")
            results_df = results_df[results_df['bi_directional_similarity'] >= similarity_threshold]
            print(f"Found {len(results_df)} results after threshold.")
        
        # Take top n_results
        if len(results_df) > n_results:
            results_df = results_df.head(n_results)
        
        # Ensure required columns exist before returning
        expected_cols = ['id', 'forward_similarity', 'backward_similarity', 'bi_directional_similarity', 
                        'distance', 'product_description', 'product_code', 'usda_code']
        for col in expected_cols:
            if col not in results_df.columns:
                print(f"Warning: Expected column '{col}' not found in results DataFrame.")
                results_df[col] = 'N/A'  # Add with N/A if missing
        
        # Reorder columns for consistency
        results_df = results_df[expected_cols + [col for col in results_df.columns if col not in expected_cols]]
        
        return results_df


def create_product_vector_db(recreate: bool = False) -> Tuple[ProductVectorDB, pd.DataFrame]:
    """
    Create a complete product vector database from the new transaction data structure.
    Uses data_processing functions and stores USDA code.
    
    Args:
        recreate: Whether to delete and recreate existing collections
        
    Returns:
        Tuple of (Initialized ProductVectorDB instance, DataFrame of unique products processed)
    """
    # Load and process transaction data using new functions
    print("Processing transaction data...")
    raw_transactions_df = load_transaction_data() # Uses paths from config
    unique_products_df = process_transaction_data(raw_transactions_df)

    if unique_products_df is None or unique_products_df.empty:
        print("Error: Failed to process transaction data. Cannot create vector DB.")
        return None, None # Return None tuple to indicate failure

    # Initialize vector database (this also builds the USDA lookup map)
    print("Initializing vector database...")
    # Pass config params explicitly if needed, or rely on defaults in ProductVectorDB init
    vector_db = ProductVectorDB(
        persist_directory=str(config.CHROMA_DB_PATH),
        collection_name=config.COLLECTION_NAME,
        embedding_model_name=config.EMBEDDING_MODEL
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
                          vector_db: Optional[ProductVectorDB] = None, 
                          initial_results: int = config.N_RESULTS_INITIAL_SEARCH) -> pd.DataFrame:
    """Helper function to find similar products using an existing or new DB with bi-directional similarity.
    
    Args:
        query: The search query text
        n_results: Number of final results to return
        similarity_threshold: Minimum bi-directional similarity threshold
        vector_db: Optional existing vector database instance
        initial_results: Number of initial results to fetch for bi-directional check
        
    Returns:
        DataFrame with results ranked by bi-directional similarity
    """
    # Create or use existing vector database
    if vector_db is None:
        print("Loading vector database...")
        vector_db, _ = create_product_vector_db(recreate=False) # Don't need unique_products_df here
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

# Example usage (updated)
if __name__ == "__main__":
    # Example: Create DB (recreating it with the simpler model)
    print("\n--- Example: Creating Vector Database with Bi-Directional Similarity ---")
    # Setting recreate=True will rebuild the DB from scratch with the simpler model
    vector_db_instance, unique_prods = create_product_vector_db(recreate=True) 
    
    if vector_db_instance:
        print("\n--- Example: Querying for Similar Products (Bi-Directional Similarity) ---")
        # Example queries using descriptions from the transaction data
        if unique_prods is not None and not unique_prods.empty:
             test_queries = unique_prods['product_description'].sample(min(3, len(unique_prods))).tolist()
             # Add some queries with abbreviations
             abbrev_queries = ["Bnls Beef Stk", "Frz Stk Portn", "Sh Cut Rst-Rdy Trmd"] 
             test_queries = abbrev_queries + test_queries
        else:
             test_queries = ["Bnls chicken breast", "Frz beef", "Portn Cut"] # Queries with abbreviations
             
        for test_query in test_queries:
            find_similar_products(test_query, n_results=5, vector_db=vector_db_instance)
            print("\n" + "-"*80 + "\n") # Add separator between results
    else:
         print("\nFailed to create or load the vector database for the example.")
