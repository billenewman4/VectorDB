import math
import chromadb
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from src import config
from src.VectorDB.helper import normalize_mapping_id, build_usda_lookup
from src.VectorDB.localEmbedder import ProductEmbedder

class ProductVectorDB:
    """Manages ChromaDB interactions for product embeddings."""
    def __init__(self, persist_directory: str = str(config.CHROMA_DB_PATH), 
                 collection_name: str = config.COLLECTION_NAME, 
                 embedding_type: str = config.EMBEDDING_TYPE,
                 embedding_model_name: Optional[str] = None,
                 api_key: Optional[str] = None):
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedder = ProductEmbedder(
            embedding_type=embedding_type,
            model_name=embedding_model_name,
            api_key=api_key
        )
        
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
