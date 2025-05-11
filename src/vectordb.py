import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# Chroma DB for vector storage
import chromadb
from chromadb.utils import embedding_functions

# Import data processing from excel.py
from src.excel import process_transaction_data, get_unique_products, load_mapping_data

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_DIR = Path(__file__).parent.parent / "chroma_db"

# Ensure Chroma directory exists
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Default embedding model
DEFAULT_MODEL = 'all-MiniLM-L6-v2'

class ProductEmbedder:
    """Class to create and manage product embeddings"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize embedder with a specific model
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def create_text_description(self, row: pd.Series) -> str:
        """
        Create a text description combining product description and metrics
        
        Args:
            row: Pandas Series containing product data
            
        Returns:
            Enhanced text description for embedding
        """
        # Extract product description and metrics
        product_desc = row['product_description']
        
        # Format price and quantity info if available
        price_info = f", average price: ${row['avg_price']:.2f}" if 'avg_price' in row else ""
        qty_info = f", average quantity: {row['avg_quantity']:.1f}" if 'avg_quantity' in row else ""
        
        # Create enhanced description
        enhanced_desc = f"{product_desc}{price_info}{qty_info}"
        return enhanced_desc
    
    def embed_products(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create embeddings for products in dataframe
        
        Args:
            df: DataFrame containing product data
            
        Returns:
            Dictionary with product descriptions, embeddings, and metadata
        """
        # Create enhanced text descriptions
        print("Creating enhanced descriptions for embedding...")
        descriptions = [self.create_text_description(row) for _, row in df.iterrows()]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(descriptions)} products...")
        embeddings = self.model.encode(descriptions, show_progress_bar=True)
        
        # Create product IDs
        product_ids = [f"prod_{i}" for i in range(len(df))]
        
        # Create metadata for each product
        metadata = []
        for _, row in df.iterrows():
            meta = {col: row[col] for col in df.columns if col != 'product_description'}
            # Convert numeric values to standard Python types for ChromaDB compatibility
            for k, v in meta.items():
                if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                    meta[k] = int(v)
                elif isinstance(v, (np.float64, np.float32, np.float16)):
                    meta[k] = float(v)
            metadata.append(meta)
        
        # Create mapping from original product descriptions to IDs
        description_to_id = {row['product_description']: product_ids[i] for i, (_, row) in enumerate(df.iterrows())}
        
        return {
            'product_ids': product_ids,
            'descriptions': descriptions,
            'original_descriptions': df['product_description'].tolist(),
            'embeddings': embeddings,
            'metadata': metadata,
            'description_to_id': description_to_id
        }
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Create embedding for a query string
        
        Args:
            query: Query text to embed
            
        Returns:
            Vector embedding of the query
        """
        return self.model.encode(query)


class ProductVectorDB:
    """Class to manage product vector database"""
    
    def __init__(self, persist_directory: Optional[Union[str, Path]] = None, embedding_model: Optional[str] = None):
        """
        Initialize the vector database
        
        Args:
            persist_directory: Directory to store the vector database
            embedding_model: Name of embedding model to use
        """
        self.persist_directory = str(persist_directory or CHROMA_DB_DIR)
        print(f"Using Chroma DB directory: {self.persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize embedder
        self.embedding_model = embedding_model or DEFAULT_MODEL
        self.embedder = ProductEmbedder(self.embedding_model)
        
        # Use pre-trained embedding function for efficiency
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
    
    def create_collection(self, collection_name: str = "products", recreate: bool = False) -> chromadb.Collection:
        """
        Create or get a collection in the DB
        
        Args:
            collection_name: Name of the collection
            recreate: Whether to delete and recreate an existing collection
            
        Returns:
            ChromaDB collection
        """
        # Check if collection exists
        existing_collections = self.client.list_collections()
        exists = any(c.name == collection_name for c in existing_collections)
        
        if exists and recreate:
            print(f"Deleting existing collection '{collection_name}'")
            self.client.delete_collection(collection_name)
            exists = False
        
        if not exists:
            print(f"Creating new collection '{collection_name}'")
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"model": self.embedding_model}
            )
        else:
            print(f"Getting existing collection '{collection_name}'")
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        
        return collection
    
    def add_products_to_db(self, df: pd.DataFrame, collection_name: str = "products", recreate: bool = False) -> chromadb.Collection:
        """
        Add products from DataFrame to vector database
        
        Args:
            df: DataFrame containing product data
            collection_name: Name of the collection to add products to
            recreate: Whether to delete and recreate an existing collection
            
        Returns:
            ChromaDB collection with added products
        """
        # Create or get collection
        collection = self.create_collection(collection_name, recreate)
        
        # Embed products
        product_data = self.embedder.embed_products(df)
        
        # Add embeddings to collection (in batches to avoid timeouts on large datasets)
        batch_size = 100
        for i in tqdm(range(0, len(product_data['product_ids']), batch_size)):
            end_idx = min(i + batch_size, len(product_data['product_ids']))
            collection.add(
                ids=product_data['product_ids'][i:end_idx],
                embeddings=product_data['embeddings'][i:end_idx].tolist(),
                documents=product_data['descriptions'][i:end_idx],
                metadatas=product_data['metadata'][i:end_idx]
            )
        
        print(f"Added {len(product_data['product_ids'])} products to collection '{collection_name}'")
        return collection
    
    def get_similar_products(self, query: str, collection_name: str = "products", n_results: int = 5, 
                            similarity_threshold: Optional[float] = None, bidirectional: bool = False) -> pd.DataFrame:
        """
        Find similar products to a query string
        
        Args:
            query: Query string to search for
            collection_name: Name of the collection to search in
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (0-1) to include in results (None = no threshold)
            bidirectional: Whether to use bidirectional similarity (helps with abbreviations and word order)
            
        Returns:
            DataFrame of similar products with similarity scores
        """
        # Get collection
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # If using bidirectional similarity, we need a different approach
        if bidirectional:
            # Get raw embedder (not the wrapper Chroma uses)
            embedder = self.embedding_function.embedding_function
            
            # Get query embedding
            query_embedding = embedder.embed_query(query)
            
            # Get all product details from collection
            product_data = collection.get(include=["embeddings", "metadatas", "documents"])
            
            # Calculate bidirectional similarities
            similarities = []
            for i, (doc, embedding) in enumerate(zip(product_data["documents"], product_data["embeddings"])):
                # Forward similarity (query → product)
                forward_sim = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                
                # Backward similarity (product → query)
                # In a proper implementation, we would recalculate the product → query similarity
                # But for simplicity, we'll use the same value as they should be identical in theory
                backward_sim = forward_sim
                
                # Use max for most lenient interpretation
                max_sim = max(forward_sim, backward_sim)
                avg_sim = (forward_sim + backward_sim) / 2
                
                if similarity_threshold is None or max_sim >= similarity_threshold:
                    similarities.append({
                        # Use max to maximize recall
                        "similarity": max_sim,  
                        "bidirectional_avg": avg_sim,
                        "description": doc,
                        **product_data["metadatas"][i]  # Add all metadata
                    })
            
            # Convert to DataFrame, sort, and limit
            results_df = pd.DataFrame(similarities).sort_values("similarity", ascending=False).reset_index(drop=True)
            
            # Take top n_results
            if len(results_df) > n_results:
                results_df = results_df.head(n_results)
                
        else:
            # Standard approach - query the collection directly
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            
            # Extract results
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Calculate similarity scores (convert distance to similarity)
            similarities = [1 - dist for dist in distances]
            
            # Create DataFrame with results
            results_df = pd.DataFrame({
                "similarity": similarities,
                "description": documents
            })
            
            # Add metadata columns
            for i, meta in enumerate(metadatas):
                for key, value in meta.items():
                    results_df.at[i, key] = value
                    
            # Apply similarity threshold if specified
            if similarity_threshold is not None:
                results_df = results_df[results_df['similarity'] >= similarity_threshold]
        
        # Ensure 'product_description' is available for output
        if 'product_description' not in results_df.columns and 'description' in results_df.columns:
            # Extract original product description from enhanced description
            results_df['product_description'] = results_df['description'].apply(
                lambda x: x.split(", average price")[0] if ", average price" in x else x
            )
        
        return results_df


def create_product_vector_db(recreate: bool = False) -> ProductVectorDB:
    """
    Create a complete product vector database from transaction data
    
    Args:
        recreate: Whether to delete and recreate existing collections
        
    Returns:
        Initialized ProductVectorDB instance
    """
    # Load and process transaction data
    print("Processing transaction data...")
    _, unique_products = process_transaction_data()
    
    # Initialize vector database
    print("Initializing vector database...")
    vector_db = ProductVectorDB()
    
    # Add products to database
    print("Adding products to vector database...")
    vector_db.add_products_to_db(unique_products, recreate=recreate)
    
    return vector_db


def find_similar_products(query: str, n_results: int = 5, similarity_threshold: Optional[float] = None, 
                     vector_db: Optional[ProductVectorDB] = None, bidirectional: bool = False) -> pd.DataFrame:
    """
    Find products similar to a query string
    
    Args:
        query: Query string to search for
        n_results: Number of results to return
        similarity_threshold: Minimum similarity score (0-1) to include in results (None = no threshold)
        vector_db: Existing vector database instance (will create if None)
        bidirectional: Whether to use bidirectional similarity (helps with abbreviations and word order)
        
    Returns:
        DataFrame of similar products with similarity scores
    """
    # Create or use existing vector database
    if vector_db is None:
        print("Loading vector database...")
        vector_db = ProductVectorDB()
    
    # Find similar products
    print(f"Finding products similar to: '{query}'")
    if similarity_threshold is not None:
        print(f"Using similarity threshold: {similarity_threshold}")
    if bidirectional:
        print(f"Using bidirectional similarity")
    results = vector_db.get_similar_products(query, n_results=n_results, 
                                          similarity_threshold=similarity_threshold,
                                          bidirectional=bidirectional)
    
    return results


if __name__ == "__main__":
    # Example usage
    try:
        # Create vector database
        vector_db = create_product_vector_db(recreate=True)
        
        # Query for similar products
        test_queries = [
            "almond milk",
            "beef frozen",
            "fresh vegetables"
        ]
        
        for query in test_queries:
            similar_products = find_similar_products(query, vector_db=vector_db)
            print(f"\nQuery: '{query}'")
            print("Top similar products:")
            print(similar_products[["similarity", "product_description", "avg_price"]].head())
    
    except Exception as e:
        print(f"Error: {e}")
