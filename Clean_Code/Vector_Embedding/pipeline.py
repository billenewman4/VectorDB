"""
Main embedding pipeline for generating embeddings for product descriptions.

This module provides high-level functions for generating embeddings for product
descriptions, using either local sentence-transformer models or OpenAI's API.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

# Import embedding models using relative imports
from Models.base_embedder import BaseEmbedder
from Models.local_embedder import LocalEmbedder
from Models.openai_embedder import OpenAIEmbedder

# Import utilities using relative imports
from Utilities.batch_processor import process_in_batches
from Utilities.normalization import normalize_vectors, check_normalized


def create_embedder(embedding_type: str = "sentence-transformer", 
                   model_name: Optional[str] = None, 
                   api_key: Optional[str] = None) -> BaseEmbedder:
    """
    Create and return an appropriate embedder based on the specified type.
    
    Args:
        embedding_type: Type of embedder to create ('sentence-transformer' or 'openai')
        model_name: Name of the model to use (defaults to appropriate model for each type)
        api_key: API key for OpenAI embeddings (required if embedding_type is 'openai')
        
    Returns:
        BaseEmbedder: Initialized embedder object
        
    Raises:
        ValueError: If embedding_type is invalid or if OpenAI API key is missing
        ImportError: If required dependencies are missing
        Exception: If embedder initialization fails
    """
    # Validate embedding type
    embedding_type = embedding_type.lower()
    valid_types = ["sentence-transformer", "openai"]
    
    if embedding_type not in valid_types:
        raise ValueError(f"Invalid embedding_type: {embedding_type}. Must be one of {valid_types}")
    
    # Create appropriate embedder
    try:
        if embedding_type == "sentence-transformer":
            # Use all-mpnet-base-v2 by default for higher quality embeddings
            default_model = "all-mpnet-base-v2"
            return LocalEmbedder(model_name=model_name or default_model)
        elif embedding_type == "openai":
            # Use text-embedding-3-small by default
            default_model = "text-embedding-3-small"
            return OpenAIEmbedder(api_key=api_key, model_name=model_name or default_model)
    except ImportError as e:
        # Specific error for missing dependencies
        if "sentence-transformers" in str(e) and embedding_type == "sentence-transformer":
            raise ImportError(
                "Missing sentence-transformers package. Install with: pip install sentence-transformers"
            ) from e
        elif "openai" in str(e) and embedding_type == "openai":
            raise ImportError(
                "Missing OpenAI package. Install with: pip install openai"
            ) from e
        else:
            raise
    except Exception as e:
        # Re-raise with more context
        raise Exception(f"Failed to create {embedding_type} embedder: {str(e)}") from e


def embed_products(df: pd.DataFrame, 
                  text_col: str = 'clustering_description',
                  embedding_type: str = 'sentence-transformer',
                  model_name: Optional[str] = None,
                  batch_size: int = 100,
                  normalize_embeddings: bool = True,
                  api_key: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Generate embeddings for product descriptions in a DataFrame.
    
    Args:
        df: DataFrame containing product descriptions
        text_col: Column name containing the text to embed
        embedding_type: Type of embedder to use ('sentence-transformer' or 'openai')
        model_name: Specific model to use (defaults based on embedding_type)
        batch_size: Number of texts to embed in each batch
        normalize_embeddings: Whether to normalize embeddings to unit length
        api_key: API key for OpenAI (required if embedding_type is 'openai')
        
    Returns:
        Tuple containing:
        - np.ndarray: Array of embeddings with shape (n_products, embedding_dim)
        - List[str]: List of corresponding product codes
        
    Raises:
        ValueError: If text_col is not in DataFrame or contains invalid values
        ImportError: If required dependencies are missing
        Exception: If embedding generation fails
    """
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
        
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        
    if 'product_code' not in df.columns:
        raise ValueError(f"Required column 'product_code' not found in DataFrame")
    
    # Extract texts and product codes
    texts = [str(text) for text in df[text_col].tolist()]
    product_codes = [str(code) for code in df['product_code'].tolist()]
    
    # Filter valid texts and codes
    valid_indices = []
    valid_texts = []
    valid_codes = []
    
    for i, (text, code) in enumerate(zip(texts, product_codes)):
        if text and text.strip() and code and code.strip():
            valid_indices.append(i)
            valid_texts.append(text.strip())
            valid_codes.append(code.strip())
    
    if not valid_texts:
        raise ValueError(f"No valid texts found in column '{text_col}'. All texts are empty or null.")
    
    print(f"Generating embeddings for {len(valid_texts)} products using {embedding_type} model")
    if len(valid_texts) < len(texts):
        print(f"Skipped {len(texts) - len(valid_texts)} invalid or empty texts")
    
    # Create embedder
    embedder = create_embedder(
        embedding_type=embedding_type,
        model_name=model_name,
        api_key=api_key
    )
    
    # Generate embeddings in batches
    embeddings = process_in_batches(
        items=valid_texts,
        processor_func=embedder,
        batch_size=batch_size,
        show_progress=True
    )
    
    # Convert to numpy array
    embeddings_array = np.vstack(embeddings)
    
    # Normalize if requested
    if normalize_embeddings:
        print("Normalizing embeddings to unit length")
        embeddings_array = normalize_vectors(embeddings_array)
    
    print(f"Successfully generated embeddings with shape {embeddings_array.shape}")
    
    return embeddings_array, valid_codes


def generate_embeddings(embedder: Union[BaseEmbedder, str], 
                       df: pd.DataFrame, 
                       text_col: str = 'product_description',
                       batch_size: int = 100,
                       normalize_embeddings: bool = True,
                       embedding_col: str = 'embedding') -> pd.DataFrame:
    """
    Generate embeddings for all products and add them to the DataFrame.
    
    Args:
        embedder: Initialized embedder object or string type ('sentence-transformer' or 'openai')
        df: DataFrame containing products to embed
        text_col: Column name containing the text to embed
        batch_size: Number of texts to embed in each batch
        normalize_embeddings: Whether to normalize embeddings to unit length
        embedding_col: Column name to store the embeddings
        
    Returns:
        pd.DataFrame: Updated DataFrame with embeddings added
        
    Raises:
        ValueError: If inputs are invalid
        TypeError: If embedder is of incorrect type
        Exception: If embedding generation fails
    """
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
        
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # Create embedder if string provided
    if isinstance(embedder, str):
        embedder = create_embedder(embedding_type=embedder)
    elif not hasattr(embedder, '__call__'):
        raise TypeError(f"Embedder must be a callable object or a string type, got {type(embedder)}")
    
    print(f"Generating embeddings for {len(df)} products...")
    
    try:
        # Get product descriptions
        descriptions = df[text_col].astype(str).tolist()
        
        # Generate embeddings in batches
        all_embeddings = process_in_batches(
            items=descriptions,
            processor_func=embedder,
            batch_size=batch_size,
            show_progress=True
        )
        
        # Normalize if requested
        if normalize_embeddings and all_embeddings:
            print("Normalizing embeddings to unit length")
            embeddings_array = np.vstack(all_embeddings)
            normalized = normalize_vectors(embeddings_array)
            all_embeddings = [normalized[i] for i in range(normalized.shape[0])]
        
        print(f"Successfully generated {len(all_embeddings)} embeddings")
        
        # Add embeddings to dataframe
        result_df = df.copy()  # Create a copy to avoid modifying the original
        result_df[embedding_col] = all_embeddings
        
        return result_df
        
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}") from e


if __name__ == "__main__":
    # Test the embedding pipeline
    try:
        import pandas as pd
        
        print("Testing embedding pipeline...")
        
        # Create a test DataFrame
        test_df = pd.DataFrame({
            'product_code': ['P001', 'P002', 'P003', 'P004'],
            'product_name': ['Apple', 'Banana', 'Orange', 'Grape'],
            'clustering_description': [
                'Fresh red apple', 
                'Yellow banana fruit', 
                'Juicy orange citrus', 
                'Purple grape bunch'
            ]
        })
        
        print("\nTest DataFrame:")
        print(test_df)
        
        # Test embed_products
        print("\nTesting embed_products...")
        embeddings, codes = embed_products(
            df=test_df,
            text_col='clustering_description',
            embedding_type='sentence-transformer',
            batch_size=2,
            normalize_embeddings=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Product codes: {codes}")
        
        # Test generate_embeddings
        print("\nTesting generate_embeddings...")
        embedder = create_embedder(embedding_type='sentence-transformer')
        df_with_embeddings = generate_embeddings(
            embedder=embedder,
            df=test_df,
            text_col='clustering_description',
            batch_size=2
        )
        
        print(f"DataFrame with embeddings shape: {df_with_embeddings.shape}")
        print(f"Embedding column data type: {type(df_with_embeddings['embedding'].iloc[0])}")
        print(f"First embedding shape: {df_with_embeddings['embedding'].iloc[0].shape}")
        
        print("\nEmbedding pipeline tests passed!")
    except Exception as e:
        print(f"Error testing embedding pipeline: {e}")
