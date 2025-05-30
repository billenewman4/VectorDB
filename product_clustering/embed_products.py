"""
Product embedding script for clustering.
Uses existing embedding infrastructure to generate embeddings for product clustering.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.VectorDB.localEmbedder import LocalEmbedder
try:
    from src.VectorDB.OpenAIEmbedder import OpenAIEmbedder
except ImportError:
    print("Warning: OpenAIEmbedder import failed - OpenAI embedding may not be available")

def embed_products(df: pd.DataFrame, 
                  text_col: str = 'clustering_description',
                  embedding_type: str = 'sentence-transformer',
                  model_name: Optional[str] = None,
                  batch_size: int = 100,
                  normalize_embeddings: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Generate embeddings for product descriptions using existing infrastructure.
    
    Args:
        df: DataFrame with product data
        text_col: Column name containing the text to embed
        embedding_type: Type of embeddings to use ('sentence-transformer' or 'openai')
        model_name: Name of specific model to use
        batch_size: Batch size for embedding generation
        normalize_embeddings: Whether to normalize embeddings to unit length
        
    Returns:
        Tuple of (embeddings array, product_codes list)
    """
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in DataFrame")
    
    # Extract text and product codes, ensure all values are strings
    texts = [str(text) for text in df[text_col].tolist()]
    product_codes = [str(code) for code in df['product_code'].tolist()]
    
    # Filter out any empty or NaN strings
    valid_indices = []
    valid_texts = []
    valid_codes = []
    
    for i, (text, code) in enumerate(zip(texts, product_codes)):
        if text and text.strip() and code and code.strip():
            valid_indices.append(i)
            valid_texts.append(text.strip())
            valid_codes.append(code.strip())
    
    texts = valid_texts
    product_codes = valid_codes
    
    print(f"After filtering, using {len(texts)} valid products for embedding")
    
    print(f"Generating embeddings for {len(texts)} products...")
    
    # Initialize appropriate embedder
    if embedding_type == 'openai':
        try:
            model_name = model_name or config.OPENAI_EMBEDDING_MODEL
            embedder = OpenAIEmbedder()
            print(f"Using OpenAI embeddings with model: {model_name}")
        except Exception as e:
            print(f"Error initializing OpenAI embedder: {e}")
            print("Falling back to sentence-transformer embeddings.")
            model_name = 'all-mpnet-base-v2'  # Default to stronger model
            embedder = LocalEmbedder(model_name=model_name)
            print(f"Using sentence-transformer embeddings with model: {model_name}")
    else:
        # Use sentence-transformer embeddings (default)
        model_name = model_name or 'all-mpnet-base-v2'  # Default to stronger model
        embedder = LocalEmbedder(model_name=model_name)
        print(f"Using sentence-transformer embeddings with model: {model_name}")
    
    # Process in batches to show progress
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        try:
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            
            # Safety check - ensure all are strings
            batch_texts = [str(text) if text is not None else "" for text in batch_texts]
            
            # Skip any empty batches
            if not batch_texts or all(not text for text in batch_texts):
                print(f"Skipping empty batch at index {i}")
                continue
            
            # Generate embeddings for batch
            if embedding_type == 'openai':
                batch_embeddings = [embedder.embed_query(text) for text in batch_texts]
            else:
                batch_embeddings = embedder(batch_texts)
            
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            print(f"Batch texts: {batch_texts}")
            # Continue with next batch instead of failing completely
            continue
    
    # Convert to numpy array for clustering
    embeddings_array = np.array(embeddings)
    
    # Normalize embeddings to unit length if requested
    if normalize_embeddings:
        print("Normalizing embeddings to unit length...")
        # Compute L2 norm (Euclidean norm) of each embedding vector
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms[norms == 0] = 1.0
        # Normalize by dividing by the L2 norm
        embeddings_array = embeddings_array / norms
        print("Embeddings normalized successfully")
    
    print(f"Generated embeddings with shape: {embeddings_array.shape}")
    return embeddings_array, product_codes

def save_embeddings(embeddings: np.ndarray, 
                   product_codes: List[str],
                   output_dir: str = None) -> Tuple[str, str]:
    """
    Save embeddings and product codes to files.
    
    Args:
        embeddings: NumPy array of embeddings
        product_codes: List of product codes
        output_dir: Directory to save files (defaults to data directory)
        
    Returns:
        Tuple of (embeddings_path, product_codes_path)
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "data"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings as numpy array
    embeddings_path = os.path.join(output_dir, "product_embeddings.npy")
    np.save(embeddings_path, embeddings)
    
    # Save product codes as text file
    product_codes_path = os.path.join(output_dir, "product_codes.txt")
    with open(product_codes_path, 'w') as f:
        for code in product_codes:
            f.write(f"{code}\n")
    
    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved product codes to {product_codes_path}")
    
    return embeddings_path, product_codes_path

if __name__ == "__main__":
    # Define output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data"
    )
    
    # Load prepared data 
    prepared_data_path = os.path.join(output_dir, "prepared_products.csv")
    if os.path.exists(prepared_data_path):
        print(f"Loading prepared data from {prepared_data_path}")
        prepared_data = pd.read_csv(prepared_data_path)
    else:
        print("Prepared data not found. Running data preparation...")
        from product_clustering.data_prep import prepare_data_for_clustering
        prepared_data = prepare_data_for_clustering()
        
        # Save prepared data
        prepared_data.to_csv(prepared_data_path, index=False)
        print(f"Saved prepared data to {prepared_data_path}")
    
    # Generate embeddings
    embeddings, product_codes = embed_products(
        df=prepared_data,
        text_col='clustering_description',
        embedding_type='sentence-transformer',  # Using sentence-transformer by default
        model_name='all-mpnet-base-v2'          # Using stronger model
    )
    
    # Save embeddings and product codes
    save_embeddings(embeddings, product_codes, output_dir)
    
    print(f"Feature engineering complete. Generated {len(embeddings)} embeddings.")
