"""
Test script for the Vector Embedding module.

This script demonstrates how to use the various components of the Vector Embedding module
and verifies that they work correctly together.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

# Import embedding module components using relative imports
from Models.base_embedder import BaseEmbedder, is_embedder
from Models.local_embedder import LocalEmbedder
from Utilities.normalization import normalize_vectors, check_normalized
from Utilities.batch_processor import process_in_batches
from pipeline import create_embedder, embed_products, generate_embeddings


def test_local_embedder():
    """Test the local embedder with sentence-transformers."""
    print("\n=== Testing Local Embedder ===")
    
    # Create local embedder
    embedder = LocalEmbedder(model_name="all-mpnet-base-v2")
    print(f"Created embedder: {embedder.name()}")
    
    # Test is_embedder utility
    assert is_embedder(embedder), "LocalEmbedder should implement the embedder interface"
    
    # Generate single embedding
    text = "Fresh organic apple"
    embedding = embedder.embed_query(text)
    print(f"Single embedding shape: {embedding.shape}")
    
    # Generate multiple embeddings
    texts = ["Fresh organic apple", "Ripe yellow banana", "Juicy orange"]
    embeddings = embedder(texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding shape: {embeddings[0].shape}")
    
    # Check normalization
    normalized = normalize_vectors(embeddings)
    assert check_normalized(normalized), "Embeddings should be normalized to unit length"
    print("Embedding normalization successful")
    
    print("Local embedder tests passed!")


def test_batch_processing():
    """Test batch processing of embeddings."""
    print("\n=== Testing Batch Processing ===")
    
    # Create sample data
    texts = [f"Sample text {i}" for i in range(120)]
    
    # Create embedder
    embedder = LocalEmbedder()
    
    # Process in batches
    results = process_in_batches(
        items=texts,
        processor_func=embedder,
        batch_size=32,
        show_progress=True
    )
    
    print(f"Processed {len(results)} items in batches")
    print(f"First result shape: {results[0].shape}")
    
    print("Batch processing tests passed!")


def test_pipeline():
    """Test the complete embedding pipeline."""
    print("\n=== Testing Complete Pipeline ===")
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'product_code': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'product_name': ['Apple', 'Banana', 'Orange', 'Grape', 'Watermelon'],
        'clustering_description': [
            'Fresh red apple from Washington',
            'Organic yellow banana',
            'Juicy orange citrus fruit',
            'Purple grape bunch seedless',
            'Large watermelon with seeds'
        ]
    })
    
    print("Sample DataFrame:")
    print(df[['product_code', 'product_name']])
    
    # Test embed_products function
    print("\nTesting embed_products...")
    embeddings, codes = embed_products(
        df=df,
        text_col='clustering_description',
        embedding_type='sentence-transformer',
        batch_size=2,
        normalize_embeddings=True
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Product codes: {codes}")
    
    # Test generate_embeddings function
    print("\nTesting generate_embeddings...")
    embedder = create_embedder(embedding_type='sentence-transformer')
    df_with_embeddings = generate_embeddings(
        embedder=embedder,
        df=df,
        text_col='clustering_description',
        batch_size=2
    )
    
    print(f"DataFrame with embeddings shape: {df_with_embeddings.shape}")
    print(f"Embedding column data type: {type(df_with_embeddings['embedding'].iloc[0])}")
    print(f"First embedding shape: {df_with_embeddings['embedding'].iloc[0].shape}")
    
    print("Pipeline tests passed!")


if __name__ == "__main__":
    try:
        print("===== Vector Embedding Module Tests =====")
        
        # Run individual tests
        test_local_embedder()
        test_batch_processing()
        test_pipeline()
        
        print("\n===== All Tests Passed! =====")
        print("The Vector Embedding module is functioning correctly.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTests failed. Please check the error message above.")
