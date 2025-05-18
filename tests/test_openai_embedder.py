#!/usr/bin/env python3
"""
Test script for OpenAIEmbedder class
"""
import os
import numpy as np
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the OpenAIEmbedder class
from src.VectorDB.OpenAIEmbedder import OpenAIEmbedder

def test_embedder():
    # Get API key from environment or .env file
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OpenAI API key not found in environment variables.")
        print("Make sure you have set OPENAI_API_KEY in your .env file or environment.")
        return

    print("Testing OpenAIEmbedder with a sample input...")
    
    # Create an instance of OpenAIEmbedder
    embedder = OpenAIEmbedder(api_key=api_key, model_name="text-embedding-3-small")
    
    # Sample text to embed
    sample_text = "Fresh beef sirloin steak, USDA Choice grade"
    
    # Test single text embedding
    print(f"\nEmbedding single text: '{sample_text}'")
    try:
        single_embedding = embedder.embed_query(sample_text)
        print(f"✅ Success! Embedding shape: {single_embedding.shape}")
        print(f"First few values: {single_embedding[:5]}")
    except Exception as e:
        print(f"❌ Error generating single embedding: {e}")
    
    # Test batch embedding
    sample_batch = [
        "Fresh beef sirloin steak, USDA Choice grade",
        "Organic chicken breast, boneless and skinless",
        "Atlantic salmon fillet, farm-raised"
    ]
    
    print(f"\nEmbedding batch of {len(sample_batch)} texts:")
    try:
        batch_embeddings = embedder(sample_batch)
        print(f"✅ Success! Got {len(batch_embeddings)} embeddings")
        for i, emb in enumerate(batch_embeddings):
            print(f"Embedding {i+1} shape: {emb.shape}")
    except Exception as e:
        print(f"❌ Error generating batch embeddings: {e}")

if __name__ == "__main__":
    test_embedder()
