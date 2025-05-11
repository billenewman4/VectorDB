#!/usr/bin/env python3
"""
Test bidirectional similarity for product clusters
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vectordb import ProductVectorDB, ProductEmbedder, find_similar_products

def calculate_bidirectional_similarity(embedder, text1, text2):
    """
    Calculate similarity in both directions and return the average
    
    Args:
        embedder: ProductEmbedder instance
        text1: First text string
        text2: Second text string
        
    Returns:
        tuple: (forward_similarity, backward_similarity, average_similarity)
    """
    # Get embeddings
    embedding1 = embedder.embed_query(text1)
    embedding2 = embedder.embed_query(text2)
    
    # Calculate cosine similarity in both directions
    # (In a proper cosine similarity, direction shouldn't matter, but our embedding function might do 
    # additional preprocessing that creates asymmetry)
    forward_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    backward_similarity = np.dot(embedding2, embedding1) / (np.linalg.norm(embedding2) * np.linalg.norm(embedding1))
    
    # Average the similarities
    avg_similarity = (forward_similarity + backward_similarity) / 2
    
    return forward_similarity, backward_similarity, avg_similarity

def test_bidirectional_similarity():
    """
    Test bidirectional similarity on problem clusters
    """
    print("\n=== Testing Bidirectional Similarity ===")
    
    # Initialize embedder
    embedder = ProductEmbedder()
    
    # Load the mapping file
    mapping_path = Path(__file__).parent.parent / "data/CorrectMapping/product_mapping_semantic.xlsx"
    if not mapping_path.exists():
        print(f"❌ Mapping file not found at {mapping_path}")
        return
    
    # Read the mapping file
    mapping_df = pd.read_excel(mapping_path)
    
    # Problem clusters we want to analyze
    problem_clusters = [
        "Organic Chicken Breast",
        "Frozen Jumbo Shrimp",
        "Brown Rice Long Grain"
    ]
    
    # Test each problem cluster
    for cluster_name in problem_clusters:
        print(f"\n{'='*50}")
        print(f"CLUSTER: {cluster_name}")
        
        # Get the row for this cluster
        row = mapping_df[mapping_df["Standardized Product Name"] == cluster_name]
        if row.empty:
            print(f"No data found for {cluster_name}")
            continue
            
        # Get possible descriptions
        possible_descriptions_text = row.iloc[0]["Possible Descriptions"]
        variations = [desc.strip() for desc in possible_descriptions_text.split(";")]
        
        print(f"Standard name: {cluster_name}")
        print(f"Variations ({len(variations)}): {', '.join(variations)}")
        
        # Test similarity in both directions
        print("\nBidirectional Similarity Analysis:")
        print(f"{'Variation':<30} | {'Forward':<10} | {'Backward':<10} | {'Average':<10} | {'Diff':<10}")
        print(f"{'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        
        for variation in variations:
            # Skip identical terms
            if variation.lower() == cluster_name.lower():
                continue
                
            # Calculate bidirectional similarity
            forward, backward, average = calculate_bidirectional_similarity(
                embedder, cluster_name, variation
            )
            
            # Calculate absolute difference
            difference = abs(forward - backward)
            
            # Print results
            print(f"{variation:<30} | {forward:10.4f} | {backward:10.4f} | {average:10.4f} | {difference:10.4f}")
            
            # Print a warning for high asymmetry
            if difference > 0.1:
                print(f"⚠️  High asymmetry detected for {variation}")
    
    print("\n✅ Bidirectional similarity test completed")

if __name__ == "__main__":
    test_bidirectional_similarity()
