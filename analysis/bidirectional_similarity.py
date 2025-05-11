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
    
    # In theory, forward and backward should be the same for proper cosine similarity
    # If they differ significantly, it suggests preprocessing asymmetry
    
    return forward_similarity, backward_similarity, avg_similarity

def find_similar_products_bidirectional(vector_db, query, product_descriptions, n_results=5, threshold=0.3):
    """
    Find similar products using bidirectional similarity approach
    
    Args:
        vector_db: ProductVectorDB instance
        query: Query string
        product_descriptions: List of product descriptions to search within
        n_results: Number of results to return
        threshold: Similarity threshold
        
    Returns:
        DataFrame of similar products with bidirectional similarity scores
    """
    # Get embedder from vector_db
    embedder = vector_db.embedding_function.embedding_function
    
    # Get query embedding
    query_embedding = embedder.embed_query(query)
    
    # Calculate bidirectional similarity for each product
    similarities = []
    for product in product_descriptions:
        forward, backward, average = calculate_bidirectional_similarity(embedder, query, product)
        similarities.append({
            'product_description': product,
            'forward_similarity': forward,
            'backward_similarity': backward,
            'bidirectional_similarity': average,
            # Use max for most generous interpretation
            'max_similarity': max(forward, backward)
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(similarities)
    
    # Filter by threshold
    results_df = results_df[results_df['max_similarity'] >= threshold]
    
    # Sort by bidirectional similarity
    results_df = results_df.sort_values('bidirectional_similarity', ascending=False).reset_index(drop=True)
    
    # Limit to n_results
    if len(results_df) > n_results:
        results_df = results_df.head(n_results)
    
    return results_df

def compare_approaches(problem_clusters, mapping_df):
    """
    Compare standard vs bidirectional similarity approaches on problematic clusters
    
    Args:
        problem_clusters: List of product clusters to test
        mapping_df: DataFrame with product mappings
        
    Returns:
        tuple: (standard_results, bidirectional_results)
    """
    print("\n=== Comparing Standard vs Bidirectional Similarity ===\n")
    
    # Load product data for searching
    vector_db = ProductVectorDB()
    collection = vector_db.client.get_collection("products")
    product_data = collection.get(include=["documents"])
    product_descriptions = product_data["documents"]
    
    standard_recall = {}
    bidirectional_recall = {}
    
    for cluster_name in problem_clusters:
        print(f"\n{'='*50}")
        print(f"CLUSTER: {cluster_name}")
        
        # Get variations for this cluster
        row = mapping_df[mapping_df["Standardized Product Name"] == cluster_name]
        if row.empty:
            continue
            
        variations = [desc.strip() for desc in row.iloc[0]["Possible Descriptions"].split(";")]
        print(f"Variations ({len(variations)}): {variations}")
        
        # TEST STANDARD APPROACH
        print("\nSTANDARD APPROACH:")
        std_results = find_similar_products(cluster_name, n_results=15, 
                                          similarity_threshold=0.25,
                                          vector_db=vector_db)
        
        # Count matches
        std_found = set()
        for _, res_row in std_results.iterrows():
            desc = res_row["product_description"]
            sim = res_row["similarity"]
            is_match = desc.lower() in [v.lower() for v in variations]
            print(f"  {'✓' if is_match else '✗'} {desc} (sim: {sim:.4f})")
            if is_match:
                std_found.add(desc.lower())
        
        std_recall_val = len(std_found) / len(variations) if variations else 0
        standard_recall[cluster_name] = std_recall_val
        print(f"  Standard Recall: {std_recall_val:.2f} ({len(std_found)}/{len(variations)})")
        
        # TEST BIDIRECTIONAL APPROACH
        print("\nBIDIRECTIONAL APPROACH:")
        bi_results = find_similar_products_bidirectional(
            vector_db, cluster_name, product_descriptions, 
            n_results=15, threshold=0.25
        )
        
        # Count matches
        bi_found = set()
        for _, res_row in bi_results.iterrows():
            desc = res_row["product_description"]
            sim = res_row["bidirectional_similarity"]
            max_sim = res_row["max_similarity"]
            is_match = desc.lower() in [v.lower() for v in variations]
            print(f"  {'✓' if is_match else '✗'} {desc} (bi_sim: {sim:.4f}, max: {max_sim:.4f})")
            if is_match:
                bi_found.add(desc.lower())
        
        bi_recall_val = len(bi_found) / len(variations) if variations else 0
        bidirectional_recall[cluster_name] = bi_recall_val
        print(f"  Bidirectional Recall: {bi_recall_val:.2f} ({len(bi_found)}/{len(variations)})")
        
        # Show improvement
        if bi_recall_val > std_recall_val:
            print(f"  ▲ Improvement: +{(bi_recall_val-std_recall_val)*100:.1f}%")
        elif bi_recall_val < std_recall_val:
            print(f"  ▼ Decrease: -{(std_recall_val-bi_recall_val)*100:.1f}%")
        else:
            print(f"  ◆ No change in recall")
            
    return standard_recall, bidirectional_recall

def test_bidirectional_similarity():
    """Test bidirectional similarity on problem clusters"""
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
    
    # Load the mapping file
    mapping_path = Path(__file__).parent.parent / "data/CorrectMapping/product_mapping_semantic.xlsx"
    mapping_df = pd.read_excel(mapping_path)
    
    # Problem clusters to analyze
    problem_clusters = [
        "Organic Chicken Breast",
        "Frozen Jumbo Shrimp",
        "Brown Rice Long Grain",
        "Orange Juice Fresh Squeezed"
    ]
    
    # Compare approaches
    compare_approaches(problem_clusters, mapping_df)
