#!/usr/bin/env python3
"""
Run bidirectional similarity analysis with proper database initialization
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.vectordb import ProductVectorDB, create_product_vector_db
from analysis.bidirectional_similarity import (
    calculate_bidirectional_similarity,
    find_similar_products_bidirectional,
    test_bidirectional_similarity,
    implement_bidirectional_solution
)

def run_complete_analysis():
    """
    Run the complete bidirectional similarity analysis with database preparation
    """
    # First create the vector database (or use existing one)
    print("\n=== Creating Vector Database ===")
    vector_db = create_product_vector_db(recreate=False)
    print("Vector database ready\n")
    
    # Load the product data for our bidirectional analysis
    collection = vector_db.client.get_collection("products")
    product_data = collection.get(include=["documents"])
    product_descriptions = product_data["documents"]
    
    # Load the mapping file for ground truth
    mapping_path = Path(__file__).parent.parent / "data/CorrectMapping/product_mapping_semantic.xlsx"
    mapping_df = pd.read_excel(mapping_path)
    
    # Load all clusters from the mapping file
    all_clusters = mapping_df["Standardized Product Name"].unique().tolist()
    
    # Identify known problem clusters (for reference)
    known_problem_clusters = {
        "Organic Chicken Breast",
        "Frozen Jumbo Shrimp", 
        "Brown Rice Long Grain",
        "Orange Juice Fresh Squeezed"
    }
    
    print("\n=== Comparing Standard vs Bidirectional Similarity ===\n")
    
    standard_recall = {}
    bidirectional_recall = {}
    
    # Process all clusters
    for cluster_name in all_clusters:
        # Mark if this is a known problematic cluster
        is_problem_cluster = cluster_name in known_problem_clusters
        print(f"\n{'='*50}")
        if is_problem_cluster:
            print(f"CLUSTER: {cluster_name} 🔍 (Known problem cluster)")
        else:
            print(f"CLUSTER: {cluster_name}")
        
        # Get variations for this cluster
        row = mapping_df[mapping_df["Standardized Product Name"] == cluster_name]
        if row.empty:
            continue
            
        variations = [desc.strip() for desc in row.iloc[0]["Possible Descriptions"].split(";")]
        print(f"Variations ({len(variations)}): {variations}")
        
        # Standard approach (current implementation)
        print("\nSTANDARD APPROACH:")
        
        from src.vectordb import find_similar_products
        std_results = find_similar_products(
            cluster_name, 
            n_results=15,
            similarity_threshold=0.25,
            vector_db=vector_db
        )
        
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
        
        # Bidirectional approach (our new method)
        print("\nBIDIRECTIONAL APPROACH:")
        
        # Get embedder directly - SentenceTransformer model for embeddings
        from sentence_transformers import SentenceTransformer
        
        # Create a wrapper class that provides the embed_query interface
        class EmbedderWrapper:
            def __init__(self, model):
                self.model = model
                
            def embed_query(self, text):
                # SentenceTransformer uses encode() instead of embed_query()
                return self.model.encode(text, convert_to_numpy=True)
        
        # Create the wrapper
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = EmbedderWrapper(model)
        
        # Calculate bidirectional similarity for each product
        similarities = []
        for product in product_descriptions:
            forward, backward, average = calculate_bidirectional_similarity(
                embedder, cluster_name, product
            )
            similarities.append({
                'product_description': product,
                'forward_similarity': forward,
                'backward_similarity': backward,
                'bidirectional_similarity': average,
                'max_similarity': max(forward, backward)
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(similarities)
        
        # Filter by threshold
        bi_results = results_df[results_df['max_similarity'] >= 0.25]
        
        # Sort by bidirectional similarity
        bi_results = bi_results.sort_values('bidirectional_similarity', ascending=False).reset_index(drop=True)
        
        # Limit to n_results
        if len(bi_results) > 15:
            bi_results = bi_results.head(15)
        
        # Count matches
        bi_found = set()
        for _, res_row in bi_results.iterrows():
            desc = res_row["product_description"]
            sim = res_row["bidirectional_similarity"]
            max_sim = res_row["max_similarity"]
            
            # Extract the base description without price and quantity info
            base_desc = desc.split(", average price")[0] if ", average price" in desc else desc
            
            # Check if any variation matches the base description
            is_match = any(v.lower() == base_desc.lower() for v in variations)
            
            print(f"  {'✓' if is_match else '✗'} {desc} (bi_sim: {sim:.4f}, max: {max_sim:.4f})")
            if is_match:
                bi_found.add(base_desc.lower())
        
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
    
    # Print overall summary
    print("\n=== OVERALL SUMMARY ===\n")
    std_avg = sum(standard_recall.values()) / len(standard_recall) if standard_recall else 0
    bi_avg = sum(bidirectional_recall.values()) / len(bidirectional_recall) if bidirectional_recall else 0
    print(f"Standard approach average recall: {std_avg:.3f} across {len(standard_recall)} clusters")
    print(f"Bidirectional approach average recall: {bi_avg:.3f} across {len(bidirectional_recall)} clusters")
    
    if bi_avg > std_avg:
        print(f"▲ Overall improvement: +{(bi_avg-std_avg)*100:.1f}%")
        
    # Detailed analysis by cluster type
    problem_std_recalls = {k: v for k, v in standard_recall.items() if k in known_problem_clusters}
    problem_bi_recalls = {k: v for k, v in bidirectional_recall.items() if k in known_problem_clusters}
    
    other_std_recalls = {k: v for k, v in standard_recall.items() if k not in known_problem_clusters}
    other_bi_recalls = {k: v for k, v in bidirectional_recall.items() if k not in known_problem_clusters}
    
    # Calculate averages for problem clusters
    prob_std_avg = sum(problem_std_recalls.values()) / len(problem_std_recalls) if problem_std_recalls else 0
    prob_bi_avg = sum(problem_bi_recalls.values()) / len(problem_bi_recalls) if problem_bi_recalls else 0
    
    # Calculate averages for other clusters
    other_std_avg = sum(other_std_recalls.values()) / len(other_std_recalls) if other_std_recalls else 0
    other_bi_avg = sum(other_bi_recalls.values()) / len(other_bi_recalls) if other_bi_recalls else 0
    
    print("\n=== CLUSTER TYPE ANALYSIS ===\n")
    print(f"Problem Clusters: Standard {prob_std_avg:.3f} vs. Bidirectional {prob_bi_avg:.3f}")
    print(f"Other Clusters: Standard {other_std_avg:.3f} vs. Bidirectional {other_bi_avg:.3f}")
    
    # Show implementation solution
    implement_bidirectional_solution()

if __name__ == "__main__":
    run_complete_analysis()
