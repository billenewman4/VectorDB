#!/usr/bin/env python3
"""
Count Unclustered Products

A simple script to count how many products are in clusters vs. total products.
"""

import os
import sys
import json
from collections import Counter

# Add parent directory to path to access source modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Load the refined clusters
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    refined_clusters_path = os.path.join(data_dir, "refined_clusters", "refined_clusters.json")
    
    if not os.path.exists(refined_clusters_path):
        print(f"Error: Refined clusters file not found at {refined_clusters_path}")
        sys.exit(1)
    
    with open(refined_clusters_path, 'r') as f:
        clusters = json.load(f)
    
    # Count products in clusters
    products_in_clusters = set()
    cluster_sizes = Counter()
    
    for cluster_id, product_codes in clusters.items():
        cluster_sizes[len(product_codes)] += 1
        for code in product_codes:
            products_in_clusters.add(code)
    
    print(f"Total clusters: {len(clusters)}")
    print(f"Unique products in clusters: {len(products_in_clusters)}")
    
    # Load transaction data to get total unique products
    try:
        from src.data_processing import load_transaction_data
        
        df = load_transaction_data()
        if df is not None:
            # Count unique product codes
            unique_product_codes = set()
            for row in df:
                if "ProductCode" in row and row["ProductCode"]:
                    unique_product_codes.add(str(row["ProductCode"]))
                elif "product_code" in row and row["product_code"]:
                    unique_product_codes.add(str(row["product_code"]))
            
            print(f"Total unique products in transaction data: {len(unique_product_codes)}")
            
            # Calculate unclustered products
            unclustered_products = unique_product_codes - products_in_clusters
            print(f"Products not in any cluster: {len(unclustered_products)} ({len(unclustered_products)/len(unique_product_codes):.2%})")
            
            # Print a few unclustered products
            print("\nSample of unclustered products:")
            sample = list(unclustered_products)[:10]
            for code in sample:
                print(f"  {code}")
    except Exception as e:
        print(f"Error accessing transaction data: {str(e)}")
        print("Using alternative method to estimate total products...")
        
        # Try to find any product-related files that might contain counts
        try:
            # Check for product embeddings or cluster assignments
            embedding_files = []
            for root, dirs, files in os.walk(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
                for file in files:
                    if "product" in file.lower() and (file.endswith(".npy") or file.endswith(".txt")):
                        embedding_files.append(os.path.join(root, file))
            
            print(f"Found {len(embedding_files)} potential product files:")
            for file in embedding_files:
                print(f"  {file}")
                
            # Try to estimate from improved_clustering.py source code
            clustering_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "improved_clustering.py"
            )
            
            if os.path.exists(clustering_file):
                with open(clustering_file, 'r') as f:
                    content = f.read()
                    if "min_cluster_size" in content and "min_samples" in content:
                        print("\nClustering parameters from improved_clustering.py:")
                        for line in content.split("\n"):
                            if ("min_cluster_size" in line or "min_samples" in line) and "=" in line:
                                print(f"  {line.strip()}")
        except Exception as inner_e:
            print(f"Error in alternative method: {str(inner_e)}")

    # Print cluster size distribution
    print("\nCluster size distribution:")
    size_ranges = [(1, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    
    for low, high in size_ranges:
        range_str = f"{low}-{int(high) if high != float('inf') else '∞'}"
        if high == float('inf'):
            count = sum(cluster_sizes[size] for size in cluster_sizes if size >= low)
        else:
            count = sum(cluster_sizes[size] for size in range(low, int(high) + 1) if size in cluster_sizes)
        pct = count / len(clusters) * 100
        print(f"  {range_str}: {count} clusters ({pct:.1f}%)")

if __name__ == "__main__":
    main()
