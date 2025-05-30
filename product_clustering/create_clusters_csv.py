#!/usr/bin/env python3
"""
Convert refined clusters from JSON to CSV format for easier analysis
"""

import json
import csv
import os
import pandas as pd
from typing import Dict, List

def convert_clusters_to_csv(json_path: str, csv_path: str):
    """
    Convert clusters from JSON format to CSV format
    
    Args:
        json_path: Path to the JSON file containing clusters
        csv_path: Path to save the CSV file
    """
    print(f"Loading clusters from {json_path}...")
    
    # Load clusters from JSON
    with open(json_path, 'r') as f:
        clusters = json.load(f)
    
    print(f"Loaded {len(clusters)} clusters with {sum(len(products) for products in clusters.values())} products")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Convert to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster_id', 'product_code'])
        
        for cluster_id, product_codes in clusters.items():
            for product_code in product_codes:
                writer.writerow([cluster_id, product_code])
    
    print(f"CSV file saved to {csv_path}")
    
    # Also create a cluster summary CSV with cluster sizes
    cluster_sizes = {cluster_id: len(products) for cluster_id, products in clusters.items()}
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    summary_csv_path = csv_path.replace('.csv', '_summary.csv')
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster_id', 'product_count'])
        
        for cluster_id, size in sorted_clusters:
            writer.writerow([cluster_id, size])
    
    print(f"Cluster summary CSV saved to {summary_csv_path}")
    
    # If products data is available, try to add product descriptions
    products_csv_path = os.path.join(os.path.dirname(os.path.dirname(json_path)), 'prepared_products.csv')
    if os.path.exists(products_csv_path):
        create_cluster_products_csv(clusters, products_csv_path, csv_path.replace('.csv', '_with_descriptions.csv'))

def create_cluster_products_csv(clusters: Dict[str, List[str]], products_csv_path: str, output_csv_path: str):
    """
    Create a CSV file with cluster ID, product code, and product description
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        products_csv_path: Path to the prepared products CSV file
        output_csv_path: Path to save the output CSV file
    """
    print(f"Loading product data from {products_csv_path}...")
    
    # Load products data
    products_df = pd.read_csv(products_csv_path)
    
    # Create a lookup for product descriptions
    product_descriptions = {}
    for _, row in products_df.iterrows():
        if 'product_code' in row and 'product_description' in row:
            product_descriptions[str(row['product_code'])] = row['product_description']
    
    print(f"Loaded {len(product_descriptions)} product descriptions")
    
    # Create CSV with descriptions
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster_id', 'product_code', 'product_description'])
        
        for cluster_id, product_codes in clusters.items():
            for product_code in product_codes:
                description = product_descriptions.get(str(product_code), "")
                writer.writerow([cluster_id, product_code, description])
    
    print(f"CSV file with descriptions saved to {output_csv_path}")

if __name__ == "__main__":
    # Set paths using absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "product_clustering", "data", "refined_clusters", "refined_clusters.json")
    csv_path = os.path.join(project_root, "product_clustering", "data", "refined_clusters", "refined_clusters.csv")
    
    # Convert clusters to CSV
    convert_clusters_to_csv(json_path, csv_path)
