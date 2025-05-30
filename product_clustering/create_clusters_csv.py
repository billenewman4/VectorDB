#!/usr/bin/env python3
"""
Convert refined clusters from JSON to CSV format for easier analysis
"""

import json
import csv
import os
import pandas as pd
from typing import Dict, List

# Import the abbreviation expansion function directly from source
from src.abbreviation_translator import expand_abbreviations

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

def extract_descriptive_name(cluster_id: str) -> str:
    """
    Extract the descriptive name from a cluster ID like 'cluster_Chicken Sausage_1175'
    
    Args:
        cluster_id: The cluster ID string
    
    Returns:
        The descriptive name (or the original ID if no descriptive name is found)
    """
    parts = cluster_id.split('_')
    
    if len(parts) >= 3:
        # For cluster IDs in the format 'cluster_Category_Number' or 'cluster_Category_Word_Number'
        descriptive_parts = parts[1:-1]  # All parts except 'cluster' and the number at the end
        return ' '.join(descriptive_parts)
    elif len(parts) == 2:
        # For cluster IDs in the format 'cluster_Number'
        return f"Cluster {parts[1]}"
    else:
        # Default fallback
        return cluster_id


def create_cluster_products_csv(clusters: Dict[str, List[str]], products_csv_path: str, output_csv_path: str):
    """
    Create a CSV file with cluster ID, product code, product description, and descriptive cluster name
    
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
    expanded_descriptions = {}
    for _, row in products_df.iterrows():
        if 'product_code' in row and 'product_description' in row:
            code = str(row['product_code'])
            description = row['product_description']
            product_descriptions[code] = description
            
            # Use the existing abbreviation expansion function directly
            expanded = expand_abbreviations(description)
            expanded_descriptions[code] = expanded
    
    print(f"Loaded {len(product_descriptions)} product descriptions with expanded abbreviations")
    
    # Extract descriptive names directly from cluster IDs (which already have them)
    cluster_names = {}
    for cluster_id in clusters.keys():
        descriptive_name = extract_descriptive_name(cluster_id)
        cluster_names[cluster_id] = descriptive_name
    
    print(f"Extracted descriptive names for {len(cluster_names)} clusters")
    
    # Create CSV with descriptions and cluster names
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster_id', 'descriptive_name', 'product_code', 'product_description', 'expanded_description'])
        
        for cluster_id, product_codes in clusters.items():
            descriptive_name = cluster_names.get(cluster_id, cluster_id)
            for product_code in product_codes:
                product_code_str = str(product_code)
                description = product_descriptions.get(product_code_str, "")
                expanded = expanded_descriptions.get(product_code_str, "")
                writer.writerow([cluster_id, descriptive_name, product_code, description, expanded])
    
    print(f"CSV file with descriptions and descriptive names saved to {output_csv_path}")
    
    # Also create a summary CSV with descriptive names
    summary_csv_path = output_csv_path.replace('.csv', '_summary.csv')
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster_id', 'descriptive_name', 'product_count'])
        
        # Sort clusters by size (largest first)
        sorted_clusters = sorted([(cid, len(products)) for cid, products in clusters.items()], 
                                 key=lambda x: x[1], reverse=True)
        
        for cluster_id, size in sorted_clusters:
            descriptive_name = cluster_names.get(cluster_id, cluster_id)
            writer.writerow([cluster_id, descriptive_name, size])
    
    print(f"Cluster summary CSV with descriptive names saved to {summary_csv_path}")

if __name__ == "__main__":
    # Set paths using absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "product_clustering", "data", "refined_clusters", "refined_clusters.json")
    csv_path = os.path.join(project_root, "product_clustering", "data", "refined_clusters", "refined_clusters.csv")
    
    # Convert clusters to CSV
    convert_clusters_to_csv(json_path, csv_path)
