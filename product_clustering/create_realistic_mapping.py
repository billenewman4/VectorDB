#!/usr/bin/env python3
"""
Create Realistic USDA Mapping File

This script creates a realistic USDA mapping file using actual product codes from the dataset.
It samples products from each category and assigns them to standardized product names.
"""

import os
import sys
import json
import pandas as pd
import random
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_category_products(category_products_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load the category-to-products mapping data.
    
    Args:
        category_products_path: Path to the category products JSON file
        
    Returns:
        Dictionary mapping categories to product code lists
    """
    try:
        with open(category_products_path, 'r') as f:
            category_products = json.load(f)
        return category_products
    except Exception as e:
        print(f"Error loading category products: {e}")
        return {}

def load_products_data(products_path: str) -> pd.DataFrame:
    """
    Load the prepared products data.
    
    Args:
        products_path: Path to the prepared products CSV file
        
    Returns:
        DataFrame containing product information
    """
    try:
        return pd.read_csv(products_path)
    except Exception as e:
        print(f"Error loading products data: {e}")
        return pd.DataFrame()

def create_realistic_mapping(
    category_products_path: str,
    products_path: str,
    output_path: str,
    min_products_per_category: int = 2,
    max_standardized_names: int = 100,
    products_per_name: int = 5
) -> None:
    """
    Create a realistic USDA mapping file using actual product codes from the dataset.
    
    Args:
        category_products_path: Path to the category products JSON file
        products_path: Path to the prepared products CSV file
        output_path: Path to save the output Excel file
        min_products_per_category: Minimum number of products a category must have
        max_standardized_names: Maximum number of standardized names to create
        products_per_name: Target number of products per standardized name
    """
    # Load the category products data
    category_products = load_category_products(category_products_path)
    
    # Load the prepared products data
    products_df = load_products_data(products_path)
    
    # Create a lookup for product descriptions
    product_descriptions = {}
    for _, row in products_df.iterrows():
        if 'product_code' in row and 'product_description' in row:
            product_descriptions[str(row['product_code'])] = row['product_description']
    
    # Create mapping for categories with sufficient products
    mapping_data = []
    standardized_names_count = 0
    
    # Shuffle categories to get a diverse sample
    categories = list(category_products.keys())
    random.shuffle(categories)
    
    for category in categories:
        category_data = category_products[category]
        
        # Get clustered products from this category
        if 'clustered' not in category_data:
            continue
            
        clustered_products = category_data['clustered']
        
        # Skip categories with too few products
        if len(clustered_products) < min_products_per_category:
            continue
        
        # Find clusters in this category
        clusters = defaultdict(list)
        
        # Need to determine which cluster each product belongs to
        # This requires loading the refined_category_clusters.json file
        refined_clusters_path = os.path.join(
            os.path.dirname(category_products_path),
            "refined",
            "refined_category_clusters.json"
        )
        
        if os.path.exists(refined_clusters_path):
            with open(refined_clusters_path, 'r') as f:
                all_clusters = json.load(f)
                
            # Map each product to its cluster
            for cluster_id, products in all_clusters.items():
                if not cluster_id.startswith(f"{category}_"):
                    continue
                    
                for product in products:
                    if product in clustered_products:
                        clusters[cluster_id].extend([product])
        
        # If we don't have cluster information, create fake clusters based on description similarity
        if not clusters:
            # Just group by first word of description as a fallback
            for product in clustered_products:
                if product in product_descriptions:
                    description = product_descriptions[product]
                    first_word = description.split()[0] if description else "Unknown"
                    clusters[f"{category}_{first_word}"].append(product)
        
        # Create standardized names from each cluster
        for cluster_id, products in clusters.items():
            if len(products) < min_products_per_category:
                continue
                
            # Create a standardized name based on the cluster
            # Use the most common words in product descriptions
            descriptions = [product_descriptions.get(p, "") for p in products if p in product_descriptions]
            if not descriptions:
                continue
                
            # Use the first product's description as a base for the standardized name
            sample_description = descriptions[0]
            words = sample_description.split()
            standardized_name = " ".join(words[:3]).title() if len(words) >= 3 else sample_description.title()
            
            # Sample products for this standardized name
            sample_size = min(products_per_name, len(products))
            sampled_products = random.sample(products, sample_size)
            
            # Create product codes string
            product_codes = ";".join(sampled_products)
            
            # Create possible descriptions string
            possible_descriptions = ";".join([product_descriptions.get(p, "") for p in sampled_products])
            
            # Add to mapping data
            mapping_data.append({
                "Standardized Product Name": standardized_name,
                "Product Codes": product_codes,
                "Possible Descriptions": possible_descriptions
            })
            
            standardized_names_count += 1
            if standardized_names_count >= max_standardized_names:
                break
        
        if standardized_names_count >= max_standardized_names:
            break
    
    # Create DataFrame and save to Excel
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_excel(output_path, index=False)
    
    print(f"Created realistic USDA mapping with {len(mapping_df)} standardized names")
    print(f"Saved to {output_path}")
    
    return mapping_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create realistic USDA mapping file")
    
    # Data paths
    parser.add_argument("--category_products_path", type=str, help="Path to category products JSON file")
    parser.add_argument("--products_path", type=str, help="Path to prepared products CSV file")
    parser.add_argument("--output_path", type=str, help="Path to save the output Excel file")
    parser.add_argument("--min_products", type=int, default=2, help="Minimum number of products per category")
    parser.add_argument("--max_names", type=int, default=100, help="Maximum number of standardized names to create")
    parser.add_argument("--products_per_name", type=int, default=5, help="Target number of products per standardized name")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not args.category_products_path:
        args.category_products_path = os.path.join(project_root, "product_clustering", "data", 
                                                 "category_clustering", "category_products.json")
    
    if not args.products_path:
        args.products_path = os.path.join(project_root, "product_clustering", "data", "prepared_products.csv")
    
    if not args.output_path:
        args.output_path = os.path.join(project_root, "data", "CorrectMapping", "product_mapping_semantic.xlsx")
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    create_realistic_mapping(
        category_products_path=args.category_products_path,
        products_path=args.products_path,
        output_path=args.output_path,
        min_products_per_category=args.min_products,
        max_standardized_names=args.max_names,
        products_per_name=args.products_per_name
    )

if __name__ == "__main__":
    main()
