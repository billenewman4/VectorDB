#!/usr/bin/env python3
"""
Create Simple USDA Mapping File

This script creates a simple USDA mapping file using actual product codes from the dataset
without relying on category clustering. It groups products based on text similarity in descriptions.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_products_data(products_path: str) -> pd.DataFrame:
    """Load the prepared products data."""
    try:
        return pd.read_csv(products_path)
    except Exception as e:
        print(f"Error loading products data: {e}")
        return pd.DataFrame()

def preprocess_description(text):
    """Clean and normalize product descriptions."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_simple_mapping(
    products_path: str,
    output_path: str,
    min_group_size: int = 3,
    max_groups: int = 50,
    similarity_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Create a simple USDA mapping file by grouping similar products.
    
    Args:
        products_path: Path to the prepared products CSV file
        output_path: Path to save the output Excel file
        min_group_size: Minimum number of products per group
        max_groups: Maximum number of standardized names to create
        similarity_threshold: Minimum similarity for products to be grouped
    """
    # Load product data
    products_df = load_products_data(products_path)
    if products_df.empty:
        print("Error: No product data loaded")
        return pd.DataFrame()
    
    # Preprocess descriptions
    products_df['processed_description'] = products_df['product_description'].apply(preprocess_description)
    
    # Filter out products with empty descriptions
    products_df = products_df[products_df['processed_description'].str.len() > 3].reset_index(drop=True)
    
    # Get unique words in descriptions to identify product types
    all_words = set()
    for desc in products_df['processed_description']:
        all_words.update(desc.split())
    
    # Create product groups based on common keywords
    common_words = [word for word in all_words if len(word) > 3]
    groups = []
    
    # Use TF-IDF for better grouping
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(products_df['processed_description'])
    
    # Sample products to create seed groups
    sample_size = min(500, len(products_df))
    np.random.seed(42)
    sample_indices = np.random.choice(len(products_df), sample_size, replace=False)
    
    # Create groups around the sampled products
    created_groups = 0
    assigned_products = set()
    
    for idx in sample_indices:
        if created_groups >= max_groups:
            break
            
        if idx in assigned_products:
            continue
            
        # Find similar products
        product_vector = tfidf_matrix[idx]
        similarities = cosine_similarity(product_vector, tfidf_matrix).flatten()
        
        # Get indices of similar products
        similar_indices = np.where(similarities >= similarity_threshold)[0]
        
        # If we have enough similar products, create a group
        if len(similar_indices) >= min_group_size:
            # Get product codes and descriptions
            product_codes = [str(products_df.iloc[i]['product_code']) for i in similar_indices]
            descriptions = [products_df.iloc[i]['product_description'] for i in similar_indices]
            
            # Create a standardized name based on the most common words
            base_description = products_df.iloc[idx]['product_description']
            words = base_description.split()
            std_name = " ".join(words[:3]).title() if len(words) >= 3 else base_description.title()
            
            # Add to groups
            groups.append({
                "Standardized Product Name": std_name,
                "Product Codes": ";".join(product_codes),
                "Possible Descriptions": ";".join(descriptions[:5])  # Limit to 5 descriptions
            })
            
            # Mark products as assigned
            assigned_products.update(similar_indices)
            created_groups += 1
    
    # Create DataFrame and save to Excel
    mapping_df = pd.DataFrame(groups)
    
    # Save the mapping file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mapping_df.to_excel(output_path, index=False)
    
    print(f"Created simple USDA mapping with {len(mapping_df)} standardized names")
    print(f"Saved to {output_path}")
    
    return mapping_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create simple USDA mapping file")
    
    # Data paths
    parser.add_argument("--products_path", type=str, required=True, help="Path to prepared products CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output Excel file")
    parser.add_argument("--min_group_size", type=int, default=3, help="Minimum number of products per group")
    parser.add_argument("--max_groups", type=int, default=50, help="Maximum number of standardized names to create")
    parser.add_argument("--similarity_threshold", type=float, default=0.6, help="Similarity threshold for grouping")
    
    args = parser.parse_args()
    
    create_simple_mapping(
        products_path=args.products_path,
        output_path=args.output_path,
        min_group_size=args.min_group_size,
        max_groups=args.max_groups,
        similarity_threshold=args.similarity_threshold
    )

if __name__ == "__main__":
    main()
