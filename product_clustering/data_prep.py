"""
Data preparation module for product clustering.
This module extends the existing data processing pipeline to prepare product data
for clustering based on similarity rather than USDA mapping.
"""
import os
import sys
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_processing import load_transaction_data, clean_text
from src.abbreviation_translator import expand_abbreviations

def preprocess_text_for_clustering(text: str) -> str:
    """
    Enhanced preprocessing function optimized for product clustering.
    Normalizes text by expanding abbreviations, removing special characters, 
    and standardizing format.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text optimized for clustering
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Expand abbreviations using existing function
    text = expand_abbreviations(text)
    
    # Remove special characters but keep numbers (unlike USDA matching)
    # Numbers are important for package sizes and weights
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Standardize white space
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text



def prepare_data_for_clustering(df_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Prepare transaction data for product clustering.
    
    Args:
        df_raw: Optional raw transaction DataFrame. If None, data will be loaded
                from the default location specified in config.
                
    Returns:
        DataFrame with products prepared for clustering
    """
    print("Preparing data for product clustering...")
    
    # Load data if not provided
    if df_raw is None:
        df_raw = load_transaction_data()
        if df_raw is None:
            print("Error: Failed to load transaction data")
            return pd.DataFrame()
    
    # Process transaction data using existing pipeline
    # This gives us unique product descriptions and their associated product codes
    from src.data_processing import process_transaction_data
    unique_products_df = process_transaction_data(df_raw)
    
    if unique_products_df is None or len(unique_products_df) == 0:
        print("Error: No products found after processing")
        return pd.DataFrame()
    
    print(f"Found {len(unique_products_df)} unique products.")
    
    # Clean the product descriptions for clustering
    unique_products_df['clustering_description'] = unique_products_df['product_description'].apply(preprocess_text_for_clustering)
    
    # Group products by category and save the mapping
    from src.data_processing import group_products_by_category, save_category_products
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGrouping products by category...")
    category_groups = group_products_by_category(unique_products_df)
    
    # Save category products mapping for clustering
    category_mapping_path = save_category_products(category_groups, output_dir)
    if category_mapping_path:
        print(f"Saved category-to-products mapping to {category_mapping_path}")
    
    print("Data preparation complete.")
    print(f"Prepared {len(unique_products_df)} products for clustering.")
    
    # Display sample of prepared data
    print("\nSample of prepared data:")
    sample_cols = ['product_code', 'product_description', 'product_category', 'clustering_description']
    print(unique_products_df[sample_cols[:3]].head().to_string())
    
    return unique_products_df

if __name__ == "__main__":
    # Test data preparation
    prepared_data = prepare_data_for_clustering()
    
    # Save to CSV for inspection if needed
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prepared_products.csv")
    
    prepared_data.to_csv(output_path, index=False)
    print(f"Saved prepared data to {output_path}")
