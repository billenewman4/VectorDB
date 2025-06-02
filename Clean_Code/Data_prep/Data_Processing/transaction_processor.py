"""
Transaction data processing module.

This module provides functions for processing transaction data to extract 
unique product descriptions and prepare them for further analysis.
"""

import os
import sys
import pandas as pd
from typing import Dict, Optional, List, Any

# Add parent directories to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Use proper relative imports
from ..Text_Processing.text_normalization import clean_text
# Direct relative import for inventory loader
from ..Data_Loading.inventory_loader import load_inventory_data

# Config already defined above

# Define a simple config with default values for this module
class Config:
    """
    Configuration class with default values for data processing.
    """
    # Default column names
    PRODUCT_CODE_COLUMN = "product_code"
    PRODUCT_DESC_COLUMN = "product_desc"
    CATEGORY_COLUMN = "category"

config = Config()


def process_transaction_data(df_raw: Optional[pd.DataFrame] = None, 
                           code_col: Optional[str] = None, 
                           desc_col: Optional[str] = None,
                           filter_no_category: bool = True) -> pd.DataFrame:
    """
    Process transaction data to extract unique product descriptions.
    
    Args:
        df_raw: Transaction DataFrame. If None, data will be loaded.
        code_col: Column name for product codes. If None, uses config value.
        desc_col: Column name for product descriptions. If None, uses config value.
        filter_no_category: Whether to filter out products without category data.
            
    Returns:
        DataFrame with unique product descriptions and their codes.
    """
    # If no DataFrame is provided, we can't proceed
    if df_raw is None or df_raw.empty:
        print("Error: No transaction data provided for processing")
        return pd.DataFrame()
    
    # Set default column names from config if not provided
    if code_col is None:
        code_col = getattr(config, 'PRODUCT_CODE_COL', 'ProductCode')
    if desc_col is None:
        desc_col = getattr(config, 'PRODUCT_DESC_COL', 'ProductDescription')
    
    print(f"Processing transaction data with {len(df_raw)} rows")
    print(f"Using columns: '{code_col}' for product codes and '{desc_col}' for descriptions")
    
    # Check if required columns exist
    if code_col not in df_raw.columns:
        print(f"Error: Product code column '{code_col}' not found in data")
        return pd.DataFrame()
    if desc_col not in df_raw.columns:
        print(f"Error: Product description column '{desc_col}' not found in data")
        return pd.DataFrame()
    
    # Clean column names (remove spaces, standardize)
    df = df_raw.copy()
    
    # Step 1: Clean and standardize product codes and descriptions
    print("Cleaning product codes and descriptions...")
    
    # Clean product codes - avoiding normalization patterns that cause USDA code mapping issues
    # Note: NOT removing trailing numbers or first digits as these were causing mapping issues
    df['product_code'] = df[code_col].astype(str).apply(lambda x: x.strip())
    
    # Clean product descriptions
    df['product_description'] = df[desc_col].apply(clean_text)
    
    # Step 2: Get unique products (one row per product code)
    print("Extracting unique products...")
    # Group by product code and take the most common description
    unique_products = df.groupby('product_code')['product_description'].agg(
        lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else ""
    ).reset_index()
    
    # Step 3: Load product categories if needed
    if filter_no_category:
        print("Loading product categories...")
        product_categories = load_inventory_data()
        print(f"Found categories for {len(product_categories)} products")
        
        # Add category information
        unique_products['product_category'] = unique_products['product_code'].map(
            lambda x: product_categories.get(x, "Unknown")
        )
        
        # Filter out products without categories if requested
        initial_count = len(unique_products)
        unique_products = unique_products[unique_products['product_category'] != "Unknown"]
        filtered_count = initial_count - len(unique_products)
        print(f"Filtered out {filtered_count} products without category information")
    else:
        # Add placeholder category
        unique_products['product_category'] = "Not Categorized"
    
    print(f"Final processed dataset contains {len(unique_products)} unique products")
    
    return unique_products


if __name__ == "__main__":
    # For testing, we need to load transaction data first
    try:
        from Clean_Code.Data_prep.Data_Loading.transaction_loader import load_transaction_data
    except ImportError:
        try:
            from src.data_processing import load_transaction_data
        except ImportError:
            # Define simple fallback
            def load_transaction_data():
                print("Error: load_transaction_data function not available")
                return None
    
    # Test the transaction processing
    raw_data = load_transaction_data()
    
    if raw_data is not None:
        processed_data = process_transaction_data(raw_data)
        
        print("\nSample of processed unique products:")
        sample_cols = ['product_code', 'product_description', 'product_category']
        print(processed_data[sample_cols].head(10))
        
        # Save to CSV for inspection
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "unique_products.csv")
        
        processed_data.to_csv(output_path, index=False)
        print(f"\nSaved processed data to {output_path}")
