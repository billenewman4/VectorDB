"""
Category management module.

This module provides functions for organizing products by category
and saving category-based product groups for clustering operations.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any


def group_products_by_category(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group products by their category for category-based clustering.
    This ensures products from different categories are never clustered together.
    
    Args:
        df: DataFrame with product information including product_category column
        
    Returns:
        Dictionary mapping categories to DataFrames of products in that category
    """
    if 'product_category' not in df.columns:
        print("Warning: DataFrame does not contain 'product_category' column")
        return {'all_products': df}  # Return all products in one group
    
    # Initialize dictionary to store category-specific DataFrames
    category_products = {}
    
    # Group by category
    categories = df['product_category'].unique()
    print(f"Grouping {len(df)} products into {len(categories)} categories")
    
    for category in categories:
        # Extract products for this category
        category_df = df[df['product_category'] == category].copy()
        
        # Only include categories with at least 2 products (needed for clustering)
        if len(category_df) >= 2:
            category_products[category] = category_df
        else:
            print(f"  - Skipping {category}: only {len(category_df)} products (minimum 2 required)")
    
    return category_products


def save_category_products(category_products: Dict[str, pd.DataFrame], 
                          output_dir: str) -> Optional[str]:
    """
    Save category-to-products mapping for use in clustering processes.
    
    Args:
        category_products: Dictionary mapping categories to DataFrames of products
        output_dir: Directory to save the mapping
        
    Returns:
        Path to the saved mapping file, or None if saving failed
    """
    if not category_products:
        print("No category products to save")
        return None
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a mapping file
        mapping_path = os.path.join(output_dir, "category_products.json")
        
        # Create a serializable version of the mapping
        # (we can't directly serialize DataFrames to JSON)
        serializable_mapping = {}
        total_products = 0
        
        for category, df in category_products.items():
            # Store product codes for each category
            product_codes = df['product_code'].tolist()
            serializable_mapping[category] = product_codes
            total_products += len(product_codes)
        
        # Save the mapping to JSON
        with open(mapping_path, 'w') as f:
            json.dump(serializable_mapping, f, indent=2)
        
        print(f"Saved mapping of {len(serializable_mapping)} categories with {total_products} total products to {mapping_path}")
        
        # Also save individual CSV files for each category if needed
        csv_dir = os.path.join(output_dir, "category_dataframes")
        os.makedirs(csv_dir, exist_ok=True)
        
        for category, df in category_products.items():
            # Create safe filename by replacing problematic characters
            safe_category = category.replace('/', '_').replace('\\', '_').replace(' ', '_')
            
            # Save to CSV
            csv_path = os.path.join(csv_dir, f"{safe_category}.csv")
            df.to_csv(csv_path, index=False)
        
        print(f"Saved individual category DataFrames to {csv_dir}")
        
        return mapping_path
        
    except Exception as e:
        print(f"Error saving category products: {e}")
        return None


if __name__ == "__main__":
    # For testing, we need sample data
    import sys
    
    # Try to import the transaction processor
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from Data_Loading.transaction_loader import load_transaction_data
        from Data_Processing.transaction_processor import process_transaction_data
    except ImportError:
        print("Error importing required modules for testing")
        sys.exit(1)
    
    # Test category management functions
    raw_data = load_transaction_data()
    
    if raw_data is not None:
        processed_data = process_transaction_data(raw_data)
        
        if not processed_data.empty:
            # Group by category
            category_groups = group_products_by_category(processed_data)
            
            # Save category groups
            test_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output", "test_categories")
            save_category_products(category_groups, test_output_dir)
