"""
Data preparation pipeline for product clustering.

This module serves as the main entry point for data preparation operations,
orchestrating the complete workflow from loading data to preparing it for clustering.
"""

import os
import sys
import pandas as pd
from typing import Dict, Optional, List, Any

# Add the current directory to the path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from our modular structure with proper relative import syntax
from .Text_Processing.text_normalization import preprocess_text_for_clustering
from .Data_Loading.transaction_loader import load_transaction_data
from .Data_Processing.transaction_processor import process_transaction_data
from .Category_Management.category_manager import group_products_by_category, save_category_products


def prepare_data_for_clustering(df_raw: Optional[pd.DataFrame] = None,
                               file_path: Optional[str] = None,
                               use_category_descriptions: bool = True,
                               normalize_text: bool = True,
                               expand_abbreviations: bool = True,
                               test_mode: bool = False,
                               test_sample_size: int = 100) -> pd.DataFrame:
    """
    Prepare transaction data for product clustering.
    
    This function orchestrates the complete data preparation workflow by:
    1. Loading transaction data if not provided
    2. Processing data to extract unique products
    3. Normalizing text for better clustering results
    4. Organizing products by category
    5. Saving prepared data for later use
    
    Args:
        df_raw: Optional raw transaction DataFrame. If None, data will be loaded
                from the default location.
        use_category_descriptions: Whether to use category descriptions in clustering.
                If False, products without category descriptions will not be filtered out.
        normalize_text: Whether to apply text normalization to product descriptions.
                If False, raw product descriptions will be used for clustering.
        expand_abbreviations: Whether to expand common abbreviations (e.g. 'oz' to 'ounce')
                during text normalization. Only applies when normalize_text is True.
        test_mode: If True, uses only a subset of data for faster processing during development.
        test_sample_size: Number of rows to use in test mode.
                
    Returns:
        DataFrame with products prepared for clustering
    """
    print("Preparing data for product clustering...")
    
    # Load data if not provided
    if df_raw is None:
        print("Loading transaction data...")
        if file_path:
            print(f"Loading from specified file path: {file_path}")
            df_raw = load_transaction_data(file_path=file_path)
        else:
            print("Using default transaction file path")
            df_raw = load_transaction_data()
            
        if df_raw is None:
            print("Error: Failed to load transaction data")
            return pd.DataFrame()
    
    # Use a smaller subset for test mode
    if test_mode and len(df_raw) > test_sample_size:
        print(f"TEST MODE: Using sample of {test_sample_size} rows from {len(df_raw)} total")
        df_raw = df_raw.sample(test_sample_size, random_state=42)
    
    # Process transaction data using our pipeline to get unique products
    print("Processing transaction data to extract unique products...")
    unique_products_df = process_transaction_data(
        df_raw, 
        filter_no_category=use_category_descriptions
    )
    
    if unique_products_df.empty:
        print("Error: No products found after processing")
        return pd.DataFrame()
    
    print(f"Found {len(unique_products_df)} unique products")
    
    # Clean the product descriptions for clustering (if normalization is enabled)
    if normalize_text:
        print("Normalizing text descriptions...")
        unique_products_df['clustering_description'] = unique_products_df['product_description'].apply(
            lambda text: preprocess_text_for_clustering(text, expand_abbreviations_flag=expand_abbreviations)
        )
    else:
        print("Using raw product descriptions without normalization...")
        unique_products_df['clustering_description'] = unique_products_df['product_description']
    
    # Handle category grouping based on configuration
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "clustering_data")
    os.makedirs(output_dir, exist_ok=True)

    if use_category_descriptions:
        print("\n[Category Mode] Grouping products by category...")
        category_groups = group_products_by_category(unique_products_df)
        # Save category products mapping for clustering
        print("Saving category-to-products mapping...")
        category_mapping_path = save_category_products(category_groups, output_dir)
        if category_mapping_path:
            print(f"Saved category-to-products mapping to {category_mapping_path}")
    else:
        print("\n[No Category Mode] Skipping category grouping. All products will be treated as a single group.")
        # Save all products as one group for downstream compatibility
        all_products_group = {'all_products': unique_products_df}
        # Optionally save a single CSV for all products
        all_products_path = os.path.join(output_dir, "all_products.csv")
        unique_products_df.to_csv(all_products_path, index=False)
        print(f"Saved all products to {all_products_path}")
        # If downstream code expects category mapping files, warn here
        # (You may need to update downstream consumers to handle this mode)

    # Save the full prepared dataset
    prepared_data_path = os.path.join(output_dir, "prepared_products.csv")
    unique_products_df.to_csv(prepared_data_path, index=False)
    print(f"Saved complete prepared dataset to {prepared_data_path}")
    
    print("Data preparation complete.")
    print(f"Prepared {len(unique_products_df)} products for clustering.")
    
    # Display sample of prepared data
    print("\nSample of prepared data:")
    sample_cols = ['product_code', 'product_description', 'product_category', 'clustering_description']
    print(unique_products_df[sample_cols].head().to_string())
    
    return unique_products_df


if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Prepare data for product clustering")
    parser.add_argument("--no-categories", action="store_false", dest="use_categories",
                        help="Don't filter products without category information")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize",
                        help="Don't normalize text descriptions")
    parser.add_argument("--no-expand-abbr", action="store_false", dest="expand_abbr",
                        help="Don't expand abbreviations in text normalization")
    parser.add_argument("--test", action="store_true", help="Run in test mode with sample data")
    parser.add_argument("--sample-size", type=int, default=100, 
                        help="Number of rows to use in test mode")
    
    args = parser.parse_args()
    
    # Run the preparation with command line arguments
    prepared_data = prepare_data_for_clustering(
        use_category_descriptions=args.use_categories,
        normalize_text=args.normalize,
        expand_abbreviations=args.expand_abbr,
        test_mode=args.test,
        test_sample_size=args.sample_size
    )
