"""
Unified data preparation module for the VectorDB project.

This module provides functions for loading, processing, and preparing product data
from various sources, including transaction reports and inventory valuation files.
It consolidates functionality from previous data processing modules and extends
capabilities to include additional product attributes like warehouse and category.
"""
import os
import sys
import pandas as pd
import numpy as np
import re
import glob
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src import config
from data_prep.abbreviation_translator import expand_abbreviations

def clean_text(text):
    """
    Basic text cleaning: lowercase, strip whitespace.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if isinstance(text, str):
        text = text.lower().strip()
        # Remove excessive whitespace inside the string
        text = re.sub(r'\s+', ' ', text)
    return text

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
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_transaction_data(file_path=config.TRANSACTION_REPORT_FILE, 
                          sheet_name=config.TRANSACTION_SHEET_NAME):
    """
    Loads transaction data from the specified Excel file and sheet.
    
    Args:
        file_path: Path to the transaction Excel file
        sheet_name: Name of the sheet to load
        
    Returns:
        DataFrame containing transaction data or None if loading fails
    """
    print(f"Loading transaction data from: {file_path}, Sheet: {sheet_name}")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: Transaction file not found at {file_path}")
        return None
    except Exception as e: # Catch other potential errors like sheet not found
        print(f"Error loading transaction data: {e}")
        return None

def load_inventory_valuation_files(directory=os.path.join(parent_dir, "data", "Actuals")):
    """
    Loads all inventory valuation files from the specified directory.
    
    Args:
        directory: Directory containing inventory valuation files
        
    Returns:
        Dictionary mapping warehouse names to their respective DataFrames
    """
    print(f"Loading inventory valuation files from: {directory}")
    inventory_files = glob.glob(os.path.join(directory, "*Inventory*.xls"))
    inventory_dfs = {}
    
    for file_path in inventory_files:
        try:
            # Extract warehouse name from filename
            filename = os.path.basename(file_path)
            warehouse_name = filename.split("Inventory")[0].strip()
            if not warehouse_name:  # For files that might not follow the naming pattern
                warehouse_name = os.path.splitext(filename)[0]
                
            print(f"Loading inventory data for {warehouse_name} from: {filename}")
            df = pd.read_excel(file_path)
            
            # Ensure required columns exist
            required_cols = ['Product Code', 'Product Description 1', 'Product Warehouse', 'Category Description']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {filename} is missing required columns. Available columns: {df.columns.tolist()}")
                continue
                
            inventory_dfs[warehouse_name] = df
            print(f"Successfully loaded {len(df)} rows for {warehouse_name}.")
        except Exception as e:
            print(f"Error loading inventory data from {file_path}: {e}")
            
    print(f"Loaded {len(inventory_dfs)} inventory valuation files.")
    return inventory_dfs

def process_transaction_data(df_raw):
    """
    Processes the raw transaction data to extract unique product descriptions 
    and their associated product codes.
    
    Args:
        df_raw: Raw transaction DataFrame
        
    Returns:
        DataFrame with unique product descriptions and codes
    """
    if df_raw is None:
        print("No raw data to process.")
        return None

    code_col = config.TRANSACTION_PRODUCT_CODE_COL
    desc_col = config.TRANSACTION_DESC_COL

    print(f"Processing transaction data using columns: Code='{code_col}', Description='{desc_col}'")

    # Check if required columns exist
    if code_col not in df_raw.columns or desc_col not in df_raw.columns:
        print(f"Error: Required columns ('{code_col}', '{desc_col}') not found in the dataframe.")
        print(f"Available columns: {df_raw.columns.tolist()}")
        return None

    # Select necessary columns
    df = df_raw[[code_col, desc_col]].copy()

    # Handle missing values
    initial_rows = len(df)
    df.dropna(subset=[code_col, desc_col], inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows with missing ProductCode or ProductDescription.")

    # Ensure consistent types
    df[code_col] = df[code_col].astype(str) # Codes should be strings
    df[desc_col] = df[desc_col].astype(str)

    # Clean description and expand meat cut abbreviations
    df['cleaned_description'] = df[desc_col].apply(lambda text: expand_abbreviations(clean_text(text)))

    # Filter out empty descriptions after cleaning
    initial_rows = len(df)
    df = df[df['cleaned_description'] != '']
    if len(df) < initial_rows:
         print(f"Dropped {initial_rows - len(df)} rows with empty descriptions after cleaning.")

    # Get unique descriptions and their first associated code
    print("Extracting unique descriptions and associated codes...")
    # Group by cleaned description, take the first occurrence's code
    unique_products_df = df.groupby('cleaned_description').first().reset_index()
    
    # Rename columns for clarity
    unique_products_df = unique_products_df.rename(columns={
        'cleaned_description': 'product_description', # This is the column to embed
        code_col: 'product_code' # The associated code for linking
    })
    
    # Select final columns
    final_cols = ['product_description', 'product_code'] 
    unique_products_df = unique_products_df[final_cols]

    print(f"Found {len(unique_products_df)} unique product descriptions for embedding.")
    
    return unique_products_df

def unify_inventory_data(inventory_dfs):
    """
    Unifies inventory data from multiple warehouses into a single DataFrame.
    
    Args:
        inventory_dfs: Dictionary of warehouse names to inventory DataFrames
        
    Returns:
        Unified DataFrame with inventory data from all warehouses
    """
    if not inventory_dfs:
        print("No inventory data to unify.")
        return None
        
    print("Unifying inventory data from multiple warehouses...")
    all_inventory = []
    
    for warehouse, df in inventory_dfs.items():
        # Ensure consistent column names
        required_cols = ['Product Code', 'Product Description 1', 'Product Warehouse', 'Category Description']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Skipping {warehouse} data due to missing required columns.")
            continue
            
        # Select and rename columns for consistency
        df_subset = df[required_cols].copy()
        df_subset = df_subset.rename(columns={
            'Product Code': 'product_code',
            'Product Description 1': 'product_description_raw',
            'Product Warehouse': 'product_warehouse',
            'Category Description': 'category_description'
        })
        
        # Ensure consistent types
        df_subset['product_code'] = df_subset['product_code'].astype(str)
        
        # Clean and expand product descriptions
        df_subset['product_description'] = df_subset['product_description_raw'].apply(
            lambda text: expand_abbreviations(clean_text(text))
        )
        
        # Add source warehouse name
        df_subset['source_warehouse'] = warehouse
        
        all_inventory.append(df_subset)
    
    if not all_inventory:
        print("No valid inventory data found.")
        return None
        
    # Combine all inventory data
    unified_df = pd.concat(all_inventory, ignore_index=True)
    
    # Remove duplicates based on product code and warehouse
    initial_rows = len(unified_df)
    unified_df = unified_df.drop_duplicates(subset=['product_code', 'product_warehouse'])
    if len(unified_df) < initial_rows:
        print(f"Removed {initial_rows - len(unified_df)} duplicate product-warehouse combinations.")
    
    print(f"Unified inventory data contains {len(unified_df)} unique product-warehouse combinations.")
    return unified_df

def merge_product_data(transaction_df, inventory_df):
    """
    Merges product data from transaction and inventory sources.
    
    Args:
        transaction_df: DataFrame with transaction data
        inventory_df: DataFrame with inventory data
        
    Returns:
        Merged DataFrame with unified product information
    """
    if transaction_df is None:
        print("No transaction data available for merging.")
        if inventory_df is not None:
            print("Using only inventory data.")
            return inventory_df
        return None
        
    if inventory_df is None:
        print("No inventory data available for merging.")
        print("Using only transaction data.")
        return transaction_df
    
    print("Merging transaction and inventory data...")
    
    # Start with transaction data
    unified_df = transaction_df.copy()
    
    # Add inventory information columns if not present
    if 'product_warehouse' not in unified_df.columns:
        unified_df['product_warehouse'] = None
    if 'category_description' not in unified_df.columns:
        unified_df['category_description'] = None
        
    # Temp dataframe from inventory for merging
    inventory_merge = inventory_df[['product_code', 'product_warehouse', 'category_description']].drop_duplicates()
    
    # Group by product code to create a mapping of codes to warehouses and categories
    product_info = {}
    for code, group in inventory_merge.groupby('product_code'):
        warehouses = group['product_warehouse'].dropna().unique().tolist()
        categories = group['category_description'].dropna().unique().tolist()
        product_info[code] = {
            'warehouses': warehouses,
            'categories': categories
        }
    
    # Update transaction data with inventory information
    for idx, row in unified_df.iterrows():
        code = row['product_code']
        if code in product_info:
            info = product_info[code]
            
            # Join warehouses with comma if multiple
            if info['warehouses']:
                unified_df.at[idx, 'product_warehouse'] = ', '.join(info['warehouses'])
                
            # Use first category if available
            if info['categories']:
                unified_df.at[idx, 'category_description'] = info['categories'][0]
    
    # Create clustering_description field for embedding/clustering
    unified_df['clustering_description'] = unified_df['product_description'].apply(preprocess_text_for_clustering)
    
    print(f"Merged data contains {len(unified_df)} unique products with enhanced information.")
    
    # Sample of prepared data
    print("\nSample of prepared data:")
    sample_cols = ['product_code', 'product_description', 'product_warehouse', 
                   'category_description', 'clustering_description']
    print(unified_df[sample_cols].head().to_string())
    
    return unified_df

def prepare_unified_product_data(
    transaction_file_path=config.TRANSACTION_REPORT_FILE,
    transaction_sheet_name=config.TRANSACTION_SHEET_NAME,
    inventory_dir=os.path.join(parent_dir, "data", "Actuals")
):
    """
    Master function that prepares unified product data from all available sources.
    
    Args:
        transaction_file_path: Path to transaction data file
        transaction_sheet_name: Sheet name in transaction file
        inventory_dir: Directory containing inventory valuation files
        
    Returns:
        DataFrame with unified product data from all sources
    """
    print("=== Starting Unified Product Data Preparation ===")
    
    # Load transaction data
    transaction_raw = load_transaction_data(transaction_file_path, transaction_sheet_name)
    if transaction_raw is not None:
        transaction_df = process_transaction_data(transaction_raw)
    else:
        transaction_df = None
        
    # Load inventory data
    inventory_dfs = load_inventory_valuation_files(inventory_dir)
    if inventory_dfs:
        inventory_df = unify_inventory_data(inventory_dfs)
    else:
        inventory_df = None
        
    # Merge all data sources
    unified_product_df = merge_product_data(transaction_df, inventory_df)
    
    if unified_product_df is None or len(unified_product_df) == 0:
        print("Error: Failed to prepare unified product data.")
        return pd.DataFrame()
    
    print("=== Unified Product Data Preparation Complete ===")
    print(f"Final dataset contains {len(unified_product_df)} products.")
    
    return unified_product_df

if __name__ == "__main__":
    # Test the unified data preparation
    unified_data = prepare_unified_product_data()
    
    # Save to CSV for inspection
    output_dir = os.path.join(parent_dir, "data_prep", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "unified_products.csv")
    
    unified_data.to_csv(output_path, index=False)
    print(f"Saved unified product data to {output_path}")
