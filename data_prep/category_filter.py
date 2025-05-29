"""
Category-based filtering for product data.

This module provides functions to filter and organize products by category
for hierarchical clustering.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add parent directory to path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def filter_products_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe to only include products with category information.
    
    Args:
        df: DataFrame with product data, including category_description column
        
    Returns:
        DataFrame containing only products with category information
    """
    if 'category_description' not in df.columns:
        print("Error: DataFrame does not contain 'category_description' column")
        return df
    
    initial_count = len(df)
    filtered_df = df.dropna(subset=['category_description'])
    filtered_count = len(filtered_df)
    
    print(f"Filtered products by category: {filtered_count}/{initial_count} products retained")
    print(f"Removed {initial_count - filtered_count} products without category information")
    
    return filtered_df

def normalize_category_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize category names for consistent grouping.
    
    Args:
        df: DataFrame with product data, including category_description column
        
    Returns:
        DataFrame with normalized category names
    """
    if 'category_description' not in df.columns:
        print("Error: DataFrame does not contain 'category_description' column")
        return df
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Apply normalization rules to categories
    if not result_df['category_description'].isna().all():
        # Convert to lowercase
        result_df['normalized_category'] = result_df['category_description'].str.lower()
        
        # Remove common unnecessary words and standardize formatting
        result_df['normalized_category'] = result_df['normalized_category'].str.replace('misc ', 'miscellaneous ', regex=False)
        result_df['normalized_category'] = result_df['normalized_category'].str.replace('misc.', 'miscellaneous', regex=False)
        
        # Trim whitespace
        result_df['normalized_category'] = result_df['normalized_category'].str.strip()
        
        # Count unique categories before and after normalization
        original_count = df['category_description'].nunique()
        normalized_count = result_df['normalized_category'].nunique()
        
        print(f"Category normalization: {original_count} original categories -> {normalized_count} normalized categories")
    
    return result_df

def group_products_by_category(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group products by their category and return a dictionary of category-specific DataFrames.
    
    Args:
        df: DataFrame with product data, including normalized_category column
        
    Returns:
        Dictionary mapping category names to DataFrames of products in that category
    """
    category_col = 'normalized_category' if 'normalized_category' in df.columns else 'category_description'
    
    if category_col not in df.columns:
        print(f"Error: DataFrame does not contain '{category_col}' column")
        return {}
    
    # Group by category
    category_groups = {}
    for category, group_df in df.groupby(category_col):
        if pd.isna(category):
            continue
            
        category_groups[category] = group_df
    
    print(f"Grouped products into {len(category_groups)} categories")
    
    # Print some statistics about the groups
    category_sizes = {cat: len(group) for cat, group in category_groups.items()}
    largest_categories = sorted(category_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("Top 5 largest categories:")
    for category, size in largest_categories:
        print(f"  - {category}: {size} products")
    
    return category_groups

def prepare_category_data(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Prepare data for category-based clustering by filtering, normalizing, and grouping.
    
    Args:
        df: DataFrame with product data
        
    Returns:
        Tuple of (category_groups dictionary, complete filtered DataFrame)
    """
    # Filter products to only include those with categories
    filtered_df = filter_products_by_category(df)
    
    # Normalize category names
    normalized_df = normalize_category_names(filtered_df)
    
    # Group by category
    category_groups = group_products_by_category(normalized_df)
    
    return category_groups, normalized_df
