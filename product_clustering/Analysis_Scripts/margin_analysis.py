#!/usr/bin/env python3
"""
margin_analysis.py - Analyze gross margin differences within product clusters

This script compares pricing consistency within product clusters by calculating
margin statistics and identifying potential pricing issues, including high-variance
clusters and outlier products with unusual margins relative to similar products.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_cluster_data(filepath: str) -> pd.DataFrame:
    """
    Load product cluster assignments from JSON file.
    
    Args:
        filepath: Path to refined_clusters.json
        
    Returns:
        DataFrame with product_id and cluster_id columns
    """
    try:
        with open(filepath, 'r') as f:
            clusters = json.load(f)
        
        # Convert to list of product records with cluster IDs
        product_clusters = []
        for cluster_id, product_ids in clusters.items():
            for product_id in product_ids:
                product_clusters.append({
                    'product_id': str(product_id),
                    'cluster_id': cluster_id
                })
        
        return pd.DataFrame(product_clusters)
    except Exception as e:
        print(f"Error loading cluster data: {str(e)}")
        return pd.DataFrame()


def load_pricing_data(filepath: str) -> pd.DataFrame:
    """
    Load product pricing data from CSV or Excel.
    
    Args:
        filepath: Path to pricing data CSV or Excel file
        
    Returns:
        DataFrame with product pricing information
    """
    try:
        # Determine file type from extension
        if filepath.lower().endswith('.csv'):
            pricing_df = pd.read_csv(filepath)
        elif filepath.lower().endswith(('.xlsx', '.xls')):
            try:
                # Try to load with default engine
                pricing_df = pd.read_excel(filepath, engine='openpyxl')
            except Exception as excel_err:
                print(f"Warning: Error with openpyxl engine: {str(excel_err)}")
                # Try with xlrd engine for older Excel formats
                try:
                    pricing_df = pd.read_excel(filepath, engine='xlrd')
                except Exception as xlrd_err:
                    # As a last resort, try with odf engine
                    try:
                        pricing_df = pd.read_excel(filepath, engine='odf')
                    except Exception as odf_err:
                        raise Exception(f"Failed to read Excel with multiple engines: {str(excel_err)}, {str(xlrd_err)}, {str(odf_err)}")
        else:
            raise Exception(f"Unsupported file format: {filepath}")
        
        # Print the first few rows to help with debugging
        print("First 5 rows of pricing data:")
        print(pricing_df.head())
        
        return pricing_df
    except Exception as e:
        print(f"Error loading pricing data: {str(e)}")
        return pd.DataFrame()


def calculate_margins(cluster_df: pd.DataFrame, pricing_df: pd.DataFrame, price_col: str = 'price', cost_col: str = 'cost', product_id_col: str = 'product_id') -> pd.DataFrame:
    """
    Join cluster and pricing data, then calculate gross margins.
    
    Args:
        cluster_df: DataFrame with product cluster assignments
        pricing_df: DataFrame with product pricing information
        price_col: Name of the price column in pricing_df
        cost_col: Name of the cost column in pricing_df
        product_id_col: Name of the product ID column in pricing_df
        
    Returns:
        DataFrame with joined data and calculated margins
    """
    # Make a copy of pricing_df to avoid modifying the original
    pricing_df = pricing_df.copy()
    
    # Rename columns for consistency if they don't match expected names
    column_mapping = {}
    if product_id_col != 'product_id' and product_id_col in pricing_df.columns:
        column_mapping[product_id_col] = 'product_id'
    if price_col != 'price' and price_col in pricing_df.columns:
        column_mapping[price_col] = 'price'
    if cost_col != 'cost' and cost_col in pricing_df.columns:
        column_mapping[cost_col] = 'cost'
    
    if column_mapping:
        pricing_df = pricing_df.rename(columns=column_mapping)
    
    # Ensure product_id is treated as string for joining
    if 'product_id' in pricing_df.columns:
        pricing_df['product_id'] = pricing_df['product_id'].astype(str)
    
    # Merge datasets on product_id
    merged_df = pd.merge(
        cluster_df, 
        pricing_df, 
        on='product_id', 
        how='inner'
    )
    
    # Check if we have the necessary columns
    if 'price' not in merged_df.columns or 'cost' not in merged_df.columns:
        print("Warning: Missing price or cost columns in the joined data")
        print(f"Available columns: {merged_df.columns.tolist()}")
        return pd.DataFrame()
    
    # Calculate gross margin
    merged_df['gross_margin'] = (merged_df['price'] - merged_df['cost']) / merged_df['price']
    
    # Drop rows with invalid margins (e.g., division by zero, negative margins)
    invalid_margins = merged_df[~np.isfinite(merged_df['gross_margin'])].shape[0]
    if invalid_margins > 0:
        print(f"Warning: Dropped {invalid_margins} rows with invalid margins (infinity, NaN, etc.)")
        merged_df = merged_df[np.isfinite(merged_df['gross_margin'])]
    
    return merged_df


def compute_cluster_statistics(margin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate margin statistics for each cluster.
    
    Args:
        margin_df: DataFrame with product margins and cluster assignments
        
    Returns:
        DataFrame with per-cluster margin statistics
    """
    # Group by cluster and calculate statistics
    cluster_stats = margin_df.groupby('cluster_id').agg({
        'product_id': 'count',
        'gross_margin': ['mean', 'median', 'std', 'min', 'max']
    })
    
    # Flatten multi-index columns
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    
    # Rename columns for clarity
    cluster_stats = cluster_stats.rename(columns={
        'product_id_count': 'product_count',
        'gross_margin_mean': 'mean_margin',
        'gross_margin_median': 'median_margin',
        'gross_margin_std': 'std_margin',
        'gross_margin_min': 'min_margin',
        'gross_margin_max': 'max_margin'
    })
    
    # Calculate margin range
    cluster_stats['margin_range'] = cluster_stats['max_margin'] - cluster_stats['min_margin']
    
    # Handle clusters with only one product (NaN standard deviation)
    cluster_stats['std_margin'] = cluster_stats['std_margin'].fillna(0)
    
    # Reset index to make cluster_id a column
    cluster_stats = cluster_stats.reset_index()
    
    return cluster_stats


def identify_pricing_issues(
    margin_df: pd.DataFrame, 
    cluster_stats: pd.DataFrame, 
    variance_threshold: float = 0.15,
    z_score_threshold: float = 2.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify clusters with high margin variance and outlier products.
    
    Args:
        margin_df: DataFrame with product margins and cluster assignments
        cluster_stats: DataFrame with per-cluster margin statistics
        variance_threshold: Threshold for flagging high-variance clusters
        z_score_threshold: Z-score threshold for flagging outlier products
        
    Returns:
        Tuple containing:
        - Updated cluster_stats with high_variance flag
        - DataFrame of margin outliers
    """
    # Flag high-variance clusters
    cluster_stats['high_variance'] = cluster_stats['std_margin'] > variance_threshold
    
    # Identify outlier products within each cluster
    outliers = []
    
    # Process each cluster
    for _, stats in cluster_stats.iterrows():
        # Skip clusters with only one product or zero standard deviation
        if stats['product_count'] <= 1 or stats['std_margin'] == 0:
            continue
        
        # Get products in this cluster
        cluster_products = margin_df[margin_df['cluster_id'] == stats['cluster_id']]
        
        # Calculate z-scores
        mean_margin = stats['mean_margin']
        std_margin = stats['std_margin']
        
        if std_margin > 0:  # Avoid division by zero
            cluster_products['z_score'] = (cluster_products['gross_margin'] - mean_margin) / std_margin
            
            # Find outliers
            cluster_outliers = cluster_products[abs(cluster_products['z_score']) > z_score_threshold]
            
            if not cluster_outliers.empty:
                # Add to outliers list
                outliers.append(cluster_outliers[['product_id', 'cluster_id', 'gross_margin', 'z_score']])
    
    # Combine all outliers
    if outliers:
        margin_outliers = pd.concat(outliers)
    else:
        margin_outliers = pd.DataFrame(columns=['product_id', 'cluster_id', 'gross_margin', 'z_score'])
    
    return cluster_stats, margin_outliers


def main():
    """Main workflow for margin analysis."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze gross margin differences within product clusters')
    parser.add_argument('--clusters', type=str, default='product_clustering/data/refined_clustering/refined_clusters.json',
                        help='Path to refined_clusters.json')
    parser.add_argument('--pricing', type=str, default='product_pricing.csv',
                        help='Path to pricing data CSV or Excel')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save output files')
    parser.add_argument('--variance_threshold', type=float, default=0.15,
                        help='Threshold for flagging high-variance clusters')
    parser.add_argument('--z_score_threshold', type=float, default=2.0,
                        help='Z-score threshold for flagging outlier products')
    parser.add_argument('--product_id_col', type=str, default='product_id',
                        help='Name of the product ID column in pricing data')
    parser.add_argument('--price_col', type=str, default='price',
                        help='Name of the price column in pricing data')
    parser.add_argument('--cost_col', type=str, default='cost',
                        help='Name of the cost column in pricing data')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load input data
    print(f"Loading cluster data from {args.clusters}...")
    cluster_df = load_cluster_data(args.clusters)
    if cluster_df.empty:
        print("Error: Failed to load cluster data. Exiting.")
        return
    
    print(f"Loading pricing data from {args.pricing}...")
    pricing_df = load_pricing_data(args.pricing)
    if pricing_df.empty:
        print("Error: Failed to load pricing data. Exiting.")
        return
    
    # Print available columns to help with debugging
    print(f"Available columns in pricing data: {pricing_df.columns.tolist()}")
    
    # Step 2: Calculate margins
    print("Calculating gross margins...")
    margin_df = calculate_margins(
        cluster_df, 
        pricing_df, 
        price_col=args.price_col,
        cost_col=args.cost_col,
        product_id_col=args.product_id_col
    )
    if margin_df.empty:
        print("Error: Failed to calculate margins. Exiting.")
        return
    
    # Step 3: Compute cluster statistics
    print("Computing per-cluster margin statistics...")
    cluster_stats = compute_cluster_statistics(margin_df)
    
    # Step 4: Identify pricing issues
    print("Identifying potential pricing issues...")
    cluster_stats, margin_outliers = identify_pricing_issues(
        margin_df, 
        cluster_stats, 
        args.variance_threshold,
        args.z_score_threshold
    )
    
    # Step 5: Save outputs
    cluster_stats_path = os.path.join(args.output_dir, 'cluster_margin_stats.csv')
    outliers_path = os.path.join(args.output_dir, 'margin_outliers.csv')
    
    print(f"Saving cluster statistics to {cluster_stats_path}...")
    cluster_stats.to_csv(cluster_stats_path, index=False)
    
    print(f"Saving margin outliers to {outliers_path}...")
    margin_outliers.to_csv(outliers_path, index=False)
    
    # Step 6: Print summary
    high_variance_clusters = cluster_stats['high_variance'].sum()
    total_clusters = len(cluster_stats)
    total_outliers = len(margin_outliers)
    
    print("\n===== MARGIN ANALYSIS SUMMARY =====")
    print(f"Total clusters analyzed: {total_clusters}")
    print(f"High-variance clusters: {high_variance_clusters} ({high_variance_clusters/total_clusters*100:.1f}%)")
    print(f"Outlier products identified: {total_outliers}")
    print("==================================")


if __name__ == "__main__":
    main()
