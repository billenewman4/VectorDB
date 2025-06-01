#!/usr/bin/env python3
"""
Create Analysis Data Script

This script creates the necessary data files for pricing variance analysis:
1. Loads the original pricing data and cluster definitions
2. Identifies product codes that exist in both datasets
3. Creates a mapping between original product codes and cluster product codes
4. Generates a new pricing data file with cluster IDs
5. Creates cluster files with products that have pricing data
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Set
import re

def load_data():
    """Load the pricing and cluster data."""
    # Load pricing data
    pricing_file = "product_pricing_averaged.csv"
    pricing_df = pd.read_csv(pricing_file)
    print(f"Loaded pricing data with {len(pricing_df)} SKUs")
    
    # Load cluster definitions
    cluster_file = "refined_clusters.json"
    with open(cluster_file, 'r') as f:
        clusters = json.load(f)
    print(f"Loaded {len(clusters)} clusters")
    
    return pricing_df, clusters

def preprocess_product_codes(pricing_df):
    """Preprocess product codes to improve matching."""
    # Make a copy of the dataframe
    df = pricing_df.copy()
    
    # Convert ProductCode to string and remove .0 from float values
    df["ProductCode_Original"] = df["ProductCode"]
    df["ProductCode"] = df["ProductCode"].astype(str).apply(
        lambda x: x.replace(".0", "") if x.endswith(".0") else x
    )
    
    # Create additional formats for matching
    df["ProductCode_Upper"] = df["ProductCode"].str.upper()
    df["ProductCode_NoLeadingZeros"] = df["ProductCode"].apply(
        lambda x: x.lstrip("0") if x.strip().isdigit() else x
    )
    df["ProductCode_Numeric"] = df["ProductCode"].apply(
        lambda x: re.sub(r'[^0-9]', '', x)
    )
    
    return df

def create_cluster_product_mapping(clusters):
    """Create a mapping from product codes to cluster IDs."""
    product_to_cluster = {}
    
    # Track all cluster products
    all_cluster_products = set()
    
    for cluster_id, products in clusters.items():
        for product in products:
            product_str = str(product)
            product_to_cluster[product_str] = cluster_id
            all_cluster_products.add(product_str)
            
            # Also add without leading zeros for better matching
            if product_str.isdigit():
                product_no_zeros = product_str.lstrip("0")
                if product_no_zeros:  # Avoid empty strings
                    product_to_cluster[product_no_zeros] = cluster_id
    
    print(f"Created mapping for {len(all_cluster_products)} unique cluster products")
    return product_to_cluster, all_cluster_products

def match_products_to_clusters(pricing_df, product_to_cluster):
    """Match products in pricing data to clusters."""
    # Make a copy
    df = pricing_df.copy()
    
    # Try to match using different product code formats
    matched_count = 0
    
    # Check primary format first
    df["ClusterID"] = df["ProductCode"].map(product_to_cluster)
    matched_count = df["ClusterID"].notna().sum()
    print(f"Matched {matched_count} products using primary format")
    
    # Try uppercase format for unmatched products
    mask = df["ClusterID"].isna()
    df.loc[mask, "ClusterID"] = df.loc[mask, "ProductCode_Upper"].map(product_to_cluster)
    new_matched = df["ClusterID"].notna().sum() - matched_count
    matched_count += new_matched
    print(f"Matched {new_matched} additional products using uppercase format")
    
    # Try without leading zeros for unmatched products
    mask = df["ClusterID"].isna()
    df.loc[mask, "ClusterID"] = df.loc[mask, "ProductCode_NoLeadingZeros"].map(product_to_cluster)
    new_matched = df["ClusterID"].notna().sum() - matched_count
    matched_count += new_matched
    print(f"Matched {new_matched} additional products using no leading zeros format")
    
    # Try numeric-only format for unmatched products
    mask = df["ClusterID"].isna()
    df.loc[mask, "ClusterID"] = df.loc[mask, "ProductCode_Numeric"].map(product_to_cluster)
    new_matched = df["ClusterID"].notna().sum() - matched_count
    matched_count += new_matched
    print(f"Matched {new_matched} additional products using numeric-only format")
    
    return df

def create_cluster_data(pricing_df_with_clusters):
    """Create a dictionary of clusters with matched products."""
    # Filter to only matched products
    matched_df = pricing_df_with_clusters[pricing_df_with_clusters["ClusterID"].notna()].copy()
    
    # Group by cluster
    clusters_with_products = {}
    for cluster_id, group in matched_df.groupby("ClusterID"):
        product_codes = group["ProductCode"].tolist()
        if len(product_codes) >= 2:  # Only include clusters with at least 2 products
            clusters_with_products[cluster_id] = product_codes
    
    print(f"Created {len(clusters_with_products)} clusters with at least 2 matched products")
    return clusters_with_products

def main():
    """Main workflow."""
    print("Loading data...")
    pricing_df, clusters = load_data()
    
    print("\nPreprocessing product codes...")
    pricing_df = preprocess_product_codes(pricing_df)
    
    print("\nCreating cluster product mapping...")
    product_to_cluster, all_cluster_products = create_cluster_product_mapping(clusters)
    
    print("\nMatching products to clusters...")
    pricing_df_with_clusters = match_products_to_clusters(pricing_df, product_to_cluster)
    
    # Print a few examples of matched products
    matched_examples = pricing_df_with_clusters[pricing_df_with_clusters["ClusterID"].notna()].head(5)
    print("\nExamples of matched products:")
    for _, row in matched_examples.iterrows():
        print(f"  Product {row['ProductCode']} matched to cluster {row['ClusterID']}")
    
    # Create a dataset with only the matched products
    print("\nCreating analysis dataset...")
    matched_df = pricing_df_with_clusters[pricing_df_with_clusters["ClusterID"].notna()].copy()
    print(f"Dataset contains {len(matched_df)} matched products")
    
    # Group matched products by cluster
    cluster_counts = matched_df.groupby("ClusterID").size()
    clusters_with_multiple = cluster_counts[cluster_counts >= 2].index.tolist()
    print(f"Found {len(clusters_with_multiple)} clusters with at least 2 matched products")
    
    # Create the final analysis dataset with only products in valid clusters
    analysis_df = matched_df[matched_df["ClusterID"].isin(clusters_with_multiple)].copy()
    
    # Keep only the necessary columns for analysis
    analysis_cols = [
        "ProductCode", 
        "ClusterID", 
        "Implied GP$ (actual)", 
        "Implied GP% (actual)", 
        "SalesPrice", 
        "AccountingCost", 
        "TransactionCount"
    ]
    
    if "ProductDescription" in analysis_df.columns:
        analysis_cols.append("ProductDescription")
    
    analysis_df = analysis_df[analysis_cols]
    
    # Save the analysis dataset
    analysis_df.to_csv("pricing_analysis_data.csv", index=False)
    print(f"Saved analysis dataset with {len(analysis_df)} products to pricing_analysis_data.csv")
    
    # Create cluster data for the variance analysis
    cluster_data = create_cluster_data(pricing_df_with_clusters)
    
    # Save cluster data
    with open("analysis_clusters.json", "w") as f:
        json.dump(cluster_data, f, indent=2)
    print(f"Saved {len(cluster_data)} clusters to analysis_clusters.json")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Run 'python cluster_variance_analysis.py' using the new data files")
    print("2. Then run 'python analyze_top_variance_clusters.py' to generate the final analysis")

if __name__ == "__main__":
    main()
