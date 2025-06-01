#!/usr/bin/env python3
"""
Updated Cluster Variance Analysis Script

This script analyzes the variance of pricing metrics within product clusters,
calculating statistics and identifying outlier products. It uses the ClusterID 
column in the pricing data rather than trying to match product codes directly.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional

def load_data(pricing_file: str) -> pd.DataFrame:
    """Load the pricing data with cluster assignments."""
    # Load pricing data
    pricing_df = pd.read_csv(pricing_file)
    print(f"Loaded pricing data with {len(pricing_df)} SKUs")
    
    return pricing_df

def analyze_cluster_variance(pricing_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze variance within each cluster for key pricing metrics.
    
    Args:
        pricing_df: DataFrame with pricing data and cluster IDs
        
    Returns:
        Tuple of DataFrames: (cluster_stats, cluster_products)
    """
    print("Analyzing variance within clusters...")
    
    # Define metrics to analyze
    metrics = ["Implied GP$ (actual)", "Implied GP% (actual)", "SalesPrice", "AccountingCost"]
    
    # Dictionary to store results
    cluster_stats = []
    cluster_products = []
    
    # Get unique clusters
    unique_clusters = pricing_df["ClusterID"].unique()
    print(f"Found {len(unique_clusters)} unique clusters in the data")
    
    # Process each cluster
    for cluster_id in unique_clusters:
        # Filter pricing data for this cluster
        cluster_df = pricing_df[pricing_df["ClusterID"] == cluster_id].copy()
        
        # Skip if no data or only one product in this cluster
        if len(cluster_df) < 2:
            print(f"Skipping cluster {cluster_id}: Not enough data ({len(cluster_df)} products)")
            continue
        
        # Calculate cluster statistics
        stats = {
            "cluster_id": cluster_id,
            "num_products": len(cluster_df)
        }
        
        # Calculate statistics for each metric
        has_enough_data = True
        for metric in metrics:
            if metric in cluster_df.columns:
                metric_values = cluster_df[metric].dropna()
                
                if len(metric_values) >= 2:
                    stats[f"{metric}_mean"] = metric_values.mean()
                    stats[f"{metric}_median"] = metric_values.median()
                    stats[f"{metric}_min"] = metric_values.min()
                    stats[f"{metric}_max"] = metric_values.max()
                    stats[f"{metric}_range"] = metric_values.max() - metric_values.min()
                    stats[f"{metric}_std"] = metric_values.std()
                    
                    # Calculate coefficient of variation (CV) - normalized measure of dispersion
                    # Only calculate if mean is not zero to avoid division by zero
                    if stats[f"{metric}_mean"] != 0:
                        stats[f"{metric}_cv"] = stats[f"{metric}_std"] / abs(stats[f"{metric}_mean"])
                    else:
                        stats[f"{metric}_cv"] = np.nan
                else:
                    print(f"Skipping metric {metric} for cluster {cluster_id}: Not enough non-null values")
                    has_enough_data = False
            else:
                print(f"Skipping metric {metric} for cluster {cluster_id}: Metric not found in data")
                has_enough_data = False
        
        if not has_enough_data:
            continue
            
        # Add to results
        cluster_stats.append(stats)
        
        # Add product details for this cluster
        for _, row in cluster_df.iterrows():
            product_detail = {
                "cluster_id": cluster_id,
                "product_code": row["ProductCode"],
                "sales_price": row["SalesPrice"],
                "accounting_cost": row["AccountingCost"],
                "gp_dollars": row["Implied GP$ (actual)"],
                "gp_percent": row["Implied GP% (actual)"],
                "transaction_count": row["TransactionCount"]
            }
            
            # Add product description if available
            if "ProductDescription" in row:
                product_detail["product_description"] = row["ProductDescription"]
            
            cluster_products.append(product_detail)
    
    # Convert to DataFrames
    if cluster_stats:
        cluster_stats_df = pd.DataFrame(cluster_stats)
        cluster_products_df = pd.DataFrame(cluster_products)
        print(f"Calculated statistics for {len(cluster_stats_df)} clusters")
        return cluster_stats_df, cluster_products_df
    else:
        print("Error: No cluster statistics calculated. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

def identify_outliers(cluster_products_df: pd.DataFrame, cluster_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify outlier products within each cluster based on z-score.
    
    Args:
        cluster_products_df: DataFrame with all products and their metrics
        cluster_stats_df: DataFrame with cluster statistics
        
    Returns:
        DataFrame with outlier products and their z-scores
    """
    print("Identifying outlier products...")
    
    # Metrics to check for outliers
    metrics = [
        ("gp_dollars", "Implied GP$ (actual)_mean", "Implied GP$ (actual)_std"),
        ("gp_percent", "Implied GP% (actual)_mean", "Implied GP% (actual)_std")
    ]
    
    # Dictionary to store outliers
    outliers = []
    
    # Process each cluster
    for cluster_id in cluster_stats_df["cluster_id"].unique():
        # Get cluster stats
        cluster_stats = cluster_stats_df[cluster_stats_df["cluster_id"] == cluster_id].iloc[0]
        
        # Get cluster products
        cluster_products = cluster_products_df[cluster_products_df["cluster_id"] == cluster_id].copy()
        
        # Skip if no data
        if len(cluster_products) < 2:
            continue
        
        # Calculate z-scores for each metric
        for product_metric, mean_metric, std_metric in metrics:
            # Skip if mean or std is missing
            if pd.isna(cluster_stats[mean_metric]) or pd.isna(cluster_stats[std_metric]) or cluster_stats[std_metric] == 0:
                continue
                
            # Calculate z-score
            mean = cluster_stats[mean_metric]
            std = cluster_stats[std_metric]
            
            z_score_col = f"{product_metric}_z_score"
            cluster_products[z_score_col] = (cluster_products[product_metric] - mean) / std
            
            # Flag outliers (|z| > 2)
            outlier_col = f"{product_metric}_outlier"
            cluster_products[outlier_col] = np.abs(cluster_products[z_score_col]) > 2
            
            # Add outliers to results
            outlier_products = cluster_products[cluster_products[outlier_col]].copy()
            
            for _, row in outlier_products.iterrows():
                outlier = {
                    "cluster_id": cluster_id,
                    "product_code": row["product_code"],
                    "metric": product_metric,
                    "value": row[product_metric],
                    "cluster_mean": mean,
                    "cluster_std": std,
                    "z_score": row[z_score_col],
                    "sales_price": row["sales_price"],
                    "accounting_cost": row["accounting_cost"]
                }
                
                # Add product description if available
                if "product_description" in row:
                    outlier["product_description"] = row["product_description"]
                
                outliers.append(outlier)
    
    # Convert to DataFrame
    if outliers:
        outliers_df = pd.DataFrame(outliers)
        print(f"Identified {len(outliers_df)} outlier products")
        return outliers_df
    else:
        print("No outlier products identified")
        return pd.DataFrame()

def main():
    """Main workflow for analyzing cluster variance."""
    
    # Set file paths
    pricing_file = "pricing_analysis_data.csv"   # Analysis dataset with ClusterID column
    output_stats_file = "cluster_variance_stats.csv"
    output_details_file = "cluster_product_details.csv"
    output_outliers_file = "cluster_product_outliers.csv"
    
    # Load pricing data
    pricing_df = load_data(pricing_file)
    
    # Analyze cluster variance
    cluster_stats_df, cluster_products_df = analyze_cluster_variance(pricing_df)
    
    # Exit if no cluster statistics calculated
    if len(cluster_stats_df) == 0:
        return
    
    # Identify outlier products
    outliers_df = identify_outliers(cluster_products_df, cluster_stats_df)
    
    # Save results
    cluster_stats_df.to_csv(output_stats_file, index=False)
    print(f"Saved cluster statistics to {output_stats_file}")
    
    cluster_products_df.to_csv(output_details_file, index=False)
    print(f"Saved cluster product details to {output_details_file}")
    
    if len(outliers_df) > 0:
        outliers_df.to_csv(output_outliers_file, index=False)
        print(f"Saved outlier products to {output_outliers_file}")
    
    # Display summary statistics
    print("\nSummary statistics:")
    print(f"Total clusters analyzed: {len(cluster_stats_df)}")
    print(f"Total products in analyzed clusters: {len(cluster_products_df)}")
    
    # Find clusters with high GP variance
    if "Implied GP% (actual)_cv" in cluster_stats_df.columns:
        high_variance_clusters = cluster_stats_df[cluster_stats_df["Implied GP% (actual)_cv"] > 0.5]
        print(f"Clusters with high GP% variance (CV > 0.5): {len(high_variance_clusters)}")
    
    print("\nTop 5 clusters with highest GP% variance:")
    if "Implied GP% (actual)_cv" in cluster_stats_df.columns:
        top_clusters = cluster_stats_df.sort_values(by="Implied GP% (actual)_cv", ascending=False).head(5)
        for _, row in top_clusters.iterrows():
            cluster_id = row["cluster_id"]
            cv = row["Implied GP% (actual)_cv"]
            num_products = row["num_products"]
            print(f"  Cluster {cluster_id}: CV = {cv:.4f}, Products = {num_products}")

if __name__ == "__main__":
    main()
