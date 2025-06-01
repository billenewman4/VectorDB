#!/usr/bin/env python3
"""
Hierarchical Cluster Analysis

This script analyzes the hierarchical clustering results and outputs a CSV file
with cluster assignments at each level, product details, and USDA codes.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import argparse

# Add parent directories to path to import from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# Import from src directory
sys.path.insert(0, os.path.join(grandparent_dir, 'src'))
from data_processing import load_transaction_data, process_transaction_data, clean_text
from abbreviation_translator import expand_abbreviations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hierarchical_analysis")


def load_hierarchical_clusters(cluster_file_path: str) -> Dict[str, Any]:
    """
    Load the hierarchical clustering results from a JSON file.
    
    Args:
        cluster_file_path: Path to the hierarchical clusters JSON file
        
    Returns:
        Dictionary containing the hierarchical clustering results
    """
    try:
        with open(cluster_file_path, 'r') as f:
            hierarchical_clusters = json.load(f)
        logger.info(f"Loaded hierarchical clusters from {cluster_file_path}")
        return hierarchical_clusters
    except Exception as e:
        logger.error(f"Error loading hierarchical clusters: {e}")
        return {}


def create_product_cluster_mapping(hierarchical_clusters: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Create a mapping from product code to cluster ID at each level.
    
    Args:
        hierarchical_clusters: Dictionary containing the hierarchical clustering results
        
    Returns:
        Dictionary mapping product code to cluster IDs at each level
    """
    # Initialize mapping from product code to cluster IDs
    product_to_clusters = {}
    
    # Process each level of the hierarchy
    for level_name, clusters in hierarchical_clusters.items():
        level_num = int(level_name.split('_')[1])
        
        # Process each cluster at this level
        for cluster_id, cluster_info in clusters.items():
            products = cluster_info.get("products", [])
            
            # Add cluster assignment for each product at this level
            for product in products:
                if product not in product_to_clusters:
                    product_to_clusters[product] = {}
                product_to_clusters[product][f"cluster_{level_num}"] = cluster_id
    
    return product_to_clusters


def load_product_details(data_dir: str) -> pd.DataFrame:
    """
    Load product details from the prepared data file.
    
    Args:
        data_dir: Directory containing the prepared products data
        
    Returns:
        DataFrame containing product details
    """
    # Try multiple locations for product data files
    # 1. Try prepared_products.csv in the data directory
    prepared_data_path = os.path.join(data_dir, "prepared_products.csv")
    
    # 2. Look for processed_transactions.csv in various locations
    possible_paths = [
        os.path.join(data_dir, "processed_transactions.csv"),
        os.path.join(os.path.dirname(data_dir), "processed_transactions.csv"),
        os.path.join(os.path.dirname(os.path.dirname(data_dir)), "processed_transactions.csv"),
        os.path.join(parent_dir, "data", "processed_transactions.csv"),
        os.path.join(grandparent_dir, "data", "processed_transactions.csv")
    ]
    
    # Add prepared_products.csv to possible paths
    possible_paths.append(prepared_data_path)
    
    # Try loading from each path
    for path in possible_paths:
        if os.path.exists(path):
            try:
                product_details = pd.read_csv(path)
                logger.info(f"Loaded product details from {path} with {len(product_details)} records")
                return product_details
            except Exception as e:
                logger.warning(f"Error loading product data from {path}: {e}")
    
    # If all else fails, try the old way of loading transaction data
    try:
        transaction_data = load_transaction_data(data_dir)
        logger.info(f"Loaded transaction data with {len(transaction_data)} records")
        return transaction_data
    except Exception as e:
        logger.error(f"Error loading transaction data: {e}")
        return pd.DataFrame()


def generate_analysis_csv(hierarchical_clusters: Dict[str, Any], 
                         product_details: pd.DataFrame,
                         output_path: str):
    """
    Generate a CSV file with hierarchical cluster assignments and product details.
    
    Args:
        hierarchical_clusters: Dictionary containing the hierarchical clustering results
        product_details: DataFrame containing product details
        output_path: Path to save the output CSV file
    """
    # Create mapping from product code to cluster assignments
    product_cluster_mapping = create_product_cluster_mapping(hierarchical_clusters)
    
    # Get the maximum level in the hierarchy
    max_level = max([int(level_name.split('_')[1]) for level_name in hierarchical_clusters.keys()])
    
    # Create a DataFrame with cluster assignments
    cluster_columns = [f"cluster_{i}" for i in range(1, max_level + 1)]
    
    # Initialize the cluster assignment DataFrame
    cluster_assignments = pd.DataFrame(columns=["product_code"] + cluster_columns)
    
    # Fill the DataFrame with cluster assignments
    for product_code, cluster_ids in product_cluster_mapping.items():
        row = {"product_code": product_code}
        
        # Add cluster ID for each level (empty string if not assigned at that level)
        for level in range(1, max_level + 1):
            column = f"cluster_{level}"
            row[column] = cluster_ids.get(column, "")
        
        # Append row to DataFrame
        cluster_assignments = pd.concat([cluster_assignments, pd.DataFrame([row])], ignore_index=True)
    
    # Merge with product details
    if 'product_code' in product_details.columns:
        # Make sure product_code is string type in both DataFrames for proper joining
        product_details['product_code'] = product_details['product_code'].astype(str)
        cluster_assignments['product_code'] = cluster_assignments['product_code'].astype(str)
        
        # Merge on product_code
        merged_data = pd.merge(
            cluster_assignments,
            product_details,
            on="product_code",
            how="left"
        )
    else:
        logger.warning("Product details DataFrame does not have 'product_code' column. Cannot merge data.")
        merged_data = cluster_assignments
    
    # Select and reorder columns
    # Include all cluster levels, product code, description, company, and USDA code
    desired_columns = cluster_columns + ["product_code"]
    
    # Add product description if available
    for col in ["product_description", "description", "product_name", "name"]:
        if col in merged_data.columns:
            desired_columns.append(col)
            break
    
    # Add company if available
    for col in ["company", "company_name", "vendor", "vendor_name"]:
        if col in merged_data.columns:
            desired_columns.append(col)
            break
    
    # Add USDA code if available
    for col in ["usda_code", "usda", "code"]:
        if col in merged_data.columns:
            desired_columns.append(col)
            break
    
    # Add any remaining columns
    for col in merged_data.columns:
        if col not in desired_columns:
            desired_columns.append(col)
    
    # Filter to only include columns that actually exist
    final_columns = [col for col in desired_columns if col in merged_data.columns]
    
    # Save the merged data
    merged_data[final_columns].to_csv(output_path, index=False)
    logger.info(f"Saved hierarchical cluster analysis to {output_path}")
    logger.info(f"Analysis contains {len(merged_data)} products across {max_level} hierarchical levels")


def analyze_hierarchical_clusters(cluster_file_path: str, data_dir: str, output_dir: str):
    """
    Analyze hierarchical clustering results and generate CSV output.
    
    Args:
        cluster_file_path: Path to the hierarchical clusters JSON file
        data_dir: Directory containing product data
        output_dir: Directory to save analysis results
    """
    logger.info(f"Starting hierarchical cluster analysis...")
    logger.info(f"Cluster file path: {cluster_file_path}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Verify cluster file exists
    if not os.path.exists(cluster_file_path):
        logger.error(f"Cluster file not found at {cluster_file_path}")
        return
    
    # Load hierarchical clusters
    try:
        hierarchical_clusters = load_hierarchical_clusters(cluster_file_path)
        logger.info(f"Successfully loaded hierarchical clusters with {len(hierarchical_clusters.get('levels', []))} levels")
        if not hierarchical_clusters:
            logger.error("No hierarchical clusters found. Cannot proceed with analysis.")
            return
    except Exception as e:
        logger.error(f"Failed to load hierarchical clusters: {e}", exc_info=True)
        return
    
    # Load product details
    try:
        product_details = load_product_details(data_dir)
        if product_details.empty:
            logger.error("No product details found. Cannot proceed with analysis.")
            return
        logger.info(f"Successfully loaded product details with {len(product_details)} products")
        logger.info(f"Product details columns: {product_details.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to load product details: {e}", exc_info=True)
        return
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created or verified output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return
    
    # Generate analysis CSV
    try:
        output_path = os.path.join(output_dir, "hierarchical_cluster_analysis.csv")
        logger.info(f"Generating analysis CSV at {output_path}")
        generate_analysis_csv(hierarchical_clusters, product_details, output_path)
        if os.path.exists(output_path):
            logger.info(f"Successfully generated CSV at {output_path} with size {os.path.getsize(output_path)} bytes")
        else:
            logger.error(f"Failed to generate CSV - file not created at {output_path}")
    except Exception as e:
        logger.error(f"Error generating analysis CSV: {e}", exc_info=True)
    
    # Print summary statistics
    try:
        print_analysis_summary(hierarchical_clusters, product_details)
    except Exception as e:
        logger.error(f"Error printing analysis summary: {e}", exc_info=True)


def print_analysis_summary(hierarchical_clusters: Dict[str, Any], product_details: pd.DataFrame):
    """
    Print summary statistics about the hierarchical clustering.
    
    Args:
        hierarchical_clusters: Dictionary containing the hierarchical clustering results
        product_details: DataFrame containing product details
    """
    logger.info("=== Hierarchical Clustering Summary ===")
    
    # Count total products
    total_products = len(product_details)
    
    # Count products in clusters at each level
    for level_name, clusters in sorted(hierarchical_clusters.items()):
        level_num = int(level_name.split('_')[1])
        
        # Count unique products at this level
        products_at_level = set()
        for cluster_info in clusters.values():
            products_at_level.update(cluster_info.get("products", []))
        
        # Calculate inclusion rate
        inclusion_rate = len(products_at_level) / total_products if total_products > 0 else 0
        
        logger.info(f"Level {level_num}: {len(clusters)} clusters, {len(products_at_level)} products ({inclusion_rate:.1%} inclusion)")
        
        # If we're at level 1, check if we've achieved >90% inclusion
        if level_num == 1 and inclusion_rate < 0.9:
            logger.warning(f"Level 1 inclusion rate is below 90% ({inclusion_rate:.1%}). "
                           "Consider adjusting clustering parameters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hierarchical clustering results")
    parser.add_argument("--clusters", default=None,
                      help="Path to hierarchical clusters JSON file")
    parser.add_argument("--data_dir", default=None,
                      help="Directory containing product data")
    parser.add_argument("--output_dir", default=None,
                      help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.clusters is None:
        args.clusters = os.path.join(base_dir, "data", "hierarchical_clustering", "hierarchical_clusters.json")
    
    if args.data_dir is None:
        args.data_dir = os.path.join(base_dir, "data")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "data", "hierarchical_clustering", "analysis")
    
    # Run analysis
    analyze_hierarchical_clusters(args.clusters, args.data_dir, args.output_dir)
