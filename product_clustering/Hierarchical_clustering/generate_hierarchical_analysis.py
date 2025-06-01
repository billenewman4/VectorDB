#!/usr/bin/env python3
"""
Generate Hierarchical Clustering Analysis

This script directly analyzes the hierarchical clustering results and
creates a CSV with all cluster assignments and product details.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import glob
import shutil

# Add parent directories to path to import from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hierarchical_analysis")

def find_file(patterns, base_dirs):
    """Find file matching any of the patterns in any of the base directories."""
    for base_dir in base_dirs:
        for pattern in patterns:
            matches = glob.glob(os.path.join(base_dir, pattern))
            if matches:
                return matches[0]
    return None

def load_product_descriptions():
    """Load product descriptions from available data sources."""
    possible_dirs = [
        os.path.join(parent_dir, "data"),
        os.path.join(grandparent_dir, "data"),
        os.path.join(os.path.abspath("."), "data"),
        os.path.join(os.path.dirname(os.path.abspath(".")), "data")
    ]
    
    patterns = ["processed_transactions.csv", "transactions*.csv", "prepared_products.csv"]
    
    data_file = find_file(patterns, possible_dirs)
    if not data_file:
        logger.warning(f"No product data file found in any of: {possible_dirs}")
        # Create a minimal empty dataframe with required columns
        return pd.DataFrame(columns=["ProductCode", "Description", "ProductName"])
    
    logger.info(f"Loading product descriptions from {data_file}")
    try:
        df = pd.read_csv(data_file)
        # Standardize column names
        if "product_code" in df.columns:
            df = df.rename(columns={"product_code": "ProductCode"})
        if "description" in df.columns:
            df = df.rename(columns={"description": "Description"})
        if "product_name" in df.columns:
            df = df.rename(columns={"product_name": "ProductName"})
            
        # Ensure required columns exist
        for col in ["ProductCode", "Description"]:
            if col not in df.columns:
                df[col] = ""
                
        if "ProductName" not in df.columns and "Description" in df.columns:
            df["ProductName"] = df["Description"].apply(
                lambda x: x.split(" - ")[0] if isinstance(x, str) and " - " in x else x
            )
            
        logger.info(f"Loaded {len(df)} product descriptions")
        return df
    except Exception as e:
        logger.error(f"Error loading product data: {e}")
        return pd.DataFrame(columns=["ProductCode", "Description", "ProductName"])

def collect_hierarchical_clusters():
    """Collect hierarchical clustering results from all levels."""
    logger.info("Collecting hierarchical clustering results from all levels")
    
    # Base directory for hierarchical clustering
    hier_dir = os.path.join(os.path.abspath("."), "data", "hierarchical_clustering")
    if not os.path.exists(hier_dir):
        logger.error(f"Hierarchical clustering directory not found: {hier_dir}")
        return None
    
    # Collect clusters from all levels
    levels = {}
    level_dirs = sorted(glob.glob(os.path.join(hier_dir, "level_*")))
    
    if not level_dirs:
        logger.error(f"No level directories found in {hier_dir}")
        return None
    
    logger.info(f"Found {len(level_dirs)} level directories")
    
    for level_dir in level_dirs:
        level_name = os.path.basename(level_dir)
        level_num = int(level_name.split("_")[1])
        
        # Find cluster files in this level
        cluster_files = glob.glob(os.path.join(level_dir, "*.json"))
        cluster_files += glob.glob(os.path.join(level_dir, "*", "*.json"))
        
        if not cluster_files:
            logger.warning(f"No cluster files found for {level_name}")
            continue
            
        # Use the latest cluster file if multiple exist
        cluster_file = max(cluster_files, key=os.path.getmtime)
        logger.info(f"Using cluster file for {level_name}: {os.path.basename(cluster_file)}")
        
        try:
            with open(cluster_file, 'r') as f:
                clusters = json.load(f)
                
            # Store level data
            levels[level_num] = {
                "name": f"level_{level_num}",
                "clusters": clusters
            }
        except Exception as e:
            logger.error(f"Error loading clusters for {level_name}: {e}")
    
    if not levels:
        logger.error("No valid clustering levels found")
        return None
        
    # Build hierarchical structure
    hierarchical_data = {
        "levels": sorted(levels.keys()),
        "level_data": levels
    }
    
    return hierarchical_data

def map_products_to_clusters(hierarchical_data):
    """Map products to their clusters at each level."""
    if not hierarchical_data:
        return {}
        
    product_map = {}
    
    for level_num in hierarchical_data["levels"]:
        level_data = hierarchical_data["level_data"][level_num]
        level_name = level_data["name"]
        clusters = level_data["clusters"]
        
        # Process each cluster in this level
        for cluster_id, products in clusters.items():
            for product in products:
                if product not in product_map:
                    product_map[product] = {}
                    
                product_map[product][f"level_{level_num}"] = cluster_id
    
    return product_map

def generate_analysis_csv(product_map, product_details, output_path):
    """Generate CSV with hierarchical cluster assignments."""
    if not product_map:
        logger.error("No product to cluster mapping available")
        return
        
    # Create dataframe from product mapping
    rows = []
    for product, clusters in product_map.items():
        row = {"ProductCode": product}
        row.update(clusters)
        rows.append(row)
        
    cluster_df = pd.DataFrame(rows)
    
    # Merge with product details if available
    if not product_details.empty:
        logger.info(f"Merging cluster assignments with product details")
        # Ensure ProductCode is string type in both dataframes
        cluster_df["ProductCode"] = cluster_df["ProductCode"].astype(str)
        product_details["ProductCode"] = product_details["ProductCode"].astype(str)
        
        result_df = pd.merge(
            cluster_df, 
            product_details, 
            on="ProductCode", 
            how="left"
        )
    else:
        result_df = cluster_df
        
    # Save to CSV
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved analysis CSV to {output_path} with {len(result_df)} rows")
    
    # Print sample
    logger.info(f"Sample rows:")
    if len(result_df) > 0:
        logger.info("\n" + str(result_df.head(3)))
    
    return result_df

def main():
    """Main function to generate hierarchical clustering analysis."""
    # Set up paths
    output_dir = os.path.join(os.path.abspath("."), "data", "hierarchical_clustering", "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hierarchical_cluster_analysis.csv")
    
    # Load product details
    product_details = load_product_descriptions()
    
    # Collect hierarchical clusters
    hierarchical_data = collect_hierarchical_clusters()
    
    if not hierarchical_data:
        logger.error("Failed to collect hierarchical clustering data")
        return
        
    # Save hierarchical data for reference
    hier_json_path = os.path.join(output_dir, "hierarchical_clusters_combined.json")
    with open(hier_json_path, 'w') as f:
        json.dump(hierarchical_data, f, indent=2)
    logger.info(f"Saved combined hierarchical data to {hier_json_path}")
    
    # Map products to clusters
    product_map = map_products_to_clusters(hierarchical_data)
    logger.info(f"Mapped {len(product_map)} products to their clusters")
    
    # Generate analysis CSV
    generate_analysis_csv(product_map, product_details, output_path)
    
    logger.info("Hierarchical clustering analysis completed")

if __name__ == "__main__":
    main()
