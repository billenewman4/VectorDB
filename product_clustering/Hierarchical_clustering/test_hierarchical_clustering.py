#!/usr/bin/env python3
"""
Test script for the hierarchical clustering implementation.
This script runs the hierarchical clustering on a small subset of data
to verify the implementation works correctly.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import time
import shutil

# Add parent directories to path to import from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# Import the hierarchical clustering module
from Hierarchical_clustering.hierarchical_clustering import HierarchicalClusterer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_hierarchical_clustering")


def create_test_data(data_dir: str, full_dataset: bool = True, sample_size: int = 1000):
    """
    Create a dataset for testing the hierarchical clustering.
    
    Args:
        data_dir: Directory containing the original data
        full_dataset: Whether to use the full dataset (True) or a sample (False)
        
    Returns:
        Path to the test data directory
    """
    # Create test data directory within the Hierarchical_clustering folder
    hierarchical_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(hierarchical_dir, "data")
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Load original embeddings and product codes
    original_embeddings_path = os.path.join(data_dir, "product_embeddings.npy")
    original_product_codes_path = os.path.join(data_dir, "product_codes.txt")
    
    logger.info(f"Loading original embeddings from {original_embeddings_path}")
    original_embeddings = np.load(original_embeddings_path)
    
    logger.info(f"Loading original product codes from {original_product_codes_path}")
    with open(original_product_codes_path, 'r') as f:
        original_product_codes = [line.strip() for line in f.readlines()]
    
    if full_dataset:
        # Use the full dataset
        logger.info(f"Using full dataset with {len(original_product_codes)} products")
        data_embeddings = original_embeddings
        data_product_codes = original_product_codes
    else:
        # Sample a subset for faster testing
        if sample_size > len(original_product_codes):
            sample_size = len(original_product_codes)
            logger.warning(f"Sample size exceeds number of products. Using all {sample_size} products.")
        
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(original_product_codes), sample_size, replace=False)
        
        # Extract sampled embeddings and product codes
        data_embeddings = original_embeddings[sample_indices]
        data_product_codes = [original_product_codes[i] for i in sample_indices]
        logger.info(f"Created sample dataset with {sample_size} products")
    
    # Save data
    data_embeddings_path = os.path.join(test_data_dir, "product_embeddings.npy")
    data_product_codes_path = os.path.join(test_data_dir, "product_codes.txt")
    
    np.save(data_embeddings_path, data_embeddings)
    with open(data_product_codes_path, 'w') as f:
        for code in data_product_codes:
            f.write(f"{code}\n")
    
    # Copy prepared_products.csv if it exists
    original_prepared_data_path = os.path.join(data_dir, "prepared_products.csv")
    if os.path.exists(original_prepared_data_path):
        logger.info(f"Copying prepared products data from {original_prepared_data_path}")
        prepared_data = pd.read_csv(original_prepared_data_path)
        
        if not full_dataset:
            # Filter to only include the selected product codes
            prepared_data = prepared_data[prepared_data['product_code'].isin(data_product_codes)]
        
        # Save prepared data
        prepared_data_path = os.path.join(test_data_dir, "prepared_products.csv")
        prepared_data.to_csv(prepared_data_path, index=False)
    
    logger.info(f"Prepared test data in {test_data_dir}")
    
    # Create subdirectories for outputs
    os.makedirs(os.path.join(test_data_dir, "hierarchical_clustering"), exist_ok=True)
    os.makedirs(os.path.join(test_data_dir, "analysis"), exist_ok=True)
    return test_data_dir


def create_run_config(config_path: str, run_config_path: str, full_dataset: bool = True):
    """
    Create a configuration for running hierarchical clustering.
    
    Args:
        config_path: Path to the original configuration file
        run_config_path: Path to save the run configuration
        full_dataset: Whether using the full dataset (True) or a sample (False)
    """
    # For speed, directly use the existing run_config.json file if it exists
    if os.path.exists(run_config_path):
        logger.info(f"Using existing run configuration at {run_config_path}")
        return
        
    # Otherwise, load original config and create a new run config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set appropriate parameters based on dataset size
    if not full_dataset:
        # For smaller test datasets, adjust parameters
        for level in config["levels"]:
            if level["level"] == 1:
                # Make level 1 very inclusive for small datasets
                level["min_cluster_size"] = 2
                level["min_samples"] = 1
                level["epsilon"] = 5.0
                level["alpha"] = 0.05
            else:
                # Make subsequent levels more inclusive for small datasets
                level["min_cluster_size"] = 2
                level["min_samples"] = 1
                
        # Adjust progression rules for smaller datasets
        config["progression_rules"]["min_products_to_proceed"] = 3
    
    # Force re-clustering
    config["global_settings"]["force"] = True
    
    # Save run config
    with open(run_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created run configuration at {run_config_path}")


def print_hierarchy_summary(hierarchical_clusters: Dict[str, Any]):
    """
    Print a summary of the hierarchical clustering results.
    
    Args:
        hierarchical_clusters: The hierarchical clustering results
    """
    logger.info("=== Hierarchical Clustering Results ===")
    
    total_clusters = 0
    total_products_covered = set()
    
    for level_name in sorted(hierarchical_clusters.keys()):
        clusters = hierarchical_clusters[level_name]
        num_clusters = len(clusters)
        total_clusters += num_clusters
        
        # Count products at this level
        level_products = set()
        for cluster_info in clusters.values():
            level_products.update(cluster_info.get("products", []))
            
        # Count products with children
        clusters_with_children = sum(1 for c in clusters.values() if c.get("children"))
        avg_children = sum(len(c.get("children", [])) for c in clusters.values()) / max(1, clusters_with_children) if clusters_with_children else 0
        
        logger.info(f"{level_name}: {num_clusters} clusters, {len(level_products)} products")
        logger.info(f"  - {clusters_with_children} clusters have children (avg {avg_children:.1f} children)")
        
        # Calculate inclusion rate
        inclusion_rate = len(level_products) / 500  # Test size is 500
        logger.info(f"  - Inclusion rate: {inclusion_rate:.1%}")
        
        # Update total products covered
        total_products_covered.update(level_products)
    
    logger.info(f"Total unique products covered: {len(total_products_covered)}")
    logger.info(f"Total clusters across all levels: {total_clusters}")


def print_sample_cluster_path(hierarchical_clusters: Dict[str, Any], sample_product: Optional[str] = None):
    """
    Print a sample path from the highest level to the lowest for a product.
    
    Args:
        hierarchical_clusters: The hierarchical clustering results
        sample_product: A specific product to trace. If None, one will be chosen randomly.
    """
    # Find a product that appears in the lowest level
    lowest_level = max(int(level_name.split('_')[1]) for level_name in hierarchical_clusters.keys())
    lowest_level_name = f"level_{lowest_level}"
    
    if not hierarchical_clusters.get(lowest_level_name):
        logger.warning(f"No clusters found at level {lowest_level}")
        return
    
    # If no sample product specified, find one that's in the lowest level
    if not sample_product:
        for cluster_info in hierarchical_clusters[lowest_level_name].values():
            if cluster_info.get("products"):
                sample_product = cluster_info["products"][0]
                break
    
    if not sample_product:
        logger.warning("No products found to trace through the hierarchy")
        return
    
    logger.info(f"\n=== Tracing product '{sample_product}' through the hierarchy ===")
    
    # Find the cluster path for this product
    cluster_path = []
    
    for level_num in range(lowest_level, 0, -1):
        level_name = f"level_{level_num}"
        
        # Find cluster containing this product at this level
        for cluster_id, cluster_info in hierarchical_clusters[level_name].items():
            if sample_product in cluster_info.get("products", []):
                cluster_path.insert(0, (level_name, cluster_id, len(cluster_info.get("products", []))))
                break
    
    # Print the path
    for level_name, cluster_id, size in cluster_path:
        logger.info(f"{level_name}: Cluster {cluster_id} (size: {size} products)")


def run_analysis(hierarchical_clusters: Dict[str, Any], test_data_dir: str):
    """
    Run the hierarchical cluster analysis and generate the CSV output.
    
    Args:
        hierarchical_clusters: The hierarchical clustering results
        test_data_dir: Directory containing the test data
    """
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from analyze_hierarchical_clusters import analyze_hierarchical_clusters
    
    # Save hierarchical clusters to a temporary file
    temp_clusters_path = os.path.join(test_data_dir, "hierarchical_clusters.json")
    with open(temp_clusters_path, 'w') as f:
        json.dump(hierarchical_clusters, f, indent=2)
    
    # Run analysis
    output_dir = os.path.join(test_data_dir, "analysis")
    analyze_hierarchical_clusters(temp_clusters_path, test_data_dir, output_dir)
    
    # Check if analysis CSV was created
    analysis_csv_path = os.path.join(output_dir, "hierarchical_cluster_analysis.csv")
    if os.path.exists(analysis_csv_path):
        logger.info(f"Analysis CSV created at {analysis_csv_path}")
        # Load and print sample rows
        analysis_df = pd.read_csv(analysis_csv_path)
        logger.info(f"Analysis contains {len(analysis_df)} rows and {len(analysis_df.columns)} columns")
        logger.info("Sample columns: " + ", ".join(analysis_df.columns[:10]))
        logger.info("\nSample rows:")
        logger.info("\n" + str(analysis_df.head(3)))
    else:
        logger.error(f"Analysis CSV not created at {analysis_csv_path}")


def run_hierarchical_clustering(full_dataset: bool = True, sample_size: int = 1000):
    """Run the hierarchical clustering on the specified dataset.
    
    Args:
        full_dataset: Whether to use the full dataset (True) or a sample (False)
        sample_size: Number of products to sample if full_dataset is False
    """
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the run config - use this directly without modification
    run_config_path = os.path.join(current_dir, "run_config.json")
    
    # Data directory (parent product_clustering/data)
    source_data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    # Create/prepare data directory
    data_dir = create_test_data(source_data_dir, full_dataset, sample_size)
    
    try:
        # Run hierarchical clustering
        logger.info(f"Running hierarchical clustering on {'full' if full_dataset else 'sample'} dataset with {sample_size if not full_dataset else 'all'} products...")
        clusterer = HierarchicalClusterer(run_config_path, data_dir)
        results = clusterer.run_hierarchical_clustering()
        
        # Print results summary
        print_hierarchy_summary(results)
        
        # Run analysis to generate CSV with cluster assignments
        run_analysis(results, data_dir)
        
        # Print sample path for a random product
        print_sample_cluster_path(results)
        
        return results
    except Exception as e:
        logger.error(f"Error during hierarchical clustering: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hierarchical clustering on product data")
    parser.add_argument("--sample", action="store_true", 
                      help="Use a sample dataset instead of the full dataset")
    parser.add_argument("--sample-size", type=int, default=1000,
                      help="Number of products to sample if using --sample (default: 1000)")
    args = parser.parse_args()
    
    # Run with either full dataset or sample based on command line arguments
    run_hierarchical_clustering(full_dataset=not args.sample, sample_size=args.sample_size)
