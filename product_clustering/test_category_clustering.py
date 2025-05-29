#!/usr/bin/env python3
"""
Test script for the category-based hierarchical clustering implementation.
This script runs a small test to verify the category-based clustering functionality.
"""
import os
import sys
import pandas as pd
from pathlib import Path
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
# Import configuration
from src import config

from data_prep.processor import prepare_unified_product_data
from data_prep.category_filter import filter_products_by_category, normalize_category_names, group_products_by_category
from product_clustering.category_clustering import run_category_clustering

def run_test(test_size=None):
    """
    Run a test of the category-based clustering.
    
    Args:
        test_size: Optional number of products per category to use for testing
    """
    print("=== Testing Category-Based Hierarchical Clustering ===")
    start_time = time.time()
    
    # Use test size from config if not provided
    if test_size is None:
        test_size = config.TEST_SAMPLE_SIZE
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_category_clustering")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get prepared data
    print("Preparing unified product data...")
    df = prepare_unified_product_data()
    
    # For testing, always use a subset of the data
    print(f"Running in TEST MODE with up to {test_size} products per category")
    # Filter to only include products with categories
    filtered_df = filter_products_by_category(df)
    normalized_df = normalize_category_names(filtered_df)
    category_groups = group_products_by_category(normalized_df)
    
    # Take a subset from each category
    test_df_parts = []
    for category, group_df in category_groups.items():
        if len(group_df) > test_size:
            test_df_parts.append(group_df.sample(n=test_size, random_state=42))
        else:
            test_df_parts.append(group_df)
    
    # Combine the subsets
    if test_df_parts:
        df = pd.concat(test_df_parts, ignore_index=True)
        print(f"Created test dataset with {len(df)} products from {len(category_groups)} categories")
    
    # Save the test data to a CSV
    test_data_path = os.path.join(output_dir, "test_products.csv")
    df.to_csv(test_data_path, index=False)
    print(f"Saved test data to {test_data_path}")
    
    # Run category-based clustering with the smaller embedding model for testing
    print("\nRunning category-based clustering...")
    clusters_path = run_category_clustering(
        prepared_data_path=test_data_path,
        output_dir=output_dir,
        model_name=config.SENTENCE_TRANSFORMER_MODEL_TESTING,  # Use smaller model for testing
        metric=config.CLUSTERING_METRIC,
        min_cluster_size=config.MIN_CLUSTER_SIZE,
        min_samples=config.MIN_SAMPLES,
        use_reranking=config.USE_RERANKING,
        cross_encoder_model=config.CROSS_ENCODER_MODEL,
        similarity_threshold=config.SIMILARITY_THRESHOLD
    )
    
    # Calculate execution time
    execution_time = time.time() - start_time
    minutes, seconds = divmod(execution_time, 60)
    
    print(f"\nTest completed in {int(minutes)}m {seconds:.1f}s.")
    print(f"Results saved to {clusters_path}")
    print("=== End of Test ===")
    
    return clusters_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test category-based clustering")
    parser.add_argument("--test_size", type=int, help="Number of products per category to use for testing")
    
    args = parser.parse_args()
    
    run_test(args.test_size)
