#!/usr/bin/env python3
"""
Analyze Refined Clusters

This script analyzes the refined clusters created by the CrossEncoder reranking process.
It loads the original product descriptions from the transaction data and displays
detailed information about each cluster.
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import argparse
from collections import defaultdict

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_refined_clusters(data_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Load refined clusters from JSON file.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary mapping cluster IDs to lists of product codes
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    refined_clusters_path = os.path.join(data_dir, "refined_clusters", "refined_clusters.json")
    if not os.path.exists(refined_clusters_path):
        print(f"Error: Refined clusters file not found at {refined_clusters_path}")
        return {}
    
    with open(refined_clusters_path, 'r') as f:
        clusters = json.load(f)
    
    print(f"Loaded {len(clusters)} refined clusters")
    return clusters

def load_product_descriptions() -> Dict[str, str]:
    """
    Load product descriptions from transaction data.
    
    Returns:
        Dictionary mapping product codes to descriptions
    """
    from src.data_processing import load_transaction_data
    
    # Try to load transaction data
    try:
        df = load_transaction_data()
        if df is None:
            print("Error: Failed to load transaction data")
            return {}
            
        print(f"Loaded transaction data with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Rename columns for consistency
        if 'product_code' not in df.columns and 'ProductCode' in df.columns:
            df = df.rename(columns={'ProductCode': 'product_code'})
        
        if 'description' not in df.columns:
            if 'Description' in df.columns:
                df = df.rename(columns={'Description': 'description'})
            elif 'ProductDescription' in df.columns:
                df = df.rename(columns={'ProductDescription': 'description'})
        
        # Extract unique product descriptions
        if 'product_code' in df.columns and 'description' in df.columns:
            products = df[['product_code', 'description']].drop_duplicates()
            print(f"Found {len(products)} unique products")
            
            # Convert product codes to strings
            products['product_code'] = products['product_code'].astype(str)
            
            # Create a dictionary for easy lookup
            product_dict = dict(zip(products['product_code'], products['description']))
            return product_dict
        else:
            print(f"Required columns not found. Available columns: {list(df.columns)}")
            return {}
            
    except Exception as e:
        print(f"Error loading transaction data: {str(e)}")
        
        # Fall back to searching for any CSV files with product info
        try:
            print("Searching for alternative product data sources...")
            import glob
            csv_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "**", "*.csv"), recursive=True)
            
            for csv_file in csv_files:
                try:
                    print(f"Trying {csv_file}...")
                    df = pd.read_csv(csv_file)
                    
                    # Check if this file has product information
                    if 'product_code' in df.columns and any(col for col in df.columns if 'description' in col.lower() or 'product' in col.lower()):
                        desc_col = next(col for col in df.columns if 'description' in col.lower() or 'product' in col.lower())
                        print(f"Found product info in {csv_file}")
                        
                        # Create a product dictionary
                        products = df[['product_code', desc_col]].drop_duplicates()
                        products['product_code'] = products['product_code'].astype(str)
                        product_dict = dict(zip(products['product_code'], products[desc_col]))
                        
                        print(f"Loaded {len(product_dict)} product descriptions")
                        return product_dict
                except Exception as inner_e:
                    continue
                    
            print("No suitable product data found in CSV files")
            return {}
        except Exception as fallback_e:
            print(f"Error in fallback method: {str(fallback_e)}")
            return {}

def analyze_cluster(cluster_id: str, product_codes: List[str], product_dict: Dict[str, str], 
                    show_all: bool = False, max_products: int = 10) -> None:
    """
    Analyze a single cluster and print its details.
    
    Args:
        cluster_id: ID of the cluster
        product_codes: List of product codes in the cluster
        product_dict: Dictionary mapping product codes to descriptions
        show_all: Whether to show all products in the cluster
        max_products: Maximum number of products to display if show_all is False
    """
    print(f"\n=== Cluster {cluster_id} ({len(product_codes)} products) ===")
    
    # Count how many products have descriptions
    described_products = sum(1 for code in product_codes if code in product_dict)
    print(f"Products with descriptions: {described_products}/{len(product_codes)} ({described_products/len(product_codes):.1%})")
    
    # Determine how many products to show
    display_count = len(product_codes) if show_all else min(max_products, len(product_codes))
    
    # Sort product codes that have descriptions first
    sorted_codes = sorted(product_codes, key=lambda c: c not in product_dict)
    
    # Display products
    for i, code in enumerate(sorted_codes[:display_count]):
        if code in product_dict:
            print(f"  {code}: {product_dict[code]}")
        else:
            print(f"  {code}: [No description available]")
    
    # Show message if not showing all products
    if not show_all and len(product_codes) > max_products:
        print(f"  ... and {len(product_codes) - max_products} more products")

def analyze_all_clusters(clusters: Dict[str, List[str]], product_dict: Dict[str, str], 
                        show_all: bool = False, max_clusters: Optional[int] = None,
                        min_size: int = 0, max_size: Optional[int] = None,
                        random_selection: bool = False, cluster_ids: Optional[List[str]] = None) -> None:
    """
    Analyze all clusters and print their details.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        product_dict: Dictionary mapping product codes to descriptions
        show_all: Whether to show all products in each cluster
        max_clusters: Maximum number of clusters to display
        min_size: Minimum cluster size to include
        max_size: Maximum cluster size to include
        random_selection: Whether to randomly select clusters
        cluster_ids: List of specific cluster IDs to analyze
    """
    # Filter clusters by size
    filtered_clusters = {
        cid: codes for cid, codes in clusters.items() 
        if len(codes) >= min_size and (max_size is None or len(codes) <= max_size)
    }
    
    print(f"\nAnalyzing clusters (filtered from {len(clusters)} to {len(filtered_clusters)} clusters)")
    
    # Determine which clusters to analyze
    if cluster_ids:
        # Use specified cluster IDs
        selected_clusters = {cid: filtered_clusters[cid] for cid in cluster_ids if cid in filtered_clusters}
        print(f"Analyzing {len(selected_clusters)} specified clusters")
    elif random_selection and max_clusters:
        # Randomly select clusters
        selected_ids = random.sample(list(filtered_clusters.keys()), min(max_clusters, len(filtered_clusters)))
        selected_clusters = {cid: filtered_clusters[cid] for cid in selected_ids}
        print(f"Randomly selected {len(selected_clusters)} clusters")
    else:
        # Use all filtered clusters, possibly limited by max_clusters
        selected_clusters = filtered_clusters
        if max_clusters:
            selected_ids = list(selected_clusters.keys())[:max_clusters]
            selected_clusters = {cid: selected_clusters[cid] for cid in selected_ids}
            print(f"Showing first {len(selected_clusters)} clusters")
    
    # Analyze each selected cluster
    for cluster_id, product_codes in selected_clusters.items():
        analyze_cluster(cluster_id, product_codes, product_dict, show_all)

def generate_cluster_statistics(clusters: Dict[str, List[str]], product_dict: Dict[str, str]) -> None:
    """
    Generate and display statistics about the clusters.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        product_dict: Dictionary mapping product codes to descriptions
    """
    # Calculate cluster sizes
    cluster_sizes = [len(codes) for codes in clusters.values()]
    
    print("\n=== Cluster Statistics ===")
    print(f"Total clusters: {len(clusters)}")
    print(f"Total products in clusters: {sum(cluster_sizes)}")
    print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")
    print(f"Median cluster size: {np.median(cluster_sizes):.1f}")
    print(f"Smallest cluster size: {min(cluster_sizes)}")
    print(f"Largest cluster size: {max(cluster_sizes)}")
    
    # Cluster size distribution
    size_bins = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    size_counts = {f"{low}-{high if high != float('inf') else '∞'}": sum(1 for s in cluster_sizes if low <= s <= high) 
                  for low, high in size_bins}
    
    print("\nCluster size distribution:")
    for size_range, count in size_counts.items():
        print(f"  {size_range}: {count} clusters ({count/len(clusters):.1%})")
    
    # Description coverage
    total_products = sum(cluster_sizes)
    products_with_desc = sum(sum(1 for code in codes if code in product_dict) for codes in clusters.values())
    
    print(f"\nDescription coverage: {products_with_desc}/{total_products} products ({products_with_desc/total_products:.1%})")
    
    # Identify clusters with mixed product types (if descriptions available)
    if product_dict:
        print("\nAnalyzing product type consistency within clusters...")
        # This is a simplified check - in real application you might want more sophisticated logic
        # to determine if products are of the same type

def save_cluster_details(clusters: Dict[str, List[str]], product_dict: Dict[str, str], 
                         output_file: str) -> None:
    """
    Save detailed cluster information to a CSV file.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        product_dict: Dictionary mapping product codes to descriptions
        output_file: Path to output CSV file
    """
    # Create a list of records for the CSV
    records = []
    for cluster_id, product_codes in clusters.items():
        for code in product_codes:
            description = product_dict.get(code, "[No description available]")
            records.append({
                'cluster_id': cluster_id,
                'product_code': code,
                'description': description,
                'cluster_size': len(product_codes)
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"\nSaved detailed cluster information to {output_file}")

def main():
    """Main function to run the cluster analysis."""
    parser = argparse.ArgumentParser(description="Analyze refined product clusters")
    parser.add_argument("--all", action="store_true", help="Show all products in each cluster")
    parser.add_argument("--max_clusters", type=int, default=10, help="Maximum number of clusters to display")
    parser.add_argument("--min_size", type=int, default=0, help="Minimum cluster size to include")
    parser.add_argument("--max_size", type=int, default=None, help="Maximum cluster size to include")
    parser.add_argument("--random", action="store_true", help="Randomly select clusters to display")
    parser.add_argument("--clusters", nargs="+", help="Specific cluster IDs to analyze (e.g., cluster_30)")
    parser.add_argument("--output", type=str, default=None, help="Save detailed cluster information to CSV file")
    parser.add_argument("--produce", action="store_true", help="Analyze the produce cluster specifically")
    
    args = parser.parse_args()
    
    # Load clusters and product descriptions
    clusters = load_refined_clusters()
    product_dict = load_product_descriptions()
    
    if not clusters:
        print("Error: No clusters found. Exiting.")
        return
    
    # Generate overall statistics
    generate_cluster_statistics(clusters, product_dict)
    
    # Analyze specific clusters if requested
    if args.produce and "cluster_30" in clusters:
        print("\n=== Analyzing Produce Cluster ===")
        analyze_cluster("cluster_30", clusters["cluster_30"], product_dict, show_all=True)
    
    # Analyze all clusters based on command-line arguments
    cluster_ids = args.clusters
    if args.produce:
        cluster_ids = ["cluster_30"] + (cluster_ids or [])
    
    analyze_all_clusters(
        clusters, 
        product_dict, 
        show_all=args.all,
        max_clusters=args.max_clusters,
        min_size=args.min_size,
        max_size=args.max_size,
        random_selection=args.random,
        cluster_ids=cluster_ids
    )
    
    # Save to CSV if requested
    if args.output:
        save_cluster_details(clusters, product_dict, args.output)

if __name__ == "__main__":
    main()
