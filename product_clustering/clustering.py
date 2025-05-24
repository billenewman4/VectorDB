"""
Clustering module for product clustering.
Implements HDBSCAN clustering for grouping similar products.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import hdbscan
from sklearn.metrics import pairwise_distances
from collections import Counter

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data(embeddings_path: str, product_codes_path: str, prepared_data_path: Optional[str] = None) -> Tuple:
    """
    Load embeddings, product codes, and optionally prepared data.
    
    Args:
        embeddings_path: Path to embeddings file
        product_codes_path: Path to product codes file
        prepared_data_path: Optional path to prepared data CSV
        
    Returns:
        Tuple of (embeddings, product_codes, prepared_data)
    """
    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    # Load product codes
    print(f"Loading product codes from {product_codes_path}")
    with open(product_codes_path, 'r') as f:
        product_codes = [line.strip() for line in f]
    
    # Load prepared data if provided
    prepared_data = None
    if prepared_data_path and os.path.exists(prepared_data_path):
        print(f"Loading prepared data from {prepared_data_path}")
        prepared_data = pd.read_csv(prepared_data_path)
    
    print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    print(f"Loaded {len(product_codes)} product codes")
    
    return embeddings, product_codes, prepared_data

def run_hdbscan(embeddings: np.ndarray, 
               min_cluster_size: int = 5, 
               min_samples: int = 3,
               metric: str = 'euclidean') -> hdbscan.HDBSCAN:
    """
    Run HDBSCAN clustering on embeddings.
    
    Args:
        embeddings: NumPy array of embeddings
        min_cluster_size: Minimum size of clusters
        min_samples: HDBSCAN min_samples parameter
        metric: Distance metric to use
        
    Returns:
        HDBSCAN object with clustering results
    """
    print(f"Running HDBSCAN clustering with parameters:")
    print(f"  - min_cluster_size: {min_cluster_size}")
    print(f"  - min_samples: {min_samples}")
    print(f"  - metric: {metric}")
    
    # Initialize and fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1,  # Use all available cores
        cluster_selection_method='eom'  # 'eom' = Excess of Mass, usually gives better results
    )
    
    clusterer.fit(embeddings)
    
    return clusterer


def analyze_clusters(clusterer: hdbscan.HDBSCAN, product_codes: List[str], prepared_data: Optional[pd.DataFrame] = None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Analyze clustering results.
    
    Args:
        clusterer: HDBSCAN object with fit results
        product_codes: List of product codes
        prepared_data: Optional DataFrame with prepared data
        
    Returns:
        Tuple of (clusters_dict, cluster_df, cluster_stats_df)
    """
    labels = clusterer.labels_
    
    # Count number of clusters and outliers
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    
    print(f"HDBSCAN Results:")
    print(f"  - Number of clusters: {n_clusters}")
    print(f"  - Number of outliers: {n_outliers} ({n_outliers/len(labels):.2%})")
    
    # Create clusters dictionary
    clusters_dict = {}
    for i, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[int(label)] = []
        
        clusters_dict[int(label)].append(product_codes[i])
    
    # Create a DataFrame with cluster assignments
    cluster_data = []
    for i, (product_code, label) in enumerate(zip(product_codes, labels)):
        cluster_info = {
            'product_code': product_code,
            'cluster': label,
            'cluster_id': f"cluster_{label}" if label >= 0 else "unclustered",
        }
        
        # Add probability scores if available
        if hasattr(clusterer, 'probabilities_') and clusterer.probabilities_ is not None:
            cluster_info['probability'] = clusterer.probabilities_[i]
        
        # Add additional info from prepared_data if available
        if prepared_data is not None:
            product_row = prepared_data[prepared_data['product_code'] == product_code]
            if not product_row.empty:
                cluster_info['product_description'] = product_row['product_description'].iloc[0]
                if 'brand' in product_row.columns:
                    cluster_info['brand'] = product_row['brand'].iloc[0]
                if 'size' in product_row.columns:
                    cluster_info['size'] = product_row['size'].iloc[0]
        
        cluster_data.append(cluster_info)
    
    cluster_df = pd.DataFrame(cluster_data)
    
    # Calculate cluster statistics
    cluster_stats = []
    for label in sorted(set(labels)):
        stats = {
            'cluster': label,
            'cluster_id': f"cluster_{label}" if label >= 0 else "unclustered",
            'size': list(labels).count(label),
            'percentage': list(labels).count(label) / len(labels)
        }
        
        cluster_stats.append(stats)
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    # Display summary statistics
    valid_clusters = cluster_stats_df[cluster_stats_df['cluster'] >= 0]
    if not valid_clusters.empty:
        print("\nCluster Size Statistics:")
        print(f"  - Min cluster size: {valid_clusters['size'].min()}")
        print(f"  - Max cluster size: {valid_clusters['size'].max()}")
        print(f"  - Median cluster size: {valid_clusters['size'].median()}")
        print(f"  - Mean cluster size: {valid_clusters['size'].mean():.2f}")
    
    return clusters_dict, cluster_df, cluster_stats_df


def export_results(clusters_dict: dict, 
                  cluster_df: pd.DataFrame, 
                  cluster_stats_df: pd.DataFrame,
                  output_dir: str):
    """
    Export clustering results to files.
    
    Args:
        clusters_dict: Dictionary mapping cluster IDs to lists of product codes
        cluster_df: DataFrame with cluster assignments
        cluster_stats_df: DataFrame with cluster statistics
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export clusters.json (excluding the -1 noise cluster)
    cleaned_clusters = {k: v for k, v in clusters_dict.items() if k >= 0}
    # Convert numeric keys to string keys for JSON
    clusters_json = {f"cluster_{k}": v for k, v in cleaned_clusters.items()}
    
    clusters_path = os.path.join(output_dir, "clusters.json")
    with open(clusters_path, 'w') as f:
        json.dump(clusters_json, f, indent=2)
    
    print(f"Saved {len(clusters_json)} clusters to {clusters_path}")
    
    # Export unclustered.csv
    unclustered_path = os.path.join(output_dir, "unclustered.csv")
    unclustered_df = cluster_df[cluster_df['cluster'] == -1]
    unclustered_df.to_csv(unclustered_path, index=False)
    
    print(f"Saved {len(unclustered_df)} unclustered products to {unclustered_path}")
    
    # Export all cluster assignments
    assignments_path = os.path.join(output_dir, "cluster_assignments.csv")
    cluster_df.to_csv(assignments_path, index=False)
    print(f"Saved all cluster assignments to {assignments_path}")
    
    # Export cluster_stats.csv
    cluster_stats_path = os.path.join(output_dir, "cluster_stats.csv")
    cluster_stats_df.to_csv(cluster_stats_path, index=False)
    
    print(f"Saved cluster statistics to {cluster_stats_path}")
    
    # Generate cluster size histogram
    plt.figure(figsize=(12, 6))
    valid_clusters = cluster_stats_df[cluster_stats_df['cluster'] >= 0]
    if not valid_clusters.empty:
        plt.hist(valid_clusters['size'], bins=30)
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster Size')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Save histogram
        hist_path = os.path.join(output_dir, "cluster_size_hist.png")
        plt.savefig(hist_path)
        print(f"Saved cluster size histogram to {hist_path}")
        plt.close()


def sample_clusters(cluster_df: pd.DataFrame, 
                   sample_size: int = 5, 
                   sample_interval: int = 30,
                   output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Sample products from clusters for evaluation.
    
    Args:
        cluster_df: DataFrame with cluster assignments
        sample_size: Number of products to sample from each selected cluster
        sample_interval: Interval for selecting clusters to sample
        output_path: Optional path to save samples
        
    Returns:
        DataFrame with sampled products
    """
    # Get unique cluster IDs (excluding unclustered)
    unique_clusters = sorted(cluster_df[cluster_df['cluster'] >= 0]['cluster'].unique())
    
    if len(unique_clusters) == 0:
        print("No clusters to sample from")
        return pd.DataFrame()
    
    # Select clusters at regular intervals
    if sample_interval >= len(unique_clusters):
        # If interval is too large, just take a few
        selected_clusters = unique_clusters[:3]
    else:
        selected_clusters = unique_clusters[::sample_interval]
    
    print(f"Sampling {sample_size} products from {len(selected_clusters)} clusters")
    
    # Sample products from each selected cluster
    samples = []
    for cluster in selected_clusters:
        cluster_products = cluster_df[cluster_df['cluster'] == cluster]
        
        # If cluster has fewer products than sample_size, take all of them
        if len(cluster_products) <= sample_size:
            samples.append(cluster_products)
        else:
            samples.append(cluster_products.sample(sample_size, random_state=42))
    
    # Combine samples into a single DataFrame
    if samples:
        sample_df = pd.concat(samples, ignore_index=True)
        
        # Save samples if output path provided
        if output_path:
            sample_df.to_csv(output_path, index=False)
            print(f"Saved {len(sample_df)} samples to {output_path}")
        
        return sample_df
    
    return pd.DataFrame()


def main(embeddings_path: Optional[str] = None,
        product_codes_path: Optional[str] = None,
        prepared_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        metric: str = 'euclidean',
        sample_size: int = 5,
        sample_interval: int = 30):
    """
    Main function to run product clustering.
    
    Args:
        embeddings_path: Path to embeddings file
        product_codes_path: Path to product codes file
        prepared_data_path: Path to prepared data CSV
        output_dir: Directory to save output files
        min_cluster_size: Minimum size of clusters
        min_samples: HDBSCAN min_samples parameter
        metric: Distance metric to use
        sample_size: Number of products to sample from each selected cluster
        sample_interval: Interval for selecting clusters to sample
    """
    # Set default paths if not provided
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    if embeddings_path is None:
        embeddings_path = os.path.join(output_dir, "product_embeddings.npy")
    
    if product_codes_path is None:
        product_codes_path = os.path.join(output_dir, "product_codes.txt")
    
    if prepared_data_path is None:
        prepared_data_path = os.path.join(output_dir, "prepared_products.csv")
    
    # Load data
    embeddings, product_codes, prepared_data = load_data(
        embeddings_path, product_codes_path, prepared_data_path
    )
    
    # Run HDBSCAN clustering
    clusterer = run_hdbscan(
        embeddings, 
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric
    )
    
    # Analyze clustering results
    clusters_dict, cluster_df, cluster_stats_df = analyze_clusters(
        clusterer, product_codes, prepared_data
    )
    
    # Export results
    export_results(
        clusters_dict, cluster_df, cluster_stats_df, output_dir
    )
    
    # Sample clusters for evaluation
    sample_df = sample_clusters(
        cluster_df,
        sample_size=sample_size,
        sample_interval=sample_interval,
        output_path=os.path.join(output_dir, "cluster_samples.csv")
    )
    
    print(f"Clustering complete. Saved results to {output_dir}")
    
    return clusters_dict, cluster_df, cluster_stats_df, sample_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Product Clustering")
    parser.add_argument("--embeddings", help="Path to embeddings file")
    parser.add_argument("--product_codes", help="Path to product codes file")
    parser.add_argument("--prepared_data", help="Path to prepared data CSV")
    parser.add_argument("--output", help="Directory to save output files")
    parser.add_argument("--min_cluster_size", type=int, default=5, help="Minimum size of clusters")
    parser.add_argument("--min_samples", type=int, default=3, help="HDBSCAN min_samples parameter")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"], help="Distance metric to use")
    parser.add_argument("--sample_size", type=int, default=5, help="Number of products to sample from each selected cluster")
    parser.add_argument("--sample_interval", type=int, default=30, help="Interval for selecting clusters to sample")
    
    args = parser.parse_args()
    
    main(
        embeddings_path=args.embeddings,
        product_codes_path=args.product_codes,
        prepared_data_path=args.prepared_data,
        output_dir=args.output,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        sample_size=args.sample_size,
        sample_interval=args.sample_interval
    )
