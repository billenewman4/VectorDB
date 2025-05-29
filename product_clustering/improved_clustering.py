"""
Improved clustering module with optimized parameters.
"""
import os
import sys
import numpy as np
from typing import Optional

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required libraries
import pandas as pd
import hdbscan
import json
from collections import defaultdict

def run_improved_clustering(data_dir: Optional[str] = None,
                           metric: str = 'euclidean',  # Using euclidean instead of cosine as HDBSCAN doesn't support cosine directly
                           min_cluster_size: int = 3,  # Lower to allow more granular clusters
                           min_samples: int = 2,  # Lower to allow more granular clusters
                           test_mode: bool = False,
                           sample_size: int = 1000,  # Sample size to use in test mode
                           use_reranking: bool = False,  # Whether to use CrossEncoder reranking
                           cross_encoder_model: str = 'cross-encoder/stsb-roberta-base',  # Model for reranking
                           similarity_threshold: float = 0.5,  # Threshold for reranking
                           use_categories: bool = True,  # Whether to cluster by category first
                           force: bool = False):  # Whether to force regeneration of embeddings/clusters
    """
    Run clustering with improved parameters.
    
    Args:
        data_dir: Directory containing data files
        metric: Distance metric to use
        min_cluster_size: Minimum size of clusters
        min_samples: HDBSCAN min_samples parameter
    """
    # Set default data directory
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    # Paths to data files
    embeddings_path = os.path.join(data_dir, "product_embeddings.npy")
    product_codes_path = os.path.join(data_dir, "product_codes.txt")
    prepared_data_path = os.path.join(data_dir, "prepared_products.csv")
    
    # If test mode is enabled, use a subset of the data
    subset_suffix = ""
    if test_mode:
        print(f"Running in TEST MODE with {sample_size} samples")
        subset_suffix = "_subset"
        
        # Load the original data
        embeddings = np.load(embeddings_path)
        with open(product_codes_path, 'r') as f:
            product_codes = [line.strip() for line in f]
        
        # Select a random subset
        np.random.seed(42)  # For reproducibility
        if sample_size >= len(embeddings):
            print(f"Sample size {sample_size} is larger than dataset size {len(embeddings)}.")
            print("Using entire dataset.")
        else:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[indices]
            product_codes = [product_codes[i] for i in indices]
            
            # Save the subset for clustering
            subset_embeddings_path = os.path.join(data_dir, f"product_embeddings{subset_suffix}.npy")
            subset_codes_path = os.path.join(data_dir, f"product_codes{subset_suffix}.txt")
            
            np.save(subset_embeddings_path, embeddings)
            with open(subset_codes_path, 'w') as f:
                for code in product_codes:
                    f.write(f"{code}\n")
            
            # Update paths to use the subset
            embeddings_path = subset_embeddings_path
            product_codes_path = subset_codes_path
    
    # Output directory for improved clustering
    output_dir = os.path.join(data_dir, f"improved_clustering{subset_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running improved clustering with parameters:")
    print(f"  - metric: {metric}")
    print(f"  - min_cluster_size: {min_cluster_size}")
    print(f"  - min_samples: {min_samples}")
    if test_mode:
        print(f"  - test_mode: {test_mode} (using {sample_size} samples)")
    print(f"Output will be saved to: {output_dir}")
    
    # Implement clustering directly instead of calling base_clustering
    
    # Load embeddings and product codes
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    print(f"Loading product codes from {product_codes_path}")
    with open(product_codes_path, 'r') as f:
        product_codes = [line.strip() for line in f.readlines()]
    
    if len(embeddings) != len(product_codes):
        print(f"Warning: Mismatch between embeddings ({len(embeddings)}) and product codes ({len(product_codes)})")
    
    # Check for category information to enable category-based clustering
    category_products_path = os.path.join(data_dir, "category_products.json")
    category_mode = use_categories and os.path.exists(category_products_path)
    
    # Main clustering approach
    if category_mode:
        print("Using category-based clustering approach")
        print(f"Loading category products from {category_products_path}")
        
        # Load category-to-products mapping
        try:
            with open(category_products_path, 'r') as f:
                category_products = json.load(f)
            print(f"Loaded {len(category_products)} product categories")
            
            # Load product code to embedding mapping
            product_to_embedding = {}
            for i, code in enumerate(product_codes):
                if i < len(embeddings):
                    product_to_embedding[code] = embeddings[i]
            
            # Perform clustering within each category separately
            clusters = defaultdict(list)
            cluster_counter = 0
            category_stats = {}
            
            for category, products in category_products.items():
                # Filter out products not in our embeddings
                category_products = [p for p in products if p in product_to_embedding]
                
                if len(category_products) < min_cluster_size:
                    print(f"Skipping category '{category}': too few products ({len(category_products)})")
                    continue
                    
                print(f"Clustering category '{category}' with {len(category_products)} products")
                
                # Extract embeddings for this category
                category_embeddings = np.array([product_to_embedding[p] for p in category_products])
                
                # Run HDBSCAN clustering on this category
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric=metric,
                    gen_min_span_tree=True,
                    cluster_selection_method='eom'
                )
                
                # Fit the clusterer
                category_labels = clusterer.fit_predict(category_embeddings)
                
                # Count clusters and noise points for this category
                cat_clusters = len(set(category_labels)) - (1 if -1 in category_labels else 0)
                cat_noise = list(category_labels).count(-1)
                
                print(f"  - {cat_clusters} clusters formed with {cat_noise} noise points")
                category_stats[category] = {
                    "total_products": len(category_products),
                    "clusters": cat_clusters,
                    "noise": cat_noise
                }
                
                # Organize products into clusters
                for i, label in enumerate(category_labels):
                    if label >= 0:
                        # Use category prefix for cluster names to keep them separated
                        cluster_id = f"cluster_{category}_{cluster_counter + label}"
                        clusters[cluster_id].append(category_products[i])
                
                # Update the counter for the next category
                cluster_counter += max(category_labels) + 1 if len(category_labels) > 0 and max(category_labels) >= 0 else 0
            
            # Print category statistics
            print("\nCategory clustering statistics:")
            for category, stats in category_stats.items():
                print(f"  - {category}: {stats['clusters']} clusters, {stats['noise']} noise points")
                
        except Exception as e:
            print(f"Error in category-based clustering: {e}")
            print("Falling back to standard clustering")
            category_mode = False
    
    # Standard clustering if not using categories or if category clustering failed
    if not category_mode:
        print(f"Running HDBSCAN clustering on {len(embeddings)} products...")
        
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            gen_min_span_tree=True,
            cluster_selection_method='eom'
        )
        
        # Fit the clusterer
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Count clusters and noise points
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Clustering complete: {n_clusters} clusters formed with {n_noise} noise points")
        
        # Organize products into clusters
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if i < len(product_codes):
                if label >= 0:
                    clusters[f"cluster_{label}"].append(product_codes[i])
    
    # Save clusters to file
    clusters_path = os.path.join(output_dir, "clusters.json")
    with open(clusters_path, 'w') as f:
        json.dump(clusters, f, indent=2)
    
    print(f"Saved {len(clusters)} clusters to {clusters_path}")
    
    # Calculate basic statistics
    total_clustered = sum(len(products) for products in clusters.values())
    coverage = total_clustered / len(product_codes) * 100 if product_codes else 0
    avg_size = total_clustered / len(clusters) if clusters else 0
    
    print(f"Clustering statistics:")
    print(f"  - Total products: {len(product_codes)}")
    print(f"  - Clustered products: {total_clustered} ({coverage:.1f}%)")
    print(f"  - Average cluster size: {avg_size:.1f} products")
    
    # Apply CrossEncoder reranking if requested
    if use_reranking:
        print("\nApplying CrossEncoder reranking to refine clusters...")
        from product_clustering.reranking import refine_clusters
        
        # Path to clusters.json produced by base clustering
        clusters_path = os.path.join(output_dir, "clusters.json")
        
        # Create a subdirectory for refined clusters
        refined_output_dir = os.path.join(output_dir, "refined")
        os.makedirs(refined_output_dir, exist_ok=True)
        
        # Run refinement
        refine_clusters(
            clusters_path=clusters_path,
            prepared_data_path=prepared_data_path,
            output_dir=refined_output_dir,
            model_name=cross_encoder_model,
            similarity_threshold=similarity_threshold
        )
        
        print(f"CrossEncoder refinement complete. Results saved to {refined_output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run improved clustering")
    parser.add_argument("--data_dir", help="Directory containing data files")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan"], 
                        help="Distance metric to use")
    parser.add_argument("--min_cluster_size", type=int, default=3, 
                        help="Minimum size of clusters")
    parser.add_argument("--min_samples", type=int, default=2, 
                        help="HDBSCAN min_samples parameter")
    parser.add_argument("--test", action="store_true", 
                        help="Run in test mode with a subset of data")
    parser.add_argument("--sample_size", type=int, default=1000, 
                        help="Number of samples to use in test mode")
    parser.add_argument("--rerank", action="store_true",
                        help="Use CrossEncoder reranking to refine clusters")
    parser.add_argument("--cross_encoder_model", default="cross-encoder/stsb-roberta-base",
                        help="CrossEncoder model to use for reranking")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Similarity threshold for CrossEncoder reranking")
    parser.add_argument("--use_categories", action="store_true",
                        help="Cluster products by category first (products from different categories will never be in the same cluster)")
    
    args = parser.parse_args()
    
    run_improved_clustering(
        data_dir=args.data_dir,
        metric=args.metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        test_mode=args.test,
        sample_size=args.sample_size,
        use_reranking=args.rerank,
        cross_encoder_model=args.cross_encoder_model,
        similarity_threshold=args.similarity_threshold
    )
