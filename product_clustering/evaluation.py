"""
Evaluation module for product clustering.
Provides tools to evaluate cluster quality and coherence.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity

def load_cluster_data(data_dir: Optional[str] = None) -> Tuple[Dict, pd.DataFrame]:
    """
    Load cluster data from files.
    
    Args:
        data_dir: Directory containing cluster data files
        
    Returns:
        Tuple of (clusters_dict, cluster_assignments_df)
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    # First try to load refined clusters if available
    refined_clusters_path = os.path.join(data_dir, "refined_clusters", "refined_clusters.json")
    if os.path.exists(refined_clusters_path):
        print(f"Found refined clusters at {refined_clusters_path}")
        with open(refined_clusters_path, 'r') as f:
            clusters_dict = json.load(f)
            
        # Create a DataFrame for the refined clusters
        cluster_data = []
        for cluster_id, product_codes in clusters_dict.items():
            for code in product_codes:
                cluster_data.append({
                    'product_code': code,
                    'cluster': int(cluster_id.replace('cluster_', '')),
                    'cluster_label': cluster_id
                })
        assignments_df = pd.DataFrame(cluster_data)
    else:
        # Fall back to regular clusters if refined not available
        clusters_path = os.path.join(data_dir, "clusters.json")
        with open(clusters_path, 'r') as f:
            clusters_dict = json.load(f)
        
        # Load cluster assignments
        assignments_path = os.path.join(data_dir, "cluster_assignments.csv")
        assignments_df = pd.read_csv(assignments_path)
    
    print(f"Loaded {len(clusters_dict)} clusters")
    print(f"Loaded assignments for {len(assignments_df)} products")
    
    return clusters_dict, assignments_df

def display_cluster_samples(data_dir: Optional[str] = None, 
                          num_clusters: int = 5, 
                          samples_per_cluster: int = 5):
    """
    Display sample products from clusters for visual inspection.
    
    Args:
        data_dir: Directory containing cluster data
        num_clusters: Number of clusters to display
        samples_per_cluster: Number of samples to show per cluster
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    # Load cluster samples
    samples_path = os.path.join(data_dir, "cluster_samples.csv")
    if not os.path.exists(samples_path):
        print(f"Cluster samples file not found at {samples_path}")
        return
    
    samples_df = pd.read_csv(samples_path)
    
    # Get unique clusters
    clusters = sorted(samples_df['cluster'].unique())
    if len(clusters) == 0:
        print("No clusters found in samples")
        return
    
    # Limit number of clusters to display
    clusters_to_show = clusters[:min(num_clusters, len(clusters))]
    
    # Display samples from each cluster
    print(f"\nDisplaying samples from {len(clusters_to_show)} clusters:\n")
    
    for cluster in clusters_to_show:
        print(f"=== Cluster {cluster} ===")
        
        # Get products in this cluster
        cluster_products = samples_df[samples_df['cluster'] == cluster]
        
        # Limit number of samples to display
        display_products = cluster_products.head(samples_per_cluster)
        
        # Display product information
        for _, product in display_products.iterrows():
            product_info = f"  - {product['product_code']}: {product['product_description']}"
            if 'brand' in product and pd.notna(product['brand']):
                product_info += f" (Brand: {product['brand']})"
            if 'size' in product and pd.notna(product['size']):
                product_info += f" (Size: {product['size']})"
            print(product_info)
        
        print()

def calculate_cluster_coherence(data_dir: Optional[str] = None):
    """
    Calculate coherence scores for each cluster based on internal similarity.
    
    Args:
        data_dir: Directory containing cluster data
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    clusters_dict, assignments_df = load_cluster_data(data_dir)
    
    # Check if we're using refined clusters
    is_refined = os.path.exists(os.path.join(data_dir, "refined_clusters", "refined_clusters.json"))
    
    # Load product embeddings
    embeddings_path = os.path.join(data_dir, "product_embeddings.npy")
    product_embeddings = np.load(embeddings_path)
    
    # Load product codes
    codes_path = os.path.join(data_dir, "product_codes.txt")
    with open(codes_path, 'r') as f:
        product_codes = [line.strip() for line in f.readlines()]
    
    # Create dict mapping product codes to embeddings
    product_to_embedding = {}
    for i, code in enumerate(product_codes):
        product_to_embedding[code] = product_embeddings[i]
    
    # Calculate coherence for each cluster
    coherence_scores = {}
    missing_products = 0
    total_products = 0
    
    for cluster_id, product_list in clusters_dict.items():
        if len(product_list) <= 1:
            coherence_scores[cluster_id] = 0.0
            continue
            
        # Get embeddings for products in this cluster
        cluster_embeddings = []
        total_products += len(product_list)
        
        for product_code in product_list:
            if product_code in product_to_embedding:
                cluster_embeddings.append(product_to_embedding[product_code])
            else:
                missing_products += 1
        
        if len(cluster_embeddings) <= 1:
            coherence_scores[cluster_id] = 0.0
            continue
            
        # Calculate all pairwise similarities
        cluster_embeddings = np.array(cluster_embeddings)
        sim_matrix = cosine_similarity(cluster_embeddings)
        
        # Remove self-similarity (diagonal)
        np.fill_diagonal(sim_matrix, 0)
        
        # Average pairwise similarity as coherence score
        avg_similarity = sim_matrix.sum() / (len(cluster_embeddings) * (len(cluster_embeddings) - 1))
        coherence_scores[cluster_id] = avg_similarity
    
    if missing_products > 0:
        print(f"Warning: {missing_products} out of {total_products} products ({missing_products/total_products:.1%}) were not found in the embeddings file")
    
    # Calculate overall statistics
    coherence_values = list(coherence_scores.values())
    avg_coherence = np.mean(coherence_values)
    median_coherence = np.median(coherence_values)
    
    print(f"Average cluster coherence: {avg_coherence:.4f}")
    print(f"Median cluster coherence: {median_coherence:.4f}")
    
    # Calculate distribution of scores
    high_coherence = sum(1 for x in coherence_values if x > 0.8)
    med_coherence = sum(1 for x in coherence_values if 0.6 <= x <= 0.8)
    low_coherence = sum(1 for x in coherence_values if x < 0.6)
    print(f"High coherence clusters (>0.8): {high_coherence} ({high_coherence/len(coherence_values):.1%})")
    print(f"Medium coherence clusters (0.6-0.8): {med_coherence} ({med_coherence/len(coherence_values):.1%})")
    print(f"Low coherence clusters (<0.6): {low_coherence} ({low_coherence/len(coherence_values):.1%})")
    
    # Plot histogram of coherence scores
    plt.figure(figsize=(10, 6))
    plt.hist(coherence_values, bins=20)
    plt.xlabel('Coherence Score')
    plt.ylabel('Number of Clusters')
    plt.title('Distribution of Cluster Coherence Scores' + (' (Refined)' if is_refined else ''))
    plt.grid(True, alpha=0.3)
    
    # Save histogram to file
    hist_path = os.path.join(data_dir, "coherence_histogram" + ("_refined" if is_refined else "") + ".png")
    plt.savefig(hist_path)
    print(f"Saved coherence histogram to {hist_path}")
    plt.close()
    
    return coherence_scores

def automated_quality_check(data_dir: Optional[str] = None, 
                           coherence_threshold: float = 0.75,
                           min_cluster_size: int = 5,
                           max_cluster_size: int = 20):
    """
    Perform automated quality check on clusters.
    
    Args:
        data_dir: Directory containing cluster data
        coherence_threshold: Minimum coherence score for a good cluster
        min_cluster_size: Minimum cluster size to be considered good
        max_cluster_size: Maximum cluster size to be considered good
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    # Load coherence data if it exists
    coherence_path = os.path.join(data_dir, "cluster_coherence.csv")
    if not os.path.exists(coherence_path):
        print(f"Coherence data not found. Running coherence calculation...")
        coherence_df = calculate_cluster_coherence(data_dir)
    else:
        coherence_df = pd.read_csv(coherence_path)
    
    if coherence_df is None or len(coherence_df) == 0:
        print("No coherence data available")
        return
    
    # Define quality criteria
    good_size = (coherence_df['size'] >= min_cluster_size) & (coherence_df['size'] <= max_cluster_size)
    good_coherence = coherence_df['coherence'] >= coherence_threshold
    good_clusters = coherence_df[good_size & good_coherence]
    
    # Calculate quality metrics
    total_clusters = len(coherence_df)
    good_count = len(good_clusters)
    good_percentage = good_count / total_clusters if total_clusters > 0 else 0
    
    print(f"\nAutomated Quality Assessment:")
    print(f"  - Total clusters: {total_clusters}")
    print(f"  - Good clusters: {good_count} ({good_percentage:.2%})")
    print(f"  - Criteria: size {min_cluster_size}-{max_cluster_size}, coherence >= {coherence_threshold:.2f}")
    
    # Check if we meet target precision (80%)
    target_precision = 0.8  # 80%
    if good_percentage >= target_precision:
        print(f"\n✅ Success! Achieved {good_percentage:.2%} quality (target: {target_precision:.2%})")
    else:
        print(f"\n❌ Quality of {good_percentage:.2%} is below target of {target_precision:.2%}")
        print("Consider adjusting clustering parameters:")
        print("  - Increase min_samples for more coherent clusters")
        print("  - Adjust min_cluster_size for better sized clusters")
        print("  - Try different distance metrics (euclidean vs cosine)")

def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate product clusters")
    parser.add_argument("--data_dir", help="Directory containing cluster data")
    parser.add_argument("--display_samples", action="store_true", help="Display sample products from clusters")
    parser.add_argument("--coherence", action="store_true", help="Calculate cluster coherence")
    parser.add_argument("--quality_check", action="store_true", help="Run automated quality check")
    parser.add_argument("--all", action="store_true", help="Run all evaluation steps")
    
    args = parser.parse_args()
    
    # Set default data directory
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    # Load basic cluster data
    clusters_dict, assignments_df = load_cluster_data(data_dir)
    
    # Display summary statistics
    clustered_count = len(assignments_df[assignments_df['cluster'] >= 0])
    unclustered_count = len(assignments_df[assignments_df['cluster'] == -1])
    total_count = len(assignments_df)
    
    print(f"Clustering Summary:")
    print(f"  - Total products: {total_count}")
    print(f"  - Clustered products: {clustered_count} ({clustered_count/total_count:.2%})")
    print(f"  - Unclustered products: {unclustered_count} ({unclustered_count/total_count:.2%})")
    
    # Run selected evaluation steps
    if args.display_samples or args.all:
        display_cluster_samples(data_dir)
    
    if args.coherence or args.all:
        calculate_cluster_coherence(data_dir)
    
    if args.quality_check or args.all:
        automated_quality_check(data_dir)
    
    # If no specific steps were requested, run quality check
    if not (args.display_samples or args.coherence or args.quality_check or args.all):
        automated_quality_check(data_dir)

if __name__ == "__main__":
    main()
