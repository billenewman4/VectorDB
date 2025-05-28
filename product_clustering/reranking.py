"""
Cluster refinement module using CrossEncoder reranking.
Utilizes the existing CrossEncoder implementation to improve cluster quality.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing CrossEncoder implementation
from src.VectorDB.CrossEncoder import CrossEncoder

def refine_clusters(
    clusters_path: str,
    prepared_data_path: str,
    output_dir: Optional[str] = None,
    model_name: str = 'cross-encoder/stsb-roberta-base',
    similarity_threshold: float = 0.5,
    batch_size: int = 32,
    max_pairs_per_cluster: int = 1000
) -> Dict[str, List[str]]:
    """
    Refine clusters using CrossEncoder similarity scoring.
    
    Args:
        clusters_path: Path to clusters.json file
        prepared_data_path: Path to prepared_products.csv
        output_dir: Directory to save refined clusters
        model_name: CrossEncoder model name
        similarity_threshold: Minimum similarity threshold to keep product in cluster
        batch_size: Batch size for CrossEncoder predictions
        max_pairs_per_cluster: Maximum number of pairs to evaluate per cluster
        
    Returns:
        Dictionary mapping cluster IDs to lists of product codes
    """
    print(f"Loading clusters from {clusters_path}")
    with open(clusters_path, 'r') as f:
        clusters = json.load(f)
    
    print(f"Loading product data from {prepared_data_path}")
    products_df = pd.read_csv(prepared_data_path)
    
    # Create a lookup dictionary for quick access to product descriptions
    product_lookup = {}
    for _, row in products_df.iterrows():
        product_lookup[row['product_code']] = row['product_description']
    
    # Initialize CrossEncoder
    print(f"Initializing CrossEncoder with model: {model_name}")
    cross_encoder = CrossEncoder(model_name=model_name)
    
    refined_clusters = {}
    total_removed = 0
    total_products = 0
    
    print(f"Refining {len(clusters)} clusters...")
    for cluster_id, product_codes in tqdm(clusters.items()):
        if len(product_codes) <= 2:
            # Keep very small clusters as is
            refined_clusters[cluster_id] = product_codes
            total_products += len(product_codes)
            continue
        
        # Get product descriptions for this cluster
        descriptions = []
        valid_codes = []
        for code in product_codes:
            if code in product_lookup:
                descriptions.append(product_lookup[code])
                valid_codes.append(code)
            
        # Skip if we don't have enough valid products
        if len(valid_codes) <= 2:
            refined_clusters[cluster_id] = valid_codes
            total_products += len(valid_codes)
            continue
            
        # Create pairs for CrossEncoder evaluation
        pairs = []
        pair_indices = []
        
        # If there are too many potential pairs, sample a subset
        if len(valid_codes) > 20:  # Arbitrary threshold for large clusters
            # Sample pairs to avoid O(n²) explosion
            import random
            random.seed(42)  # For reproducibility
            
            # Generate representative pairs by comparing each item to a few others
            for i in range(len(valid_codes)):
                # Select a few random items to compare against
                compare_indices = random.sample(
                    [j for j in range(len(valid_codes)) if j != i],
                    min(5, len(valid_codes) - 1)  # Compare with at most 5 other items
                )
                
                for j in compare_indices:
                    pairs.append([descriptions[i], descriptions[j]])
                    pair_indices.append((i, j))
                    
                    # Limit total pairs to avoid memory issues
                    if len(pairs) >= max_pairs_per_cluster:
                        break
                
                if len(pairs) >= max_pairs_per_cluster:
                    print(f"  Limiting cluster {cluster_id} to {max_pairs_per_cluster} pairs for evaluation")
                    break
        else:
            # For small clusters, evaluate all pairs
            for i in range(len(valid_codes)):
                for j in range(i+1, len(valid_codes)):
                    pairs.append([descriptions[i], descriptions[j]])
                    pair_indices.append((i, j))
        
        # Format pairs for the CrossEncoder
        candidates = [{"text": pair[1]} for pair in pairs]
        
        # Score all pairs using the CrossEncoder's rerank method
        similarity_scores = []
        
        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            # Adapt to the CrossEncoder interface - create text pairs
            text_pairs = []
            for pair in batch_pairs:
                text_pairs.append([pair[0], pair[1]])
                
            # Use the internal sentence-transformers CrossEncoder for direct prediction
            batch_scores = cross_encoder.model.predict(text_pairs)
            similarity_scores.extend(batch_scores)
        
        # Build similarity matrix from pair scores
        similarity_matrix = np.zeros((len(valid_codes), len(valid_codes)))
        
        for idx, (i, j) in enumerate(pair_indices):
            score = similarity_scores[idx]
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score  # Symmetric
            
        # Set diagonal to 1.0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Calculate average similarity of each product to the rest of the cluster
        avg_similarities = similarity_matrix.mean(axis=1)
        
        # Identify products to keep (above threshold)
        keep_indices = [i for i, sim in enumerate(avg_similarities) if sim >= similarity_threshold]
        
        # If we're removing more than 30% of items, just keep the top 70%
        if len(keep_indices) < 0.7 * len(valid_codes):
            keep_count = max(2, int(0.7 * len(valid_codes)))
            keep_indices = np.argsort(-avg_similarities)[:keep_count].tolist()
        
        # Get refined product codes
        refined_product_codes = [valid_codes[i] for i in keep_indices]
        
        # Track statistics
        removed_count = len(valid_codes) - len(refined_product_codes)
        total_removed += removed_count
        total_products += len(refined_product_codes)
        
        # Store refined cluster
        refined_clusters[cluster_id] = refined_product_codes
        
        if removed_count > 0:
            print(f"  Cluster {cluster_id}: Removed {removed_count} of {len(valid_codes)} products")
    
    # Print summary
    print(f"Refinement complete:")
    print(f"  - Original clusters: {len(clusters)}")
    print(f"  - Refined clusters: {len(refined_clusters)}")
    print(f"  - Products removed: {total_removed}")
    print(f"  - Products retained: {total_products}")
    
    # Save refined clusters if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save refined clusters to JSON
        refined_clusters_path = os.path.join(output_dir, "refined_clusters.json")
        with open(refined_clusters_path, 'w') as f:
            json.dump(refined_clusters, f, indent=2)
        print(f"Saved refined clusters to {refined_clusters_path}")
        
        # Convert to assignments format and save
        cluster_assignments = []
        for cluster_id, product_codes in refined_clusters.items():
            # Extract numeric part or use index as fallback
            try:
                if '_' in cluster_id:
                    # For IDs like 'cluster_435', extract the number after the underscore
                    cluster_num = int(cluster_id.split('_')[-1])
                else:
                    # If no underscore, try to convert the whole string to int
                    cluster_num = int(cluster_id)
            except ValueError:
                # If conversion fails, use a hash of the string as a unique number
                cluster_num = hash(cluster_id) % 100000
                
            for product_code in product_codes:
                cluster_assignments.append({
                    'product_code': product_code,
                    'cluster': cluster_num,
                    'cluster_id': f"refined_cluster_{cluster_id}"
                })
        
        # Add unclustered products
        clustered_products = set()
        for cluster in refined_clusters.values():
            clustered_products.update(cluster)
        
        for _, row in products_df.iterrows():
            product_code = row['product_code']
            if product_code not in clustered_products:
                cluster_assignments.append({
                    'product_code': product_code,
                    'cluster': -1,  # Noise cluster
                    'cluster_id': "noise"
                })
        
        # Save assignments
        assignments_df = pd.DataFrame(cluster_assignments)
        assignments_path = os.path.join(output_dir, "refined_assignments.csv")
        assignments_df.to_csv(assignments_path, index=False)
        print(f"Saved refined assignments to {assignments_path}")
        
        # Save unclustered products
        unclustered = assignments_df[assignments_df['cluster'] == -1]
        unclustered_path = os.path.join(output_dir, "refined_unclustered.csv")
        unclustered.to_csv(unclustered_path, index=False)
        print(f"Saved {len(unclustered)} unclustered products to {unclustered_path}")
    
    return refined_clusters

def main(clusters_path: Optional[str] = None,
         prepared_data_path: Optional[str] = None,
         output_dir: Optional[str] = None,
         model_name: str = 'cross-encoder/stsb-roberta-base',
         similarity_threshold: float = 0.5):
    """
    Main function to run cluster refinement.
    
    Args:
        clusters_path: Path to clusters.json file
        prepared_data_path: Path to prepared_products.csv
        output_dir: Directory to save refined clusters
        model_name: CrossEncoder model name
        similarity_threshold: Minimum similarity threshold
    """
    # Set default paths if not provided
    if clusters_path is None or prepared_data_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        
        if clusters_path is None:
            clusters_path = os.path.join(data_dir, "clusters.json")
            
        if prepared_data_path is None:
            prepared_data_path = os.path.join(data_dir, "prepared_products.csv")
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "refined_clusters")
    
    # Run refinement
    refine_clusters(
        clusters_path=clusters_path,
        prepared_data_path=prepared_data_path,
        output_dir=output_dir,
        model_name=model_name,
        similarity_threshold=similarity_threshold
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Refine clusters using CrossEncoder")
    parser.add_argument("--clusters_path", help="Path to clusters.json file")
    parser.add_argument("--prepared_data_path", help="Path to prepared_products.csv")
    parser.add_argument("--output_dir", help="Directory to save refined clusters")
    parser.add_argument("--model_name", default="cross-encoder/stsb-roberta-base",
                        help="CrossEncoder model name")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Minimum similarity threshold")
    
    args = parser.parse_args()
    
    main(
        clusters_path=args.clusters_path,
        prepared_data_path=args.prepared_data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        similarity_threshold=args.similarity_threshold
    )
