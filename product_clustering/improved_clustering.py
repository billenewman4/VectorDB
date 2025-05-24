"""
Improved clustering module with optimized parameters.
"""
import os
import sys
import numpy as np
from typing import Optional

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base clustering module
from product_clustering.clustering import main as base_clustering

def run_improved_clustering(data_dir: Optional[str] = None,
                           metric: str = 'euclidean',  # Using euclidean instead of cosine as HDBSCAN doesn't support cosine directly
                           min_cluster_size: int = 3,  # Lower to allow more granular clusters
                           min_samples: int = 2,  # Lower to allow more granular clusters
                           test_mode: bool = False,
                           sample_size: int = 1000,  # Sample size to use in test mode
                           use_reranking: bool = False,  # Whether to use CrossEncoder reranking
                           cross_encoder_model: str = 'cross-encoder/stsb-roberta-base',  # Model for reranking
                           similarity_threshold: float = 0.5):  # Threshold for reranking
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
    
    # Call the base clustering function with improved parameters
    base_clustering(
        embeddings_path=embeddings_path,
        product_codes_path=product_codes_path,
        prepared_data_path=prepared_data_path,
        output_dir=output_dir,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        sample_size=5,
        sample_interval=30
    )
    
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
