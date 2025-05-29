#!/usr/bin/env python3
"""
Unified Product Clustering Script

This script provides a unified interface to run the entire product clustering pipeline:
1. Data preparation
2. Embedding generation
3. Clustering with optimized parameters
4. Optional CrossEncoder refinement
5. Evaluation and analysis

Usage:
  python run_clustering.py --help                  # Show help message
  python run_clustering.py --all                   # Run complete pipeline
  python run_clustering.py --prepare --embed       # Only prepare data and generate embeddings
  python run_clustering.py --cluster --rerank      # Only run clustering with reranking
  python run_clustering.py --analyze               # Only analyze results
  python run_clustering.py --test --sample_size 500 # Run in test mode with 500 samples
"""

import os
import sys
import argparse
import time
from typing import Optional

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_data_preparation(data_dir: Optional[str] = None, force: bool = False):
    """
    Run the data preparation step.
    
    Args:
        data_dir: Directory to store prepared data
        force: Whether to force reprocessing even if files exist
    """
    from data_prep.processor import prepare_unified_product_data
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    output_path = os.path.join(data_dir, "prepared_products.csv")
    
    if os.path.exists(output_path) and not force:
        print(f"Prepared data already exists at {output_path}")
        print("Use --force to reprocess")
        return output_path
    
    print("Running data preparation...")
    # Call the function with its correct signature (no output_path parameter)
    prepared_data = prepare_unified_product_data()
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the prepared data
    prepared_data.to_csv(output_path, index=False)
    print(f"Data preparation complete. Results saved to {output_path}")
    
    return output_path

def run_embedding_generation(data_dir: Optional[str] = None, 
                            prepared_data_path: Optional[str] = None,
                            model_name: str = "all-mpnet-base-v2",
                            force: bool = False):
    """
    Run the embedding generation step.
    
    Args:
        data_dir: Directory to store embeddings
        prepared_data_path: Path to prepared data CSV
        model_name: Name of the embedding model to use
        force: Whether to force regeneration even if files exist
    """
    from product_clustering.embed_products import embed_products, save_embeddings
    import pandas as pd
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    if prepared_data_path is None:
        prepared_data_path = os.path.join(data_dir, "prepared_products.csv")
    
    embeddings_path = os.path.join(data_dir, "product_embeddings.npy")
    product_codes_path = os.path.join(data_dir, "product_codes.txt")
    
    if os.path.exists(embeddings_path) and os.path.exists(product_codes_path) and not force:
        print(f"Embeddings already exist at {embeddings_path}")
        print("Use --force to regenerate")
        return embeddings_path, product_codes_path
    
    # Load the prepared data
    if not os.path.exists(prepared_data_path):
        print(f"Error: Prepared data file not found at {prepared_data_path}")
        print("Run data preparation first")
        return None, None
    
    print(f"Loading prepared data from {prepared_data_path}")
    df = pd.read_csv(prepared_data_path)
    
    print(f"Generating embeddings using model {model_name}...")
    # Generate embeddings according to the function's actual signature
    embeddings, product_codes = embed_products(
        df=df,
        embedding_type='sentence-transformer',
        model_name=model_name
    )
    
    # Save the embeddings to files
    save_embeddings(embeddings, product_codes, data_dir)
    
    print(f"Embedding generation complete. Results saved to {embeddings_path}")
    
    return embeddings_path, product_codes_path

def run_clustering(data_dir: Optional[str] = None,
                  metric: str = "euclidean",
                  min_cluster_size: int = 3,
                  min_samples: int = 2,
                  test_mode: bool = False,
                  sample_size: int = 1000,
                  use_reranking: bool = False,
                  cross_encoder_model: str = "cross-encoder/stsb-roberta-base",
                  similarity_threshold: float = 0.6,
                  use_categories: bool = True,
                  force: bool = False):
    """
    Run the clustering step with optional reranking.
    
    Args:
        data_dir: Directory containing data files
        metric: Distance metric to use
        min_cluster_size: Minimum size of clusters
        min_samples: HDBSCAN min_samples parameter
        test_mode: Whether to run in test mode
        sample_size: Number of samples to use in test mode
        use_reranking: Whether to use CrossEncoder reranking
        cross_encoder_model: Model for reranking
        similarity_threshold: Threshold for reranking
        use_categories: Whether to use hierarchical category-based clustering
        force: Whether to force reprocessing even if files exist
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Prepare data file path
    prepared_data_path = os.path.join(data_dir, "prepared_products.csv")
    
    # Determine which clustering approach to use
    if use_categories:
        from product_clustering.category_clustering import run_category_clustering
        
        # Determine output directory for category-based clustering
        suffix = "_subset" if test_mode else ""
        output_dir = os.path.join(data_dir, f"category_clustering{suffix}")
        
        # Define path for final results
        if use_reranking:
            final_dir = os.path.join(output_dir, "refined")
            final_path = os.path.join(final_dir, "refined_category_clusters.json")
        else:
            final_dir = output_dir
            final_path = os.path.join(final_dir, "category_clusters.json")
        
        # Check if results already exist
        if os.path.exists(final_path) and not force:
            print(f"Category-based clustering results already exist at {final_path}")
            print("Use --force to reprocess")
            return final_path
        
        print("Running category-based hierarchical clustering...")
        start_time = time.time()
        
        # Run category-based clustering
        clusters_path = run_category_clustering(
            prepared_data_path=prepared_data_path,
            output_dir=output_dir,
            model_name="all-mpnet-base-v2",
            metric=metric,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            use_reranking=use_reranking,
            cross_encoder_model=cross_encoder_model,
            similarity_threshold=similarity_threshold
        )
        
        end_time = time.time()
        minutes, seconds = divmod(end_time - start_time, 60)
        print(f"Category-based clustering complete in {int(minutes)}m {seconds:.1f}s")
        
        # Load and print basic stats from the final results
        import json
        with open(clusters_path, 'r') as f:
            clusters = json.load(f)
        
        total_products = sum(len(products) for products in clusters.values())
        print(f"Created {len(clusters)} clusters with {total_products} total products")
        print(f"Final results saved to {clusters_path}")
        
        return clusters_path
    else:
        # Use the original improved clustering approach
        from product_clustering.improved_clustering import run_improved_clustering
        
        # Determine output directory for regular clustering
        suffix = "_subset" if test_mode else ""
        output_dir = os.path.join(data_dir, f"improved_clustering{suffix}")
        
        # Refined clusters path
        refined_dir = os.path.join(data_dir, "refined_clusters")
        refined_path = os.path.join(refined_dir, "refined_clusters.json")
        
        # Check if results already exist
        if os.path.exists(refined_path) and not force:
            print(f"Clustering results already exist at {refined_path}")
            print("Use --force to reprocess")
            return refined_path
        
        print("Running standard clustering...")
        start_time = time.time()
        
        run_improved_clustering(
            data_dir=data_dir,
            metric=metric,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            test_mode=test_mode,
            sample_size=sample_size,
            use_reranking=use_reranking,
            cross_encoder_model=cross_encoder_model,
            similarity_threshold=similarity_threshold
        )
        
        end_time = time.time()
        print(f"Clustering complete in {end_time - start_time:.2f} seconds")
        
        # Copy final results to standard location
        if use_reranking:
            os.makedirs(refined_dir, exist_ok=True)
            import shutil
            import json
            
            refined_output_path = os.path.join(output_dir, "refined", "refined_clusters.json")
            
            if os.path.exists(refined_output_path):
                shutil.copy(refined_output_path, refined_path)
                
                # Load and print basic stats
                with open(refined_path, 'r') as f:
                    clusters = json.load(f)
                
                total_products = sum(len(products) for products in clusters.values())
                print(f"Created {len(clusters)} clusters with {total_products} total products")
                print(f"Final results saved to {refined_path}")
                
                return refined_path
    
    return output_dir

def run_analysis(data_dir: Optional[str] = None, refined: bool = True, use_categories: bool = True):
    """
    Run the analysis step.
    
    Args:
        data_dir: Directory containing clustering results
        refined: Whether to analyze refined clusters
        use_categories: Whether to analyze category-based clustering results
    """
    from product_clustering.analyze_clusters import analyze_clusters
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    if use_categories:
        # Paths for category-based clustering results
        if refined:
            clusters_path = os.path.join(data_dir, "category_clustering", "refined", "refined_category_clusters.json")
            if not os.path.exists(clusters_path):
                print(f"Refined category clusters not found at {clusters_path}")
                print("Run category clustering with reranking first or use --refined=False")
                return None
        else:
            clusters_path = os.path.join(data_dir, "category_clustering", "category_clusters.json")
            if not os.path.exists(clusters_path):
                print(f"Category clusters not found at {clusters_path}")
                print("Run category clustering first")
                return None
    else:
        # Paths for regular clustering results
        if refined:
            clusters_path = os.path.join(data_dir, "refined_clusters", "refined_clusters.json")
            if not os.path.exists(clusters_path):
                print(f"Refined clusters not found at {clusters_path}")
                print("Run clustering with reranking first or use --refined=False")
                return None
        else:
            clusters_path = os.path.join(data_dir, "improved_clustering", "clusters.json")
        if not os.path.exists(clusters_path):
            print(f"Clusters not found at {clusters_path}")
            print("Run clustering first")
            return None
    
    print(f"Analyzing {'refined ' if refined else ''}clusters...")
    output_path = analyze_clusters(
        clusters_path=clusters_path,
        data_dir=data_dir,
        refined=refined
    )
    
    print(f"Analysis complete. Results saved to {output_path}")
    return output_path

def main():
    """Main function to run the product clustering pipeline."""
    parser = argparse.ArgumentParser(description="Run product clustering pipeline")
    
    # Pipeline steps
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--prepare", action="store_true", help="Run data preparation")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings")
    parser.add_argument("--cluster", action="store_true", help="Run clustering")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    
    # Data options
    parser.add_argument("--data_dir", help="Directory for data files")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if files exist")
    
    # Embedding options
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model to use")
    
    # Clustering options
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
    parser.add_argument("--categories", action="store_true", default=True,
                        help="Use category-based hierarchical clustering (default)")
    parser.add_argument("--no-categories", action="store_false", dest="categories",
                        help="Use standard clustering without category hierarchy")
    
    # Reranking options
    parser.add_argument("--rerank", action="store_true",
                        help="Use CrossEncoder reranking to refine clusters")
    parser.add_argument("--cross_encoder_model", default="cross-encoder/stsb-roberta-base",
                        help="CrossEncoder model to use for reranking")
    parser.add_argument("--similarity_threshold", type=float, default=0.6,
                        help="Similarity threshold for CrossEncoder reranking (higher = more strict)")
    
    # Analysis options
    parser.add_argument("--refined", action="store_true", default=True,
                        help="Analyze refined clusters (default)")
    parser.add_argument("--no-refined", action="store_false", dest="refined",
                        help="Analyze original clusters instead of refined")
    
    args = parser.parse_args()
    
    # If no specific steps are specified, print help and exit
    if not (args.all or args.prepare or args.embed or args.cluster or args.analyze):
        parser.print_help()
        return
    
    # Set up data directory
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Run selected pipeline steps
    if args.all or args.prepare:
        prepared_data_path = run_data_preparation(data_dir, args.force)
    
    if args.all or args.embed:
        embeddings_path, product_codes_path = run_embedding_generation(
            data_dir, 
            prepared_data_path if 'prepared_data_path' in locals() else None,
            args.model,
            args.force
        )
    
    if args.all or args.cluster:
        clusters_path = run_clustering(
            data_dir,
            args.metric,
            args.min_cluster_size,
            args.min_samples,
            args.test,
            args.sample_size,
            args.rerank,
            args.cross_encoder_model,
            args.similarity_threshold,
            args.categories,
            args.force
        )
    
    if args.all or args.analyze:
        analysis_path = run_analysis(data_dir, args.refined, args.categories)
    
    print("\nProduct clustering pipeline completed successfully!")

if __name__ == "__main__":
    main()
