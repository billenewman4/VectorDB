"""
Product hierarchical clustering pipeline script for VectorDB.
Uses the hierarchical clustering pipeline with configurable embedding and cross-encoder options.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
import argparse
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import hierarchical pipeline
from Clean_Code.Pipeline.hierarchical_pipeline import HierarchicalClusteringPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_product_data(data_path: str, embeddings_path: str) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Load product data and embeddings.
    
    Args:
        data_path: Path to product data CSV
        embeddings_path: Path to product embeddings NPZ
        
    Returns:
        Tuple containing:
        - Embeddings array
        - List of product descriptions
        - List of product metadata dictionaries
    """
    # Load product data
    logger.info(f"Loading product data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    vectors = embeddings['embeddings']
    
    # Verify dimensions match
    if len(df) != vectors.shape[0]:
        raise ValueError(f"Mismatch between number of products ({len(df)}) and embeddings ({vectors.shape[0]})")
    
    # Extract product descriptions and normalize if needed
    descriptions = df['description'].tolist()
    
    # Create metadata for each product
    metadata = []
    for _, row in df.iterrows():
        item = {col: row[col] for col in df.columns if col != 'description'}
        metadata.append(item)
    
    logger.info(f"Loaded {len(descriptions)} products with {vectors.shape[1]}-dimensional embeddings")
    
    return vectors, descriptions, metadata

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Product Hierarchical Clustering")
    parser.add_argument("--data_path", type=str, required=True, help="Path to product data CSV")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to product embeddings NPZ")
    parser.add_argument("--levels", type=int, default=3, help="Number of hierarchical levels")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Minimum cluster size (default: 3)")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum samples (default: 2)")
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", 
                       help="Cross-encoder model to use for refinement")
    parser.add_argument("--l1_cross_encoder", action="store_true", 
                       help="Use cross-encoder for level 1 clustering (default: False, uses embeddings)")
    parser.add_argument("--l2_cross_encoder", action="store_true", 
                       help="Use cross-encoder for level 2 clustering (default: False, uses embeddings)")
    parser.add_argument("--l3_cross_encoder", action="store_true", 
                       help="Use cross-encoder for level 3 clustering (default: False, uses embeddings)")
    parser.add_argument("--refine_l1", action="store_true", 
                       help="Apply cross-encoder refinement after level 1 clustering")
    parser.add_argument("--refine_l2", action="store_true", 
                       help="Apply cross-encoder refinement after level 2 clustering")
    parser.add_argument("--refine_l3", action="store_true", 
                       help="Apply cross-encoder refinement after level 3 clustering")
    parser.add_argument("--output_dir", type=str, default="product_hierarchical_results", 
                       help="Output directory")
    parser.add_argument("--test_mode", action="store_true", help="Run on a small subset of data for testing")
    parser.add_argument("--test_samples", type=int, default=500, 
                       help="Number of samples to use in test mode")
    parser.add_argument("--save_plots", action="store_true", help="Save visualization plots")
    args = parser.parse_args()
    
    # Load product data and embeddings
    vectors, descriptions, metadata = load_product_data(args.data_path, args.embeddings_path)
    
    # If test mode, use a subset of the data
    if args.test_mode:
        logger.info(f"Running in test mode with {args.test_samples} samples")
        sample_indices = np.random.choice(
            len(descriptions), 
            min(args.test_samples, len(descriptions)), 
            replace=False
        )
        vectors = vectors[sample_indices]
        descriptions = [descriptions[i] for i in sample_indices]
        metadata = [metadata[i] for i in sample_indices]
    
    # Configure hierarchical pipeline with level-specific settings
    config = {
        "levels": args.levels,
        "level_configs": {
            # Level 1 config
            1: {
                "use_cross_encoder": args.l1_cross_encoder,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "refine_after_clustering": args.refine_l1,
                "refinement_method": "borderline"
            },
            # Level 2 config
            2: {
                "use_cross_encoder": args.l2_cross_encoder,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "refine_after_clustering": args.refine_l2,
                "refinement_method": "borderline"
            },
            # Level 3 config
            3: {
                "use_cross_encoder": args.l3_cross_encoder,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "refine_after_clustering": args.refine_l3,
                "refinement_method": "borderline"
            }
        },
        "embedding": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "metric": "cosine",
            "cluster_selection_method": "eom",
            "prediction_data": True
        },
        "preprocessing": {
            "normalize": True,
            "normalize_method": "l2",
            "remove_outliers": False
        },
        "cross_encoder": {
            "use_refinement": args.refine_l1 or args.refine_l2 or args.refine_l3 or 
                             args.l1_cross_encoder or args.l2_cross_encoder or args.l3_cross_encoder,
            "refinement_method": "borderline",
            "embedding_weight": 0.7,
            "cross_encoder_weight": 0.3,
            "confidence_threshold": 0.6,
            "batch_size": 32
        },
        "visualization": {
            "create_visualizations": args.save_plots,
            "method": "umap",
            "dims": 2,
            "figsize": (12, 8),
            "save_plots": args.save_plots
        },
        "output": {
            "save_results": True,
            "output_dir": args.output_dir
        }
    }
    
    # Initialize cross-encoder reranker if needed
    reranker = None
    if (args.refine_l1 or args.refine_l2 or args.refine_l3 or 
        args.l1_cross_encoder or args.l2_cross_encoder or args.l3_cross_encoder):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Initializing cross-encoder with model: {args.reranker_model}")
            reranker = CrossEncoder(args.reranker_model)
        except ImportError:
            logger.warning("sentence-transformers not installed. Cannot use CrossEncoder refinement.")
            logger.warning("Install with: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"Failed to initialize CrossEncoder: {str(e)}")
    
    # Initialize and run hierarchical clustering pipeline
    logger.info("Initializing hierarchical clustering pipeline...")
    pipeline = HierarchicalClusteringPipeline(config)
    
    # Run hierarchical clustering
    start_time = time.time()
    logger.info("Running hierarchical clustering...")
    results = pipeline.run(vectors, descriptions, reranker, metadata)
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Hierarchical clustering complete in {elapsed_time:.2f} seconds!")
    
    for level, level_results in results["levels"].items():
        stats = level_results.get("statistics", {})
        num_clusters = stats.get("num_clusters", 0)
        points_assigned = stats.get("points_assigned", 0)
        noise_percentage = stats.get("noise_percentage", 0)
        clustering_method = stats.get("clustering_method", "unknown")
        refined = stats.get("refined", False)
        
        method_str = f"{clustering_method} + refinement" if refined else clustering_method
        logger.info(f"Level {level}: {num_clusters} clusters, {points_assigned}/{len(descriptions)} points " +
                   f"assigned ({noise_percentage:.2f}% noise) [Method: {method_str}]")
    
    # Get path to results
    output_path = pipeline.results.get("summary", {}).get("output_dir", args.output_dir)
    logger.info(f"Results saved to: {output_path}")
    
    # Optional: Query example
    if len(descriptions) > 0:
        # Show an example of querying similar products for the first product
        try:
            query_idx = 0
            similar_indices = pipeline.find_similar_items(query_idx, level=1, k=5)
            
            logger.info("\nExample query:")
            logger.info(f"Query product: {descriptions[query_idx][:50]}...")
            
            logger.info("\nSimilar products:")
            for idx in similar_indices:
                logger.info(f"- {descriptions[idx][:50]}...")
        except Exception as e:
            logger.warning(f"Failed to query similar products: {str(e)}")

if __name__ == "__main__":
    main()
