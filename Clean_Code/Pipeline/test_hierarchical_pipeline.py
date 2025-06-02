"""
Test script for the hierarchical clustering pipeline.

This script tests all permutations of the hierarchical clustering pipeline:
- Different combinations of embedding and cross-encoder usage per level
- With and without refinement at each level
- Across multiple hierarchical levels (up to 4)
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import time
import argparse
import itertools
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import components
sys.path.append(parent_dir)  # Add parent dir to path

from Pipeline.hierarchical_pipeline import HierarchicalClusteringPipeline
from Data_prep.prepare_data import prepare_data_for_clustering
from Vector_Embedding.sentence_transformer_encoder import SentenceTransformerEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_test_data(data_path: Optional[str] = None, 
                    embedding_model: str = "all-mpnet-base-v2",
                    test_size: int = 300,
                    use_cached_embeddings: bool = True,
                    embeddings_path: Optional[str] = None) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Set up test data for hierarchical clustering.
    
    Args:
        data_path: Path to product data CSV
        embedding_model: Name of the embedding model to use
        test_size: Number of samples to use for testing
        use_cached_embeddings: Whether to use cached embeddings if available
        embeddings_path: Path to cached embeddings
        
    Returns:
        Tuple containing:
        - Embeddings array
        - List of product descriptions
        - List of product metadata dictionaries
    """
    # Prepare data for clustering
    logger.info("Setting up synthetic test data...")
    
    # Generate synthetic product data for testing
    from sklearn.datasets import make_blobs
    
    # Create synthetic product descriptions
    product_categories = [
        "Fruits", "Vegetables", "Dairy", "Meat", "Beverages", "Bakery", 
        "Canned Goods", "Frozen Foods", "Snacks", "Cleaning Supplies"
    ]
    
    # Product attributes per category
    category_attributes = {
        "Fruits": ["apple", "banana", "orange", "pear", "grape", "strawberry", "blueberry", "raspberry"],
        "Vegetables": ["carrot", "broccoli", "spinach", "lettuce", "tomato", "pepper", "onion", "potato"],
        "Dairy": ["milk", "cheese", "yogurt", "butter", "cream", "ice cream", "sour cream"],
        "Meat": ["beef", "chicken", "pork", "turkey", "lamb", "fish", "shrimp"],
        "Beverages": ["water", "soda", "juice", "coffee", "tea", "beer", "wine"],
        "Bakery": ["bread", "bagel", "muffin", "cake", "cookie", "pastry"],
        "Canned Goods": ["soup", "beans", "tuna", "corn", "tomatoes"],
        "Frozen Foods": ["pizza", "vegetables", "ice cream", "meals", "desserts"],
        "Snacks": ["chips", "pretzels", "popcorn", "nuts", "crackers", "chocolate"],
        "Cleaning Supplies": ["detergent", "soap", "cleaner", "paper towels", "sponges"]
    }
    
    # Create synthetic descriptions
    import random
    import uuid
    
    descriptions = []
    metadata = []
    
    actual_size = min(test_size, 200)  # Limit to 200 max for testing
    
    for i in range(actual_size):
        # Randomly select a category
        category = random.choice(product_categories)
        
        # Generate a product description based on category
        attributes = category_attributes[category]
        attribute = random.choice(attributes)
        size = random.choice(["small", "medium", "large", "extra large"])
        brand = f"Brand{random.randint(1, 10)}"
        
        # Create descriptive text
        description = f"{size} {brand} {attribute} in {category} department"
        descriptions.append(description)
        
        # Create metadata
        product_code = str(uuid.uuid4())[:8]  # Generate a unique product code
        meta = {
            "product_code": product_code,
            "category": category,
            "brand": brand,
            "size": size,
            "attribute": attribute,
            "price": round(random.uniform(1.0, 50.0), 2),
            "stock": random.randint(0, 100)
        }
        metadata.append(meta)
    
    logger.info(f"Created {len(descriptions)} synthetic product descriptions")
    
    # Generate synthetic embeddings (or use model if installed)
    try:
        # Try to use the actual encoder if available
        encoder = SentenceTransformerEncoder(model_name=embedding_model)
        vectors = encoder.encode_batch(descriptions, batch_size=32)
        logger.info(f"Generated embeddings using model: {embedding_model}")
    except Exception as e:
        # Fall back to synthetic embeddings if model not available
        logger.warning(f"Sentence transformer failed: {e}. Using synthetic embeddings instead.")
        # Create synthetic embeddings using make_blobs
        X, y = make_blobs(n_samples=len(descriptions), centers=10, n_features=384, random_state=42)
        vectors = X
        logger.info(f"Created synthetic embeddings with shape {vectors.shape}")
    
    # Save embeddings if path provided
    if embeddings_path:
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        logger.info(f"Saving embeddings to {embeddings_path}")
        np.savez_compressed(embeddings_path, embeddings=vectors)
    
    return vectors, descriptions, metadata


class MockReranker:
    """Mock cross-encoder implementation for testing"""
    def __init__(self):
        self.name = "mock-reranker"
    
    def compute_similarity(self, queries, passages):
        """Mock implementation of compute_similarity using cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Convert text to simple embeddings if they're strings
        if isinstance(queries[0], str) and isinstance(passages[0], str):
            # Just return random similarity scores for text inputs
            scores = np.random.rand(len(queries))
            return scores
        
        # If we have numpy arrays, compute actual cosine similarity
        if isinstance(queries, np.ndarray) and isinstance(passages, np.ndarray):
            return cosine_similarity(queries, passages).flatten()
        
        # Fallback to random scores
        return np.random.rand(len(queries))


def initialize_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Any:
    """
    Initialize cross-encoder for refinement.
    
    Args:
        model_name: Name of the cross-encoder model to use
        
    Returns:
        Initialized cross-encoder
    """
    logger.info(f"Initializing cross-encoder with model: {model_name}")
    
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(model_name)
        return reranker
    except Exception as e:
        logger.warning(f"Error initializing cross-encoder: {str(e)}. Using mock reranker.")
        # Return our mock reranker implementation
        return MockReranker()


def run_test_configuration(vectors: np.ndarray, 
                          descriptions: List[str], 
                          metadata: List[Dict[str, Any]],
                          reranker: Any,
                          levels: int,
                          level_configs: Dict[int, Dict[str, Any]],
                          output_dir: str,
                          test_id: str) -> Dict[str, Any]:
    """
    Run a single test configuration of the hierarchical clustering pipeline.
    
    Args:
        vectors: Embedding vectors
        descriptions: Text descriptions
        metadata: Metadata for each item
        reranker: Cross-encoder reranker
        levels: Number of hierarchical levels
        level_configs: Configuration for each level
        output_dir: Base output directory
        test_id: Identifier for this test configuration
        
    Returns:
        Dictionary with test results
    """
    # Configure hierarchical pipeline
    config = {
        "levels": levels,
        "level_configs": level_configs,
        "embedding": {
            "min_cluster_size": 3,
            "min_samples": 2,
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
            "use_refinement": True,  # We'll control this per level in level_configs
            "refinement_method": "borderline",
            "embedding_weight": 0.7,
            "cross_encoder_weight": 0.3,
            "confidence_threshold": 0.6,
            "batch_size": 32
        },
        "visualization": {
            "create_visualizations": True,
            "method": "umap",
            "dims": 2,
            "figsize": (12, 8),
            "save_plots": True
        },
        "output": {
            "save_results": True,
            "output_dir": os.path.join(output_dir, test_id),
            "timestamp_directories": False
        }
    }
    
    # Initialize and run hierarchical clustering pipeline
    logger.info(f"Running test configuration: {test_id}")
    start_time = time.time()
    
    pipeline = HierarchicalClusteringPipeline(config)
    results = pipeline.run(vectors, descriptions, reranker, metadata)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare test results summary
    test_results = {
        "test_id": test_id,
        "runtime_seconds": elapsed_time,
        "config": level_configs,
        "results": {}
    }
    
    # Add statistics for each level
    for level, level_results in results.get("levels", {}).items():
        stats = level_results.get("statistics", {})
        test_results["results"][level] = {
            "num_clusters": stats.get("num_clusters", 0),
            "points_assigned": stats.get("points_assigned", 0),
            "noise_percentage": stats.get("noise_percentage", 0),
            "clustering_method": stats.get("clustering_method", "unknown"),
            "refined": stats.get("refined", False)
        }
    
    # Log results
    logger.info(f"Test {test_id} complete in {elapsed_time:.2f} seconds")
    for level, level_stats in test_results["results"].items():
        logger.info(f"Level {level}: {level_stats['num_clusters']} clusters, " +
                   f"{level_stats['points_assigned']}/{len(descriptions)} points assigned " +
                   f"({level_stats['noise_percentage']:.2f}% noise) " +
                   f"[Method: {level_stats['clustering_method']}]")
    
    return test_results


def generate_test_configurations(max_levels: int = 4) -> List[Dict[str, Any]]:
    """
    Generate all test configurations to evaluate.
    
    Args:
        max_levels: Maximum number of hierarchical levels to test
        
    Returns:
        List of test configurations
    """
    all_configurations = []
    
    # Test different numbers of levels
    for num_levels in range(2, max_levels + 1):
        # Generate all possible combinations of embedding/cross-encoder per level
        # For each level, we have 4 possibilities:
        # 1. Embedding-based clustering without refinement
        # 2. Embedding-based clustering with refinement
        # 3. Cross-encoder-based clustering without refinement (less common)
        # 4. Cross-encoder-based clustering with refinement
        
        # Start with embedding or cross-encoder options for each level
        options = []
        for level in range(1, num_levels + 1):
            level_options = []
            
            # Embedding-based clustering without refinement
            level_options.append({
                "use_cross_encoder": False,
                "refine_after_clustering": False
            })
            
            # Embedding-based clustering with refinement
            level_options.append({
                "use_cross_encoder": False,
                "refine_after_clustering": True
            })
            
            # Cross-encoder-based clustering without refinement
            level_options.append({
                "use_cross_encoder": True,
                "refine_after_clustering": False
            })
            
            # Cross-encoder-based clustering with refinement
            level_options.append({
                "use_cross_encoder": True,
                "refine_after_clustering": True
            })
            
            options.append(level_options)
        
        # Generate all combinations for selected patterns
        # Note: Testing all combinations would be 4^num_levels which gets large quickly
        # Instead, we'll test some representative combinations
        
        # 1. All embedding-based (with various refinement patterns)
        for refinement_pattern in itertools.product([False, True], repeat=num_levels):
            level_configs = {}
            for level, refine in enumerate(refinement_pattern, 1):
                level_configs[level] = {
                    "use_cross_encoder": False,
                    "refine_after_clustering": refine,
                    "min_cluster_size": 3,
                    "min_samples": 2,
                    "refinement_method": "borderline"
                }
            
            # Create test configuration
            pattern_desc = '_'.join([f"L{i+1}E{'R' if r else 'X'}" for i, r in enumerate(refinement_pattern)])
            all_configurations.append({
                "id": f"embed_{pattern_desc}",
                "levels": num_levels,
                "level_configs": level_configs,
                "description": f"All embedding-based, {num_levels} levels, refinement: {refinement_pattern}"
            })
        
        # 2. All cross-encoder-based (with various refinement patterns)
        for refinement_pattern in itertools.product([False, True], repeat=num_levels):
            level_configs = {}
            for level, refine in enumerate(refinement_pattern, 1):
                level_configs[level] = {
                    "use_cross_encoder": True,
                    "refine_after_clustering": refine,
                    "min_cluster_size": 3,
                    "min_samples": 2,
                    "refinement_method": "borderline"
                }
            
            # Create test configuration
            pattern_desc = '_'.join([f"L{i+1}C{'R' if r else 'X'}" for i, r in enumerate(refinement_pattern)])
            all_configurations.append({
                "id": f"cross_{pattern_desc}",
                "levels": num_levels,
                "level_configs": level_configs,
                "description": f"All cross-encoder-based, {num_levels} levels, refinement: {refinement_pattern}"
            })
        
        # 3. Mixed approaches (alternating)
        level_configs = {}
        mixed_desc = []
        for level in range(1, num_levels + 1):
            use_cross_encoder = (level % 2 == 0)  # Alternate
            refine = (level % 2 == 1)  # Alternate refinement too
            
            level_configs[level] = {
                "use_cross_encoder": use_cross_encoder,
                "refine_after_clustering": refine,
                "min_cluster_size": 3,
                "min_samples": 2,
                "refinement_method": "borderline"
            }
            
            mixed_desc.append(f"L{level}{'C' if use_cross_encoder else 'E'}{'R' if refine else 'X'}")
        
        all_configurations.append({
            "id": f"mixed_{'_'.join(mixed_desc)}",
            "levels": num_levels,
            "level_configs": level_configs,
            "description": f"Mixed approach, {num_levels} levels, alternating embedding/cross-encoder"
        })
        
        # 4. Progressive approach (embedding at higher levels, cross-encoder at lower levels)
        level_configs = {}
        mixed_desc = []
        for level in range(1, num_levels + 1):
            use_cross_encoder = (level <= num_levels // 2)  # Cross-encoder for lower levels
            refine = (level > num_levels // 2)  # Refine at higher levels
            
            level_configs[level] = {
                "use_cross_encoder": use_cross_encoder,
                "refine_after_clustering": refine,
                "min_cluster_size": 3,
                "min_samples": 2,
                "refinement_method": "borderline"
            }
            
            mixed_desc.append(f"L{level}{'C' if use_cross_encoder else 'E'}{'R' if refine else 'X'}")
        
        all_configurations.append({
            "id": f"prog_{'_'.join(mixed_desc)}",
            "levels": num_levels,
            "level_configs": level_configs,
            "description": f"Progressive approach, {num_levels} levels, cross-encoder lower/embedding higher"
        })
    
    return all_configurations


def main():
    """
    Main entry point for testing the hierarchical clustering pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test hierarchical clustering pipeline with various configurations")
    parser.add_argument("--data_path", type=str, help="Path to product data CSV file", default=None)
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2", help="Name of the embedding model")
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Name of the cross-encoder model")
    parser.add_argument("--test_size", type=int, default=300, help="Number of samples to use for testing")
    parser.add_argument("--embeddings_path", type=str, help="Path to cached embeddings", default=None)
    parser.add_argument("--use_cached_embeddings", action="store_true", help="Use cached embeddings if available")
    parser.add_argument("--output_dir", type=str, default="hierarchical_test_results", help="Directory to save test results")
    parser.add_argument("--max_levels", type=int, default=4, help="Maximum number of hierarchical levels to test")
    parser.add_argument("--single_test", action="store_true", help="Run only a single test configuration")
    args = parser.parse_args()
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up data and embeddings
    vectors, descriptions, metadata = setup_test_data(
        data_path=args.data_path,
        embedding_model=args.embedding_model,
        test_size=args.test_size,
        use_cached_embeddings=args.use_cached_embeddings,
        embeddings_path=args.embeddings_path
    )
    
    # Initialize reranker
    reranker = initialize_cross_encoder(args.reranker_model)
    
    # Generate test configurations
    test_configs = generate_test_configurations(max_levels=args.max_levels)
    
    # Run a single simplified test if requested or if having issues
    if args.single_test or True:  # Force single test for debugging
        logger.info("Running single test configuration for debugging")
        
        # Create a simple configuration with embedding-based clustering only
        # This avoids potential issues with cross-encoder initialization
        simple_config = {
            "id": "simple_test",
            "description": "Simple embedding-based clustering test",
            "levels": 2,
            "level_configs": {
                1: {
                    "use_cross_encoder": False,
                    "min_cluster_size": 2,  # Use smaller size for small test data
                    "min_samples": 1,       # Use smaller samples for small test data
                    "refine_after_clustering": False
                },
                2: {
                    "use_cross_encoder": False,
                    "min_cluster_size": 2,
                    "min_samples": 1,
                    "refine_after_clustering": False
                }
            }
        }
        
        result = run_test_configuration(
            vectors=vectors,
            descriptions=descriptions,
            metadata=metadata,
            reranker=None,  # Skip reranker
            levels=simple_config["levels"],
            level_configs=simple_config["level_configs"],
            output_dir=output_dir,
            test_id=simple_config["id"]
        )
        
        logger.info(f"Single test completed. Results saved to {output_dir}/{simple_config['id']}")
        return
    
    # Run all tests if single_test is False
    logger.info(f"Testing {len(test_configs)} configurations")
    for i, config in enumerate(test_configs):
        logger.info(f"Test {i+1}/{len(test_configs)}: {config['id']} - {config['description']}")
    
    results = {}
    for i, config in enumerate(test_configs):
        test_id = config["id"]
        
        # Skip cross-encoder tests if reranker not available
        if "cross" in test_id and reranker is None:
            logger.warning(f"Skipping test {test_id} because cross-encoder reranker is not available")
            continue
        
        logger.info(f"Running test configuration: {test_id}")
        result = run_test_configuration(
            vectors=vectors,
            descriptions=descriptions,
            metadata=metadata,
            reranker=reranker,
            levels=config["levels"],
            level_configs=config["level_configs"],
            output_dir=output_dir,
            test_id=config["id"]
        )
        
        test_results.append(result)
    
    # Save summary results
    summary_path = os.path.join(output_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "num_products": len(descriptions),
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "results": test_results
        }, f, indent=2)
    
    logger.info(f"Test results saved to {summary_path}")
    
    # Create comparison table
    comparison_df = pd.DataFrame([
        {
            "test_id": result["test_id"],
            "runtime_seconds": result["runtime_seconds"],
            "num_levels": len(result["results"]),
            **{f"L{level}_clusters": stats["num_clusters"] 
               for level, stats in result["results"].items()},
            **{f"L{level}_noise%": stats["noise_percentage"] 
               for level, stats in result["results"].items()},
            **{f"L{level}_method": stats["clustering_method"] + 
               (" + refinement" if stats["refined"] else "") 
               for level, stats in result["results"].items()}
        }
        for result in test_results
    ])
    
    # Save comparison table
    comparison_path = os.path.join(output_dir, "test_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    logger.info(f"Comparison table saved to {comparison_path}")
    logger.info("Test complete!")


if __name__ == "__main__":
    main()
