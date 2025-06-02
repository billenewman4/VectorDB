"""
Main entry point for the hierarchical clustering pipeline.

This script integrates data preparation, embedding, clustering, and results export
into a complete end-to-end pipeline for hierarchical product clustering.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import json

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import components from various modules
from Data_prep.prepare_data import prepare_data_for_clustering
from Vector_Embedding.sentence_transformer_encoder import SentenceTransformerEncoder
from Pipeline.hierarchical_pipeline import HierarchicalClusteringPipeline
from Clustering.Analytics.visualization import ClusterVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments.
    
    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(description="Hierarchical Product Clustering Pipeline")
    
    # Core arguments
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--data_path", type=str, help="Path to product data CSV (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    
    # Pipeline control
    parser.add_argument("--levels", type=int, help="Number of hierarchical levels (overrides config)")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with sample data")
    parser.add_argument("--test_samples", type=int, help="Number of samples for test mode")
    
    # Embedding options
    parser.add_argument("--embedding_model", type=str, help="Embedding model name (overrides config)")
    parser.add_argument("--use_cached_embeddings", action="store_true", help="Use cached embeddings if available")
    parser.add_argument("--embeddings_path", type=str, help="Path to cached embeddings")
    
    # Clustering method options for each level
    parser.add_argument("--l1_cross_encoder", action="store_true", help="Use cross-encoder for level 1")
    parser.add_argument("--l2_cross_encoder", action="store_true", help="Use cross-encoder for level 2")
    parser.add_argument("--l3_cross_encoder", action="store_true", help="Use cross-encoder for level 3")
    parser.add_argument("--l4_cross_encoder", action="store_true", help="Use cross-encoder for level 4")
    
    # Refinement options for each level
    parser.add_argument("--refine_l1", action="store_true", help="Apply refinement after level 1 clustering")
    parser.add_argument("--refine_l2", action="store_true", help="Apply refinement after level 2 clustering")
    parser.add_argument("--refine_l3", action="store_true", help="Apply refinement after level 3 clustering")
    parser.add_argument("--refine_l4", action="store_true", help="Apply refinement after level 4 clustering")
    
    # Cross-encoder options
    parser.add_argument("--reranker_model", type=str, help="Cross-encoder model name (overrides config)")
    
    # Output options
    parser.add_argument("--save_plots", action="store_true", help="Save visualization plots")
    parser.add_argument("--skip_visualizations", action="store_true", help="Skip generating visualizations")
    
    args = parser.parse_args()
    print("\n==============================================================")
    print("=== STARTING HIERARCHICAL CLUSTERING PIPELINE - DIRECT LOG ====")
    print("==============================================================\n")
    
    import time
    pipeline_start = time.time()
    
    return vars(args)


def prepare_pipeline_config(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare hierarchical pipeline configuration by merging config file with command line arguments.
    
    Args:
        config: Configuration from YAML file
        args: Command line arguments
        
    Returns:
        Combined configuration dictionary
    """
    # Start with clustering config from file
    pipeline_config = config.get("clustering", {})
    
    # Add level-specific configurations
    if "level_configs" not in pipeline_config:
        pipeline_config["level_configs"] = {}
    
    # Override number of levels if specified
    if args.get("levels"):
        pipeline_config["levels"] = args["levels"]
    
    # Get the total number of levels
    num_levels = pipeline_config.get("levels", 3)
    
    # Configure each level based on command line arguments
    for level in range(1, num_levels + 1):
        if level not in pipeline_config["level_configs"]:
            pipeline_config["level_configs"][level] = {}
            
        # Set cross-encoder usage for this level if specified
        cross_encoder_arg = f"l{level}_cross_encoder"
        if args.get(cross_encoder_arg):
            pipeline_config["level_configs"][level]["use_cross_encoder"] = True
            
        # Set refinement for this level if specified
        refine_arg = f"refine_l{level}"
        if args.get(refine_arg):
            pipeline_config["level_configs"][level]["refine_after_clustering"] = True
    
    # Add cross-encoder configuration
    pipeline_config["cross_encoder"] = config.get("cross_encoder", {})
    
    # Override cross-encoder model if specified
    if args.get("reranker_model"):
        pipeline_config["cross_encoder"]["model_name"] = args["reranker_model"]
        
    # Set global refinement flag if any level uses refinement
    any_refinement = any([
        args.get(f"refine_l{level}") for level in range(1, num_levels + 1)
    ])
    any_cross_encoder = any([
        args.get(f"l{level}_cross_encoder") for level in range(1, num_levels + 1)
    ])
    
    if any_refinement or any_cross_encoder:
        pipeline_config["cross_encoder"]["use_refinement"] = True
    
    # Add visualization configuration
    pipeline_config["visualization"] = config.get("visualization", {})
    
    # Override visualization settings if specified
    if args.get("save_plots"):
        pipeline_config["visualization"]["save_plots"] = True
        pipeline_config["visualization"]["create_visualizations"] = True
    
    if args.get("skip_visualizations"):
        pipeline_config["visualization"]["create_visualizations"] = False
        
    # Add output configuration
    pipeline_config["output"] = config.get("output", {})
    
    # Override output directory if specified
    if args.get("output_dir"):
        pipeline_config["output"]["output_dir"] = args["output_dir"]
        
    return pipeline_config


def prepare_data(config: Dict[str, Any], args: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare product data for clustering by delegating to Data_Prep module.
    This function acts as an orchestrator, avoiding redundant code.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Prepared DataFrame with product data
    """
    # Get data configuration
    data_config = config.get("data_preparation", {})
    
    # Override data path if specified in command line
    data_path = args.get("data_path") or data_config.get("data_path")
    if not data_path:
        raise ValueError("No data path specified in config or command line arguments")
    
    # Check if data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Determine if we're running in test mode
    test_mode = args.get("test_mode") or data_config.get("test_mode", False)
    test_samples = args.get("test_samples") or data_config.get("test_samples", 500)
    
    # Import the data preparation function from the dedicated module
    logger.info("Delegating data preparation to Data_prep module")
    from Data_prep.prepare_data import prepare_data_for_clustering
    
    # Let the dedicated module handle all data loading and preparation
    # This eliminates redundant code in the pipeline and ensures proper separation of concerns
    df_prepared = prepare_data_for_clustering(
        # We pass None as df_raw, letting the prepare_data_for_clustering function handle loading
        df_raw=None,  
        file_path=data_path,  # Pass the file path for the module to use
        use_category_descriptions=data_config.get("use_category_descriptions", True),
        normalize_text=data_config.get("normalize_text", True), 
        expand_abbreviations=data_config.get("expand_abbreviations", True),
        test_mode=test_mode,
        test_sample_size=test_samples
    )
    
    if df_prepared is None or df_prepared.empty:
        raise ValueError(f"Data preparation failed: No data returned from prepare_data_for_clustering")
    
    logger.info(f"Data preparation complete: {len(df_prepared)} products ready for clustering")
    return df_prepared


def generate_embeddings(df: pd.DataFrame, config: Dict[str, Any], args: Dict[str, Any]) -> np.ndarray:
    """
    Generate embeddings for product descriptions.
    
    Args:
        df: DataFrame with prepared product data
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        NumPy array of embeddings
    """
    print("\n==== STARTING GENERATE_EMBEDDINGS FUNCTION ====\n")
    start_time = time.time()
    
    # Get embedding configuration
    embedding_config = config.get("embedding", {})
    
    # Check for cached embeddings first
    use_cached = args.get("use_cached_embeddings") or embedding_config.get("cache_embeddings", False)
    embeddings_path = args.get("embeddings_path") or embedding_config.get("embeddings_path")
    
    if use_cached and embeddings_path and os.path.exists(embeddings_path):
        logger.info(f"Loading cached embeddings from {embeddings_path}")
        load_start = time.time()
        embeddings = np.load(embeddings_path)
        vectors = embeddings['embeddings']
        
        # Verify dimensions match
        if len(df) != vectors.shape[0]:
            logger.warning(f"Cached embeddings size ({vectors.shape[0]}) doesn't match data size ({len(df)})")
            logger.warning("Regenerating embeddings...")
            use_cached = False
        else:
            logger.info(f"Loaded {vectors.shape[0]} embeddings with dimension {vectors.shape[1]}")
            print(f"Loaded cached embeddings in {time.time() - load_start:.2f} seconds")
            return vectors
    
    # Generate embeddings if needed
    model_name = args.get("embedding_model") or embedding_config.get("model_name", "all-mpnet-base-v2")
    logger.info(f"Generating embeddings using model: {model_name}")
    
    # Get product descriptions - use the appropriate column name
    description_col = None
    if 'clustering_description' in df.columns:
        description_col = 'clustering_description'
    elif 'product_description' in df.columns:
        description_col = 'product_description'
    else:
        description_col = 'description'
            
    descriptions = df[description_col].tolist()
    logger.info(f"Using {description_col} column for embeddings generation")
    
    # Initialize encoder
    encoder = SentenceTransformerEncoder(model_name=model_name)
    
    # Generate embeddings
    batch_size = embedding_config.get("embedding_batch_size", 32)
    vectors = encoder.encode_batch(descriptions, batch_size=batch_size)
    
    # Save embeddings if caching is enabled
    if embedding_config.get("cache_embeddings", True) and embeddings_path:
        logger.info(f"Saving embeddings to {embeddings_path}")
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.savez_compressed(embeddings_path, embeddings=vectors)
    
    logger.info(f"Generated {vectors.shape[0]} embeddings with dimension {vectors.shape[1]}")
    return vectors


def initialize_cross_encoder(config: Dict[str, Any], args: Dict[str, Any]) -> Optional[Any]:
    """
    Initialize cross-encoder model if needed.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Initialized cross-encoder or None if not needed
    """
    # Check if cross-encoder is needed
    cross_encoder_config = config.get("cross_encoder", {})
    
    # Get model name
    model_name = args.get("reranker_model") or cross_encoder_config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Check if any level is using cross-encoder
    num_levels = config.get("clustering", {}).get("levels", 3)
    any_cross_encoder = any([
        args.get(f"l{level}_cross_encoder") for level in range(1, num_levels + 1)
    ])
    
    # Check if any level is using refinement
    any_refinement = any([
        args.get(f"refine_l{level}") for level in range(1, num_levels + 1)
    ])
    
    # Initialize cross-encoder if needed
    if any_cross_encoder or any_refinement or cross_encoder_config.get("use_refinement", False):
        try:
            from sentence_transformers import CrossEncoder
            from Clean_Code.Vector_Embedding.cross_encoder_wrapper import CrossEncoderWrapper
            logger.info(f"Initializing cross-encoder with model: {model_name}")
            # Create CrossEncoder and wrap it to add compute_similarity method
            cross_encoder = CrossEncoder(model_name)
            return CrossEncoderWrapper(cross_encoder)
        except ImportError:
            logger.warning("sentence-transformers not installed. Cannot use CrossEncoder.")
            logger.warning("Install with: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"Failed to initialize CrossEncoder: {str(e)}")
    
    return None


# Import our new export module
from Clean_Code.Analysis.Export.cluster_export import export_clusters_to_csv, generate_cluster_summary


def main():
    """
    Main entry point for the hierarchical clustering pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args["config"])
    print(f"Loaded config: use_category_descriptions={config['data_preparation'].get('use_category_descriptions', True)}, test_mode={config['data_preparation'].get('test_mode', False)}")
    
    try:
        # Prepare data
        df = prepare_data(config, args)
        
        # Generate embeddings
        vectors = generate_embeddings(df, config, args)
        
        # Extract descriptions for cross-encoder - use the appropriate column name
        description_col = None
        if 'clustering_description' in df.columns:
            description_col = 'clustering_description'
        elif 'product_description' in df.columns:
            description_col = 'product_description'
        else:
            description_col = 'description'
            
        descriptions = df[description_col].tolist()
        
        # Prepare metadata for each product
        metadata = []
        for _, row in df.iterrows():
            item = {col: row[col] for col in df.columns if col != description_col}
            metadata.append(item)
        
        # Initialize cross-encoder if needed
        reranker = initialize_cross_encoder(config, args)
        
        # Prepare pipeline configuration
        pipeline_config = prepare_pipeline_config(config, args)
        
        # Initialize hierarchical clustering pipeline
        logger.info("Initializing hierarchical clustering pipeline...")
        pipeline = HierarchicalClusteringPipeline(pipeline_config)
        
        print("\n=== STARTING CLUSTERING PIPELINE RUN ===\n")
        cluster_start = time.time()
        
        # Run hierarchical clustering
        logger.info("Running hierarchical clustering...")
        results = pipeline.run(vectors, descriptions, reranker, metadata)
        
        print(f"\n=== CLUSTERING COMPLETE! Total time: {time.time() - cluster_start:.2f} seconds ===\n")
        print(f"=== FULL PIPELINE EXECUTION TIME: {time.time() - pipeline_start:.2f} seconds ===\n")
        
        # Calculate elapsed time
        elapsed_time = time.time() - cluster_start
        logger.info(f"Hierarchical clustering complete in {elapsed_time:.2f} seconds!")
        
        # Print summary
        for level, level_results in results["levels"].items():
            stats = level_results.get("statistics", {})
            num_clusters = stats.get("num_clusters", 0)
            points_assigned = stats.get("points_assigned", 0)
            noise_percentage = stats.get("noise_percentage", 0)
            clustering_method = stats.get("clustering_method", "unknown")
            refined = stats.get("refined", False)
            
            method_str = f"{clustering_method} + refinement" if refined else clustering_method
            logger.info(f"Level {level}: {num_clusters} clusters, {points_assigned}/{len(descriptions)} " +
                       f"points assigned ({noise_percentage:.2f}% noise) [Method: {method_str}]")
        
        # Get output directory
        output_dir = pipeline.results.get("summary", {}).get("output_dir", 
                                                           pipeline_config["output"].get("output_dir", 
                                                                                      "hierarchical_clustering_results"))
        
        # Export clusters to CSV
        csv_path = export_clusters_to_csv(results, df, output_dir)
        
        # Generate cluster summary
        summary_path = generate_cluster_summary(results, df, output_dir)
        
        # Print paths to results
        logger.info("\nResults saved to:")
        logger.info(f"- Cluster assignments CSV: {csv_path}")
        logger.info(f"- Cluster summary: {summary_path}")
        logger.info(f"- Full results directory: {output_dir}")
        
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
                
    except Exception as e:
        logger.error(f"Error in hierarchical clustering pipeline: {str(e)}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
