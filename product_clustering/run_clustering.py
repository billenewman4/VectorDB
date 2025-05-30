#!/usr/bin/env python3
"""
Enhanced Product Clustering Script with Configurable Parameters

This script provides a unified interface to run the entire product clustering pipeline
with highly configurable parameters for experimentation and optimization:

1. Data preparation - Control how data is preprocessed and normalized
2. Embedding generation - Select embedding models and parameters
3. Clustering - Configure HDBSCAN and other clustering parameters 
4. Refinement - Control cross-encoder reranking, weights, and thresholds
5. Evaluation and analysis - Select different analysis methods and metrics

Usage:
  python run_clustering.py --help                  # Show help message
  python run_clustering.py --all                   # Run complete pipeline with default settings
  python run_clustering.py --prepare --embed       # Only prepare data and generate embeddings
  python run_clustering.py --cluster --rerank      # Only run clustering with reranking
  python run_clustering.py --analyze               # Only analyze results
  python run_clustering.py --test --sample_size 500 # Run in test mode with 500 samples
  
  # Example with custom parameters:
  python run_clustering.py --all \
    --embedding_model all-mpnet-base-v2 \
    --min_cluster_size 3 \
    --min_samples 2 \
    --rerank \
    --rerank_weight 0.7 \
    --similarity_threshold 0.65 \
    --analyze_margins --analyze_usda
"""

import os
import sys
import argparse
import time
from typing import Optional, Dict, Any
import json

# Local module for interactive input
from product_clustering.interactive_input import (
    get_yes_no_input,
    get_string_input, 
    get_int_input, 
    get_float_input
)

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_data_preparation(data_dir: Optional[str] = None, 
                      force: bool = False,
                      use_category_descriptions: bool = True,
                      normalize_text: bool = True,
                      expand_abbreviations: bool = True):
    """
    Run the data preparation step.
    
    Args:
        data_dir: Directory to store prepared data
        force: Whether to force reprocessing even if files exist
        use_category_descriptions: Whether to use category descriptions in clustering
        normalize_text: Whether to normalize text descriptions
        expand_abbreviations: Whether to expand abbreviations in descriptions
    """
    from product_clustering.data_prep import prepare_data_for_clustering
    import pandas as pd
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    output_path = os.path.join(data_dir, "prepared_products.csv")
    
    if os.path.exists(output_path) and not force:
        print(f"Prepared data already exists at {output_path}")
        print("Use --force to reprocess")
        return output_path
    
    print("Running data preparation...")
    # Call the function to prepare data for clustering
    prepared_data = prepare_data_for_clustering()
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Modify the clustering description based on options
    if not use_category_descriptions:
        print("Excluding category descriptions from clustering...")
        # Remove category information from clustering description if requested
        prepared_data['clustering_description'] = prepared_data['product_description']
        # Re-apply normalization if needed
        if normalize_text:
            from product_clustering.data_prep import preprocess_text_for_clustering
            prepared_data['clustering_description'] = prepared_data['clustering_description'].apply(
                lambda x: preprocess_text_for_clustering(x, expand_abbreviations=expand_abbreviations)
            )
    
    # Save the prepared data
    prepared_data.to_csv(output_path, index=False)
    print(f"Data preparation complete. Results saved to {output_path}")
    print(f"Used category descriptions: {use_category_descriptions}")
    print(f"Applied text normalization: {normalize_text}")
    print(f"Expanded abbreviations: {expand_abbreviations}")
    
    return output_path

def run_embedding_generation(data_dir: Optional[str] = None, 
                            prepared_data_path: Optional[str] = None,
                            model_name: str = "all-mpnet-base-v2",
                            embedding_batch_size: int = 32,
                            embedding_normalize: bool = True,
                            force: bool = False):
    """
    Run the embedding generation step.
    
    Args:
        data_dir: Directory to store embeddings
        prepared_data_path: Path to prepared data CSV
        model_name: Name of the embedding model to use
        embedding_batch_size: Batch size for embedding generation
        embedding_normalize: Whether to normalize embeddings
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
    # Generate embeddings with configurable parameters
    embeddings, product_codes = embed_products(
        df=df,
        embedding_type='sentence-transformer',
        model_name=model_name,
        batch_size=embedding_batch_size,
        normalize_embeddings=embedding_normalize
    )
    
    # Save the embeddings to files
    save_embeddings(embeddings, product_codes, data_dir)
    
    print(f"Embedding generation complete. Results saved to {embeddings_path}")
    
    return embeddings_path, product_codes_path

def run_clustering(data_dir: Optional[str] = None,
                  # HDBSCAN clustering parameters
                  metric: str = "euclidean",
                  min_cluster_size: int = 3,
                  min_samples: int = 2,
                  cluster_selection_epsilon: float = 0.0,
                  alpha: float = 1.0,
                  cluster_selection_method: str = "eom",
                  # Testing parameters
                  test_mode: bool = False,
                  sample_size: int = 1000,
                  # Cross-encoder reranking parameters
                  use_reranking: bool = False,
                  cross_encoder_model: str = "cross-encoder/stsb-roberta-base",
                  cross_encoder_batch_size: int = 32,
                  similarity_threshold: float = 0.6,
                  rerank_weight: float = 0.5,  # Weight between original embeddings and cross-encoder scores
                  test_clusters: int = 0,  # Number of clusters to test reranking on (0 = all clusters)
                  min_cluster_size_for_reranking: int = 3,  # Minimum cluster size to consider for reranking
                  # Category handling
                  use_categories: bool = True,
                  category_exclusivity: float = 1.0,  # How strict to keep products within categories
                  # Processing control
                  force: bool = False,
                  n_jobs: int = -1):  # Number of CPU cores to use (-1 for all)
    """
    Run the clustering step with optional reranking.
    
    Args:
        data_dir: Directory containing data files
        
        # HDBSCAN clustering parameters
        metric: Distance metric to use ('euclidean', 'manhattan', 'cosine', etc.)
        min_cluster_size: Minimum size of clusters
        min_samples: HDBSCAN min_samples parameter (higher = more strict clustering)
        cluster_selection_epsilon: Allow points closer than this to join existing clusters
        alpha: Scaling parameter for how conservative to be in preserving clusters
        cluster_selection_method: Method to extract clusters ('eom' or 'leaf')
        
        # Testing parameters
        test_mode: Whether to run in test mode on a subset of data
        sample_size: Number of samples to use in test mode
        
        # Cross-encoder reranking parameters
        use_reranking: Whether to use CrossEncoder reranking
        cross_encoder_model: Model for reranking
        cross_encoder_batch_size: Batch size for cross-encoder inference
        similarity_threshold: Threshold for similarity (higher = more strict matching)
        rerank_weight: Weight between embeddings and cross-encoder (0=only embeddings, 1=only cross-encoder)
        test_clusters: Number of clusters to test reranking on (0 = all clusters)
        min_cluster_size_for_reranking: Minimum cluster size to consider for reranking scores
        
        # Category handling
        use_categories: Whether to cluster by categories
        category_exclusivity: How strictly to keep products within categories
                             (0 = mix freely, 1 = strict category separation)
        
        # Processing control
        force: Whether to force reprocessing even if results exist
        n_jobs: Number of CPU cores to use (-1 for all)
    """
    from product_clustering.improved_clustering import run_improved_clustering
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Determine output directory
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
    
    print("Running clustering...")
    start_time = time.time()
    
    clusters_path = run_improved_clustering(
        data_dir=data_dir,
        # HDBSCAN parameters
        metric=metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
        cluster_selection_method=cluster_selection_method,
        # Testing parameters
        test_mode=test_mode,
        sample_size=sample_size,
        # Reranking parameters
        use_reranking=use_reranking,
        cross_encoder_model=cross_encoder_model,
        cross_encoder_batch_size=cross_encoder_batch_size,
        similarity_threshold=similarity_threshold,
        rerank_weight=rerank_weight,
        test_clusters=test_clusters,
        min_cluster_size_for_reranking=min_cluster_size_for_reranking,
        # Category parameters
        use_categories=use_categories,
        category_exclusivity=category_exclusivity,
        # Processing parameters
        force=force,
        n_jobs=n_jobs
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

def run_analysis(data_dir: Optional[str] = None, 
                 refined: bool = True,
                 run_basic_analysis: bool = True,
                 run_margin_analysis: bool = False,
                 run_usda_analysis: bool = False,
                 run_llm_analysis: bool = False,
                 llm_model: str = "gpt-3.5-turbo",
                 cluster_size_threshold: int = 5,
                 price_variation_threshold: float = 0.2,
                 detailed_output: bool = False):
    """
    Run various analysis steps on clustering results.
    
    Args:
        data_dir: Directory containing clustering results
        refined: Whether to analyze refined clusters
        run_basic_analysis: Whether to run basic cluster statistics analysis
        run_margin_analysis: Whether to analyze price/margin variations within clusters
        run_usda_analysis: Whether to analyze USDA mapping alignment
        run_llm_analysis: Whether to use LLM for analyzing cluster coherence
        llm_model: LLM model to use for analysis if run_llm_analysis is True
        cluster_size_threshold: Minimum cluster size for detailed analysis
        price_variation_threshold: Threshold for identifying significant price variations
        detailed_output: Whether to generate detailed analysis output
    
    Returns:
        Dictionary of paths to the generated analysis files
    """
    from product_clustering.analyze_clusters import run_cluster_analysis
    import importlib.util
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Determine the path to clusters based on whether we're using refined clusters
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
    
    # Create a dictionary to store output paths
    analysis_paths = {}
    
    # Run the basic cluster analysis
    if run_basic_analysis:
        print(f"Running basic analysis on {'refined ' if refined else ''}clusters...")
        # Note: run_cluster_analysis does not accept 'detailed' or 'min_cluster_size' parameters
        # so we will just use the parameters it does accept
        output_path = run_cluster_analysis(
            clusters_path=clusters_path,
            data_dir=data_dir,
            refined=refined
        )
        analysis_paths['basic'] = output_path
        print(f"Basic analysis complete. Results saved to {output_path}")
    
    # Run margin analysis if requested
    if run_margin_analysis:
        # Check if the margin analysis module exists and import it dynamically
        margin_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "margin_analysis.py")
        if os.path.exists(margin_module_path):
            try:
                print("Running margin analysis on clusters...")
                spec = importlib.util.spec_from_file_location("margin_analysis", margin_module_path)
                margin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(margin_module)
                
                margin_output_path = margin_module.analyze_margins(
                    clusters_path=clusters_path,
                    data_dir=data_dir,
                    price_variation_threshold=price_variation_threshold,
                    detailed=detailed_output
                )
                
                analysis_paths['margin'] = margin_output_path
                print(f"Margin analysis complete. Results saved to {margin_output_path}")
            except Exception as e:
                print(f"Error running margin analysis: {e}")
        else:
            print("Margin analysis module not found. Skipping margin analysis.")
    
    # Run USDA mapping analysis if requested
    if run_usda_analysis:
        try:
            print("Running USDA mapping analysis...")
            from product_clustering.analyze_usda_mapping import analyze_usda_grouping_alignment, generate_usda_mapping_report
            
            usda_analysis_path = os.path.join(data_dir, "analysis", "usda_mapping_analysis.md")
            os.makedirs(os.path.dirname(usda_analysis_path), exist_ok=True)
            
            # Run the USDA mapping analysis using the analyze_usda_mapping module
            usda_output_path = os.path.join(data_dir, "analysis", "usda_mapping_analysis.md")
            
            # Use the command-line tool for convenience
            import subprocess
            cmd = [
                "python3", 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_usda_mapping.py"),
                f"--clusters_path={clusters_path}",
                f"--output_dir={os.path.join(data_dir, 'analysis')}"
            ]
            subprocess.run(cmd)
            
            analysis_paths['usda'] = usda_output_path
            print(f"USDA mapping analysis complete. Results saved to {usda_output_path}")
        except Exception as e:
            print(f"Error running USDA mapping analysis: {e}")
    
    # Run LLM analysis if requested
    if run_llm_analysis:
        try:
            print(f"Running LLM analysis using {llm_model}...")
            
            # Check if the cluster_analyzer_llm module exists
            llm_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_analyzer_llm.py")
            if os.path.exists(llm_module_path):
                spec = importlib.util.spec_from_file_location("cluster_analyzer_llm", llm_module_path)
                llm_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(llm_module)
                
                llm_output_path = llm_module.analyze_clusters_with_llm(
                    clusters_path=clusters_path,
                    data_dir=data_dir,
                    model_name=llm_model,
                    min_cluster_size=cluster_size_threshold
                )
                
                analysis_paths['llm'] = llm_output_path
                print(f"LLM analysis complete. Results saved to {llm_output_path}")
            else:
                print("LLM analysis module not found. Skipping LLM analysis.")
        except Exception as e:
            print(f"Error running LLM analysis: {e}")
    
    return analysis_paths

def main():
    """Main function to run the product clustering pipeline."""
    parser = argparse.ArgumentParser(description="Run product clustering pipeline")
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode Options')
    mode_group.add_argument("--interactive", action="store_true", 
                      help="Run in interactive mode, prompting for parameters")
    mode_group.add_argument("--save_config", type=str, metavar="CONFIG_FILE",
                      help="Save current parameters to a config file")
    mode_group.add_argument("--load_config", type=str, metavar="CONFIG_FILE",
                      help="Load parameters from a config file")
    
    # Pipeline steps
    pipeline_group = parser.add_argument_group('Pipeline Steps')
    pipeline_group.add_argument("--all", action="store_true", help="Run complete pipeline")
    pipeline_group.add_argument("--prepare", action="store_true", help="Run data preparation")
    pipeline_group.add_argument("--embed", action="store_true", help="Generate embeddings")
    pipeline_group.add_argument("--cluster", action="store_true", help="Run clustering")
    pipeline_group.add_argument("--analyze", action="store_true", help="Analyze results")
    
    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument("--data_dir", help="Directory for data files")
    data_group.add_argument("--force", action="store_true", default=True, 
                        help="Force reprocessing even if files exist (default: True)")
    data_group.add_argument("--no_force", action="store_false", dest="force",
                        help="Use cached files when available instead of reprocessing")
    data_group.add_argument("--no_category_descriptions", action="store_false", dest="use_category_descriptions",
                        help="Exclude category descriptions from clustering (use only product descriptions)")
    data_group.add_argument("--no_text_normalization", action="store_false", dest="normalize_text",
                        help="Disable text normalization in descriptions")
    data_group.add_argument("--no_expand_abbreviations", action="store_false", dest="expand_abbreviations",
                        help="Disable expansion of abbreviations in descriptions")
    
    # Embedding options
    embed_group = parser.add_argument_group('Embedding Options')
    embed_group.add_argument("--embedding_model", default="all-mpnet-base-v2", help="Embedding model to use")
    embed_group.add_argument("--embedding_batch_size", type=int, default=32, 
                        help="Batch size for embedding generation")
    embed_group.add_argument("--no_normalize_embeddings", action="store_false", dest="normalize_embeddings",
                        help="Disable embedding normalization")
    
    # HDBSCAN clustering options
    cluster_group = parser.add_argument_group('HDBSCAN Clustering Options')
    cluster_group.add_argument("--metric", default="euclidean", 
                        choices=["euclidean", "manhattan", "cosine", "minkowski"], 
                        help="Distance metric to use")
    cluster_group.add_argument("--min_cluster_size", type=int, default=3, 
                        help="Minimum size of clusters")
    cluster_group.add_argument("--min_samples", type=int, default=2, 
                        help="HDBSCAN min_samples parameter (higher = more strict clustering)")
    cluster_group.add_argument("--cluster_selection_epsilon", type=float, default=0.0,
                        help="Distance threshold for cluster merging")
    cluster_group.add_argument("--alpha", type=float, default=1.0,
                        help="HDBSCAN alpha parameter for point weighting")
    cluster_group.add_argument("--cluster_selection_method", default="eom", choices=["eom", "leaf"],
                        help="Algorithm for cluster extraction")
    cluster_group.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of CPU cores to use (-1 for all)")
    
    # Test mode options
    test_group = parser.add_argument_group('Test Mode Options')
    test_group.add_argument("--test", action="store_true", help="Run in test mode with reduced dataset")
    test_group.add_argument("--sample_size", type=int, default=1000, help="Number of samples to use in test mode")
    
    # Reranking options
    rerank_group = parser.add_argument_group('Cross-Encoder Reranking Options')
    rerank_group.add_argument("--rerank", action="store_true",
                        help="Use CrossEncoder reranking to refine clusters")
    rerank_group.add_argument("--cross_encoder", default="cross-encoder/stsb-roberta-base",
                        help="CrossEncoder model to use for reranking")
    rerank_group.add_argument("--cross_encoder_batch_size", type=int, default=32,
                        help="Batch size for cross-encoder inference")
    rerank_group.add_argument("--similarity_threshold", type=float, default=0.6,
                        help="Similarity threshold (higher = more strict matching)")
    rerank_group.add_argument("--rerank_weight", type=float, default=0.5,
                        help="Weight between embeddings and cross-encoder (0=only embeddings, 1=only cross-encoder)")
    rerank_group.add_argument("--test_clusters", type=int, default=0,
                        help="Number of clusters to test reranking on (0 = all clusters)")
    rerank_group.add_argument("--min_cluster_size_for_reranking", type=int, default=3,
                        help="Minimum cluster size to consider for reranking")
    
    # Category options
    category_group = parser.add_argument_group('Category Options')
    category_group.add_argument("--no_categories", action="store_true", 
                        help="Disable category-based clustering (allows mixing products across categories)")
    category_group.add_argument("--category_exclusivity", type=float, default=1.0,
                        help="How strictly to keep products within categories (0=mix freely, 1=strict separation)")
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--refined", action="store_true", default=True,
                        help="Analyze refined clusters (default)")
    analysis_group.add_argument("--no_refined", action="store_false", dest="refined",
                        help="Analyze original clusters instead of refined")
    analysis_group.add_argument("--analyze_basic", action="store_true", default=True,
                        help="Run basic cluster statistics analysis (default)")
    analysis_group.add_argument("--analyze_margins", action="store_true", 
                        help="Analyze price/margin variations within clusters")
    analysis_group.add_argument("--analyze_usda", action="store_true", 
                        help="Analyze USDA mapping alignment")
    analysis_group.add_argument("--analyze_llm", action="store_true", 
                        help="Use LLM for analyzing cluster coherence")
    analysis_group.add_argument("--llm_model", default="gpt-3.5-turbo", 
                        help="LLM model to use for analysis")
    analysis_group.add_argument("--cluster_size_threshold", type=int, default=5, 
                        help="Minimum cluster size for detailed analysis")
    analysis_group.add_argument("--price_variation_threshold", type=float, default=0.2, 
                        help="Threshold for identifying significant price variations")
    analysis_group.add_argument("--detailed_output", action="store_true", 
                        help="Generate detailed analysis output")
    
    # Parse arguments
    args = parser.parse_args()
    
    # For backward compatibility
    if hasattr(args, 'model') and not hasattr(args, 'embedding_model'):
        args.embedding_model = args.model
    
    # Handle configuration loading if specified
    if hasattr(args, 'load_config') and args.load_config:
        from interactive_config import load_config
        config = load_config(args.load_config)
        # Update args with loaded config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Run interactive mode if requested
    if hasattr(args, 'interactive') and args.interactive:
        print("\n===== Interactive Configuration Mode =====\n")
        print("You'll be prompted for various configuration parameters.")
        print("Press Enter to accept default values shown in [brackets].\n")
        
        from interactive_config import (
            get_processing_options,
            get_data_preparation_params,
            get_embedding_params,
            get_clustering_params,
            get_reranking_params,
            get_category_params,
            get_analysis_params,
            save_config
        )
        
        # Only prompt for parameters related to selected pipeline steps
        config = {}
        
        # Always get processing options first
        config.update(get_processing_options(args))
        
        if args.all or args.prepare:
            config.update(get_data_preparation_params(args))
        
        if args.all or args.embed:
            config.update(get_embedding_params(args))
        
        if args.all or args.cluster:
            config.update(get_clustering_params(args))
            config.update(get_reranking_params(args))
            config.update(get_category_params(args))
        
        if args.all or args.analyze:
            config.update(get_analysis_params(args))
        
        # Update args with interactively provided values
        for key, value in config.items():
            setattr(args, key, value)
        
        # Save configuration if requested
        if hasattr(args, 'save_config') and args.save_config:
            save_config(config, args.save_config)
    
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
        prepared_data_path = run_data_preparation(
            data_dir=data_dir, 
            force=args.force,
            use_category_descriptions=args.use_category_descriptions,
            normalize_text=args.normalize_text,
            expand_abbreviations=args.expand_abbreviations
        )
    
    if args.all or args.embed:
        embeddings_path, product_codes_path = run_embedding_generation(
            data_dir=data_dir, 
            prepared_data_path=prepared_data_path if 'prepared_data_path' in locals() else None,
            model_name=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_normalize=args.normalize_embeddings,
            force=args.force
        )
    
    if args.all or args.cluster:
        clusters_path = run_clustering(
            data_dir=data_dir,
            # HDBSCAN parameters
            metric=args.metric,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            alpha=args.alpha,
            cluster_selection_method=args.cluster_selection_method,
            # Testing parameters
            test_mode=args.test,
            sample_size=args.sample_size,
            # Reranking parameters
            use_reranking=args.rerank,
            cross_encoder_model=args.cross_encoder,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            similarity_threshold=args.similarity_threshold,
            rerank_weight=args.rerank_weight,
            # Category parameters
            use_categories=not args.no_categories,
            category_exclusivity=args.category_exclusivity,
            # Processing parameters
            force=args.force,
            n_jobs=args.n_jobs
        )
    
    if args.all or args.analyze:
        analysis_paths = run_analysis(
            data_dir=data_dir, 
            refined=args.refined,
            run_basic_analysis=args.analyze_basic,
            run_margin_analysis=args.analyze_margins,
            run_usda_analysis=args.analyze_usda,
            run_llm_analysis=args.analyze_llm,
            llm_model=args.llm_model,
            cluster_size_threshold=args.cluster_size_threshold,
            price_variation_threshold=args.price_variation_threshold,
            detailed_output=args.detailed_output
        )
    
    print("\nProduct clustering pipeline completed successfully!")

if __name__ == "__main__":
    main()
