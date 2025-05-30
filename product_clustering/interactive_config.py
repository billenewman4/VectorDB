#!/usr/bin/env python3
"""
Interactive configuration module for run_clustering.py.
Provides functions for interactive parameter configuration.
"""

import json
from typing import Dict, Any
import os
from interactive_input import (
    get_yes_no_input,
    get_string_input,
    get_int_input,
    get_float_input
)

def get_processing_options(args) -> Dict[str, Any]:
    """Get general processing options interactively."""
    print("\n=== Processing Options ===\n")
    params = {}
    
    params['force'] = get_yes_no_input(
        "Force reprocessing of all steps (ignore cached files)?", 
        default=getattr(args, 'force', True)  # Default to True to force reprocessing
    )
    
    return params

def get_data_preparation_params(args) -> Dict[str, Any]:
    """Get data preparation parameters interactively."""
    print("\n=== Data Preparation Configuration ===")
    params = {}
    
    params['use_category_descriptions'] = get_yes_no_input(
        "Use category descriptions for clustering?", 
        default=getattr(args, 'use_category_descriptions', True)
    )
    
    params['normalize_text'] = get_yes_no_input(
        "Apply text normalization to descriptions?", 
        default=getattr(args, 'normalize_text', True)
    )
    
    params['expand_abbreviations'] = get_yes_no_input(
        "Expand abbreviations in descriptions?", 
        default=getattr(args, 'expand_abbreviations', True)
    )
    
    return params

def get_embedding_params(args) -> Dict[str, Any]:
    """Get embedding generation parameters interactively."""
    print("\n=== Embedding Configuration ===")
    params = {}
    
    params['embedding_model'] = get_string_input(
        "Embedding model to use",
        default=getattr(args, 'embedding_model', 'all-mpnet-base-v2'),
        options=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'paraphrase-mpnet-base-v2']
    )
    
    params['embedding_batch_size'] = get_int_input(
        "Batch size for embedding generation",
        default=getattr(args, 'embedding_batch_size', 32),
        min_val=1
    )
    
    params['normalize_embeddings'] = get_yes_no_input(
        "Normalize embeddings?",
        default=getattr(args, 'normalize_embeddings', True)
    )
    
    return params

def get_clustering_params(args) -> Dict[str, Any]:
    """Get clustering parameters interactively."""
    print("\n=== Clustering Configuration ===")
    params = {}
    
    params['metric'] = get_string_input(
        "Distance metric to use",
        default=getattr(args, 'metric', 'euclidean'),
        options=['euclidean', 'manhattan', 'cosine', 'minkowski']
    )
    
    # Use the optimized parameters from memory as defaults
    params['min_cluster_size'] = get_int_input(
        "Minimum cluster size",
        default=getattr(args, 'min_cluster_size', 3),
        min_val=2
    )
    
    params['min_samples'] = get_int_input(
        "HDBSCAN min_samples parameter (higher = stricter clustering)",
        default=getattr(args, 'min_samples', 2),
        min_val=1
    )
    
    params['cluster_selection_epsilon'] = get_float_input(
        "Distance threshold for cluster merging",
        default=getattr(args, 'cluster_selection_epsilon', 0.0),
        min_val=0.0
    )
    
    params['alpha'] = get_float_input(
        "HDBSCAN alpha parameter for point weighting",
        default=getattr(args, 'alpha', 1.0),
        min_val=0.0
    )
    
    params['cluster_selection_method'] = get_string_input(
        "Algorithm for cluster extraction",
        default=getattr(args, 'cluster_selection_method', 'eom'),
        options=['eom', 'leaf']
    )
    
    params['n_jobs'] = get_int_input(
        "Number of CPU cores to use (-1 for all)",
        default=getattr(args, 'n_jobs', -1)
    )
    
    params['test_mode'] = get_yes_no_input(
        "Run in test mode with a subset of data?",
        default=getattr(args, 'test', False)
    )
    
    if params['test_mode']:
        params['sample_size'] = get_int_input(
            "Number of samples to use in test mode",
            default=getattr(args, 'sample_size', 1000),
            min_val=100
        )
    else:
        params['sample_size'] = getattr(args, 'sample_size', 1000)
    
    return params

def get_reranking_params(args) -> Dict[str, Any]:
    """Get reranking parameters interactively."""
    print("\n=== Reranking Configuration ===")
    params = {}
    
    params['use_reranking'] = get_yes_no_input(
        "Use cross-encoder reranking?",
        default=getattr(args, 'rerank', False)
    )
    
    if params['use_reranking']:
        params['cross_encoder_model'] = get_string_input(
            "Cross-encoder model to use",
            default=getattr(args, 'cross_encoder', 'cross-encoder/stsb-roberta-base')
        )
        
        params['cross_encoder_batch_size'] = get_int_input(
            "Batch size for cross-encoder inference",
            default=getattr(args, 'cross_encoder_batch_size', 32),
            min_val=1
        )
        
        params['similarity_threshold'] = get_float_input(
            "Similarity threshold (higher = stricter matching)",
            default=getattr(args, 'similarity_threshold', 0.6),
            min_val=0.0,
            max_val=1.0
        )
        
        params['rerank_weight'] = get_float_input(
            "Weight between embeddings and cross-encoder (0=only embeddings, 1=only cross-encoder)",
            default=getattr(args, 'rerank_weight', 0.5),
            min_val=0.0,
            max_val=1.0
        )
    else:
        # Set defaults for non-interactive mode
        params['cross_encoder_model'] = getattr(args, 'cross_encoder', 'cross-encoder/stsb-roberta-base')
        params['cross_encoder_batch_size'] = getattr(args, 'cross_encoder_batch_size', 32)
        params['similarity_threshold'] = getattr(args, 'similarity_threshold', 0.6)
        params['rerank_weight'] = getattr(args, 'rerank_weight', 0.5)
    
    return params

def get_category_params(args) -> Dict[str, Any]:
    """Get category handling parameters interactively."""
    print("\n=== Category Handling Configuration ===")
    params = {}
    
    use_categories = not getattr(args, 'no_categories', False)
    params['use_categories'] = get_yes_no_input(
        "Use category-based clustering?",
        default=use_categories
    )
    
    if params['use_categories']:
        params['category_exclusivity'] = get_float_input(
            "Category exclusivity (0=mix freely, 1=strict separation)",
            default=getattr(args, 'category_exclusivity', 1.0),
            min_val=0.0,
            max_val=1.0
        )
    else:
        params['category_exclusivity'] = getattr(args, 'category_exclusivity', 1.0)
    
    return params

def get_analysis_params(args) -> Dict[str, Any]:
    """Get analysis parameters interactively."""
    print("\n=== Analysis Configuration ===")
    params = {}
    
    params['refined'] = get_yes_no_input(
        "Analyze refined clusters?",
        default=getattr(args, 'refined', True)
    )
    
    params['run_basic_analysis'] = get_yes_no_input(
        "Run basic cluster statistics analysis?",
        default=getattr(args, 'analyze_basic', True)
    )
    
    params['run_margin_analysis'] = get_yes_no_input(
        "Analyze price/margin variations within clusters?",
        default=getattr(args, 'analyze_margins', False)
    )
    
    params['run_usda_analysis'] = get_yes_no_input(
        "Analyze USDA mapping alignment?",
        default=getattr(args, 'analyze_usda', False)
    )
    
    params['run_llm_analysis'] = get_yes_no_input(
        "Use LLM for analyzing cluster coherence?",
        default=getattr(args, 'analyze_llm', False)
    )
    
    if params['run_llm_analysis']:
        params['llm_model'] = get_string_input(
            "LLM model to use for analysis",
            default=getattr(args, 'llm_model', 'gpt-3.5-turbo')
        )
    else:
        params['llm_model'] = getattr(args, 'llm_model', 'gpt-3.5-turbo')
    
    params['cluster_size_threshold'] = get_int_input(
        "Minimum cluster size for detailed analysis",
        default=getattr(args, 'cluster_size_threshold', 5),
        min_val=2
    )
    
    if params['run_margin_analysis']:
        params['price_variation_threshold'] = get_float_input(
            "Threshold for identifying significant price variations",
            default=getattr(args, 'price_variation_threshold', 0.2),
            min_val=0.0,
            max_val=1.0
        )
    else:
        params['price_variation_threshold'] = getattr(args, 'price_variation_threshold', 0.2)
    
    params['detailed_output'] = get_yes_no_input(
        "Generate detailed analysis output?",
        default=getattr(args, 'detailed_output', False)
    )
    
    return params

def save_config(config: Dict[str, Any], filepath: str) -> bool:
    """Save configuration to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {filepath}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}
