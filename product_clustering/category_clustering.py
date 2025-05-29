"""
Category-based hierarchical clustering module.

This module implements a hierarchical clustering approach that:
1. Groups products by their category
2. Applies HDBSCAN clustering within each category group
3. Optionally applies cross-encoder re-ranking to refine clusters
"""
import os
import sys
import numpy as np
import pandas as pd
import hdbscan
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
# Import configuration
from src import config

from data_prep.category_filter import filter_products_by_category, normalize_category_names, group_products_by_category
from product_clustering.embed_products import embed_products

def create_category_embeddings(
    category_groups: Dict[str, pd.DataFrame],
    text_col: str = 'clustering_description',
    embedding_type: str = 'sentence-transformer',
    model_name: str = 'all-mpnet-base-v2',
    batch_size: int = 100
) -> Dict[str, Dict[str, Any]]:
    """
    Generate embeddings for products within each category using a shared embedder.
    
    Args:
        category_groups: Dictionary mapping category names to DataFrames of products
        text_col: Column name containing the text to embed
        embedding_type: Type of embeddings to use
        model_name: Name of specific model to use
        batch_size: Batch size for embedding generation
        
    Returns:
        Dictionary mapping categories to their embedding data
    """
    from src.VectorDB.localEmbedder import LocalEmbedder
    from tqdm.auto import tqdm
    import numpy as np
    
    # Create model cache directory if it doesn't exist
    cache_dir = os.path.join(parent_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    category_embeddings = {}
    total_categories = len(category_groups)
    
    print(f"Generating embeddings for {total_categories} categories using {model_name}...")
    
    # Create a single shared embedder instead of creating one per category
    # This prevents hitting the Hugging Face rate limits
    if embedding_type == 'sentence-transformer':
        # Initialize embedder once with cache_dir
        shared_embedder = LocalEmbedder(model_name=model_name, cache_dir=cache_dir)
        print(f"Initialized shared embedder with model: {model_name} (cached in {cache_dir})")
    else:
        print(f"Using embedding type: {embedding_type}")
        shared_embedder = None
    
    for i, (category, group_df) in enumerate(tqdm(category_groups.items(), total=total_categories)):
        print(f"\nProcessing category {i+1}/{total_categories}: {category} ({len(group_df)} products)")
        
        if len(group_df) == 0:
            print(f"  No products in category '{category}', skipping")
            continue
        
        # Keep track of valid products (with non-empty descriptions)
        valid_indices = []
        valid_texts = []
        product_codes = []
        
        # Extract text and product codes, ensure all values are strings
        for idx, row in group_df.iterrows():
            if pd.isna(row[text_col]) or str(row[text_col]).strip() == '':
                continue
                
            valid_indices.append(idx)
            valid_texts.append(str(row[text_col]))
            product_codes.append(str(row['product_code']))
        
        print(f"After filtering, using {len(valid_texts)} valid products for embedding")
        
        if len(valid_texts) == 0:
            print(f"  No valid products in category '{category}', skipping")
            continue
            
        # Generate embeddings for this category's products
        try:
            print(f"Generating embeddings for {len(valid_texts)} products...")
            
            if embedding_type == 'sentence-transformer':
                # Use the shared embedder
                print(f"Using sentence-transformer embeddings with model: {model_name}")
                
                # Process in batches
                all_embeddings = []
                for i in range(0, len(valid_texts), batch_size):
                    batch_texts = valid_texts[i:i + batch_size]
                    batch_embeddings = shared_embedder(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    
                embeddings = np.array(all_embeddings)
            else:
                # For other embedding types, use the existing function
                temp_df = pd.DataFrame({
                    'product_code': product_codes,
                    text_col: valid_texts
                })
                
                from product_clustering.embed_products import embed_products
                embeddings, product_codes = embed_products(
                    df=temp_df,
                    text_col=text_col,
                    embedding_type=embedding_type,
                    model_name=model_name,
                    batch_size=batch_size
                )
            
            # Store the embeddings and product codes for this category
            category_embeddings[category] = {
                'embeddings': embeddings,
                'product_codes': product_codes,
                'product_count': len(product_codes)
            }
            
            print(f"Generated embeddings with shape: {embeddings.shape}")
            print(f"  Successfully embedded {len(product_codes)} products for category '{category}'")
        except Exception as e:
            print(f"  Error embedding category '{category}': {e}")
    
    print(f"Generated embeddings for {len(category_embeddings)}/{total_categories} categories")
    return category_embeddings

def cluster_category_products(
    category_embeddings: Dict[str, Dict[str, Any]],
    metric: str = 'euclidean',
    min_cluster_size: int = 3,
    min_samples: int = 2
) -> Dict[str, Dict[str, List[str]]]:
    """
    Perform HDBSCAN clustering within each category.
    
    Args:
        category_embeddings: Dictionary mapping categories to their embedding data
        metric: Distance metric to use
        min_cluster_size: Minimum size of clusters
        min_samples: HDBSCAN min_samples parameter
        
    Returns:
        Dictionary mapping categories to their clusters
    """
    category_clusters = {}
    total_categories = len(category_embeddings)
    total_clustered_products = 0
    total_products = 0
    
    print(f"Clustering products within {total_categories} categories...")
    print(f"Using parameters: metric={metric}, min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    for i, (category, data) in enumerate(tqdm(category_embeddings.items(), total=total_categories)):
        embeddings = data['embeddings']
        product_codes = data['product_codes']
        
        if len(product_codes) < min_cluster_size:
            print(f"  Category '{category}' has fewer products ({len(product_codes)}) than min_cluster_size ({min_cluster_size}), skipping clustering")
            # Store as a single cluster
            category_clusters[category] = {
                'clusters': {'small_category': product_codes},
                'noise': [],
                'stats': {
                    'total_products': len(product_codes),
                    'clustered_products': len(product_codes),
                    'noise_products': 0,
                    'cluster_count': 1
                }
            }
            total_clustered_products += len(product_codes)
            total_products += len(product_codes)
            continue
        
        print(f"\nClustering category {i+1}/{total_categories}: {category} ({len(product_codes)} products)")
        
        # Create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            gen_min_span_tree=True,
            cluster_selection_method='eom'  # Excess of Mass for better stability
        )
        
        # Perform clustering
        try:
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Organize products by cluster
            clusters = defaultdict(list)
            noise_points = []
            
            for i, (code, label) in enumerate(zip(product_codes, cluster_labels)):
                if label >= 0:
                    # Valid cluster
                    cluster_id = f"{category}_{label}"
                    clusters[cluster_id].append(code)
                else:
                    # Noise point
                    noise_points.append(code)
            
            # Calculate statistics
            clustered_count = sum(len(cluster) for cluster in clusters.values())
            noise_count = len(noise_points)
            
            category_clusters[category] = {
                'clusters': dict(clusters),
                'noise': noise_points,
                'stats': {
                    'total_products': len(product_codes),
                    'clustered_products': clustered_count,
                    'noise_products': noise_count,
                    'cluster_count': len(clusters)
                }
            }
            
            total_clustered_products += clustered_count
            total_products += len(product_codes)
            
            # Print results for this category
            clustering_rate = (clustered_count / len(product_codes)) * 100 if len(product_codes) > 0 else 0
            print(f"  Found {len(clusters)} clusters containing {clustered_count} products ({clustering_rate:.1f}%)")
            print(f"  {noise_count} products were classified as noise")
            
            # Show cluster sizes
            if clusters:
                cluster_sizes = [len(products) for products in clusters.values()]
                avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
                print(f"  Average cluster size: {avg_size:.1f} products")
                print(f"  Cluster size range: {min(cluster_sizes) if cluster_sizes else 0} - {max(cluster_sizes) if cluster_sizes else 0} products")
        
        except Exception as e:
            print(f"  Error clustering category '{category}': {e}")
    
    # Print overall statistics
    overall_clustering_rate = (total_clustered_products / total_products) * 100 if total_products > 0 else 0
    print(f"\nOverall clustering results:")
    print(f"  Total products processed: {total_products}")
    print(f"  Products in clusters: {total_clustered_products} ({overall_clustering_rate:.1f}%)")
    print(f"  Products as noise: {total_products - total_clustered_products} ({100 - overall_clustering_rate:.1f}%)")
    
    return category_clusters

def refine_category_clusters(
    category_clusters: Dict[str, Dict[str, Any]],
    products_df: pd.DataFrame,
    cross_encoder_model: str = "cross-encoder/stsb-roberta-base",
    similarity_threshold: float = 0.6,
    batch_size: int = 32,
    silence_tqdm: bool = False,
    min_cluster_size: int = 2,
    max_pairs_per_cluster: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Refine clusters within each category using CrossEncoder reranking.
    
    Args:
        category_clusters: Dictionary mapping categories to their clusters
        products_df: DataFrame containing product information
        cross_encoder_model: CrossEncoder model name
        similarity_threshold: Minimum similarity threshold
        batch_size: Batch size for CrossEncoder predictions
        silence_tqdm: Whether to hide progress bars
        min_cluster_size: Minimum size of clusters (default: 2)
        max_pairs_per_cluster: Maximum number of pairs to evaluate per cluster (default: 1000)
        
    Returns:
        Dictionary mapping categories to their refined clusters
    """
    # Import CrossEncoder here to avoid loading it unless needed
    try:
        from src.VectorDB.CrossEncoder import CrossEncoder
    except ImportError:
        print("Error: Could not import CrossEncoder. Make sure the module is available.")
        return category_clusters

    # The products DataFrame is already passed in, no need to load it
    # Just a sanity check to make sure we have the right columns
    if 'product_code' not in products_df.columns or 'product_description' not in products_df.columns:
        print("Error: products_df must contain 'product_code' and 'product_description' columns")
        return category_clusters
    
    # Create a lookup dictionary for quick access to product descriptions
    product_lookup = {}
    for _, row in products_df.iterrows():
        if 'product_code' in row and 'product_description' in row:
            product_lookup[str(row['product_code'])] = row['product_description']
    
    # Initialize CrossEncoder
    print(f"Initializing CrossEncoder with model: {cross_encoder_model}")
    cross_encoder = CrossEncoder(model_name=cross_encoder_model)
    
    refined_category_clusters = {}
    total_categories = len(category_clusters)
    total_original_clusters = 0
    total_refined_clusters = 0
    
    print(f"Refining clusters across {total_categories} categories...")
    
    for i, (category, data) in enumerate(tqdm(category_clusters.items(), total=total_categories)):
        clusters = data.get('clusters', {})
        total_original_clusters += len(clusters)
        
        print(f"\nRefining clusters for category {i+1}/{total_categories}: {category} ({len(clusters)} clusters)")
        
        refined_clusters = {}
        skipped_clusters = 0
        
        for cluster_id, product_codes in tqdm(clusters.items(), desc=f"Refining {category} clusters", leave=False):
            # Skip tiny clusters that are below threshold
            if len(product_codes) < 3:
                refined_clusters[cluster_id] = product_codes
                skipped_clusters += 1
                continue
            
            # Get product descriptions for all products in this cluster
            cluster_products = []
            for code in product_codes:
                if code in product_lookup:
                    cluster_products.append({
                        'code': code,
                        'description': product_lookup[code]
                    })
            
            if len(cluster_products) < 2:
                # Not enough products with descriptions, keep as is
                refined_clusters[cluster_id] = product_codes
                continue
            
            # Generate all pairs for scoring, limiting to max_pairs_per_cluster
            pairs = []
            center_idx = len(cluster_products) // 2  # Use middle product as reference
            center_product = cluster_products[center_idx]
            
            for i, product in enumerate(cluster_products):
                if i != center_idx:
                    pairs.append((center_product['description'], product['description']))
            
            # Limit number of pairs if needed
            if len(pairs) > max_pairs_per_cluster:
                np.random.seed(42)  # For reproducibility
                pairs = np.random.choice(pairs, max_pairs_per_cluster, replace=False).tolist()
            
            # Score pairs with CrossEncoder
            if pairs:
                # Use the model's predict method instead of directly on the CrossEncoder instance
                similarity_scores = cross_encoder.model.predict(pairs, batch_size=batch_size)
                
                # Create two lists: high confidence products and lower confidence products
                high_confidence_products = [center_product['code']]  # Always keep center product as high confidence
                lower_confidence_products = []
                
                pair_idx = 0
                for i, product in enumerate(cluster_products):
                    if i != center_idx:
                        # Check if product meets similarity threshold
                        if pair_idx < len(similarity_scores) and similarity_scores[pair_idx] >= similarity_threshold:
                            high_confidence_products.append(product['code'])
                        else:
                            # Keep product but mark as lower confidence
                            lower_confidence_products.append(product['code'])
                        pair_idx += 1
                
                # Combine all products, but store high confidence ones first
                all_products = high_confidence_products + lower_confidence_products
                
                # Store clusters - we keep all products but separate by confidence level
                if len(high_confidence_products) >= min_cluster_size:
                    # Store high confidence cluster
                    refined_clusters[cluster_id] = high_confidence_products
                    # Store lower confidence products as a separate sub-cluster if there are any
                    if lower_confidence_products:
                        refined_clusters[f"{cluster_id}_supplemental"] = lower_confidence_products
                elif len(all_products) >= min_cluster_size:
                    # If high confidence doesn't meet min size but combined does, store all together
                    refined_clusters[cluster_id] = all_products
                elif len(all_products) > 0:
                    # Store as a small cluster if it has any products
                    refined_clusters[f"{cluster_id}_small"] = all_products
            else:
                # No pairs to evaluate, keep original cluster
                refined_clusters[cluster_id] = product_codes
        
        # Store refined clusters for this category
        refined_category_clusters[category] = {
            'clusters': refined_clusters,
            'noise': data.get('noise', []),
            'stats': {
                'total_products': data.get('stats', {}).get('total_products', 0),
                'original_clusters': len(clusters),
                'refined_clusters': len(refined_clusters),
                'skipped_clusters': skipped_clusters
            }
        }
        
        total_refined_clusters += len(refined_clusters)
        
        # Print results for this category
        print(f"  Original clusters: {len(clusters)}")
        print(f"  Refined clusters: {len(refined_clusters)}")
        print(f"  Skipped clusters: {skipped_clusters}")
    
    # Print overall statistics
    print(f"\nOverall refinement results:")
    print(f"  Total original clusters: {total_original_clusters}")
    print(f"  Total refined clusters: {total_refined_clusters}")
    
    return refined_category_clusters

def save_category_clusters(
    category_clusters: Dict[str, Dict[str, Any]],
    output_dir: str,
    filename: str = 'category_clusters.json'
) -> str:
    """
    Save category clusters to a JSON file.
    
    Args:
        category_clusters: Dictionary mapping categories to their clusters
        output_dir: Directory to save the clusters
        filename: Name of the output file
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten clusters for saving
    flattened_clusters = {}
    
    for category, data in category_clusters.items():
        clusters = data.get('clusters', {})
        
        # Add all clusters with their hierarchical IDs
        for cluster_id, product_codes in clusters.items():
            flattened_clusters[cluster_id] = product_codes
    
    # Also save category statistics
    category_stats = {
        category: data.get('stats', {})
        for category, data in category_clusters.items()
    }
    
    # Save flattened clusters
    clusters_path = os.path.join(output_dir, filename)
    with open(clusters_path, 'w') as f:
        json.dump(flattened_clusters, f, indent=2)
    
    # Save category statistics
    stats_path = os.path.join(output_dir, 'category_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(category_stats, f, indent=2)
    
    # Save category-to-product mapping for analysis
    category_products = {}
    for category, data in category_clusters.items():
        all_products = []
        for products in data.get('clusters', {}).values():
            all_products.extend(products)
        
        # Add noise products
        noise_products = data.get('noise', [])
        
        category_products[category] = {
            'clustered': all_products,
            'noise': noise_products,
            'total': len(all_products) + len(noise_products)
        }
    
    category_products_path = os.path.join(output_dir, 'category_products.json')
    with open(category_products_path, 'w') as f:
        json.dump(category_products, f, indent=2)
    
    print(f"Saved {len(flattened_clusters)} clusters to {clusters_path}")
    print(f"Saved category statistics to {stats_path}")
    print(f"Saved category-to-product mapping to {category_products_path}")
    
    return clusters_path

def run_category_clustering(
    prepared_data_path: str,
    output_dir: str,
    embedding_type: str = 'sentence-transformer',
    model_name: str = None,  # Will use config value if None
    metric: str = None,      # Will use config value if None
    min_cluster_size: int = None,  # Will use config value if None
    min_samples: int = None,       # Will use config value if None
    use_reranking: bool = None,    # Will use config value if None
    cross_encoder_model: str = None,  # Will use config value if None
    similarity_threshold: float = None  # Will use config value if None
) -> str:
    """
    Run the complete category-based clustering pipeline.
    
    Args:
        prepared_data_path: Path to the prepared data CSV
        output_dir: Directory to save results
        embedding_type: Type of embeddings to use
        model_name: Name of the embedding model
        metric: Distance metric for clustering
        min_cluster_size: Minimum size of clusters
        min_samples: HDBSCAN min_samples parameter
        use_reranking: Whether to use CrossEncoder reranking
        cross_encoder_model: Model for reranking
        similarity_threshold: Threshold for reranking
        
    Returns:
        Path to the saved clusters file
    """
    start_time = time.time()
    
    # Use default values from config if parameters are None
    if model_name is None:
        model_name = config.SENTENCE_TRANSFORMER_MODEL
    if metric is None:
        metric = config.CLUSTERING_METRIC
    if min_cluster_size is None:
        min_cluster_size = config.MIN_CLUSTER_SIZE
    if min_samples is None:
        min_samples = config.MIN_SAMPLES
    if use_reranking is None:
        use_reranking = config.USE_RERANKING
    if cross_encoder_model is None:
        cross_encoder_model = config.CROSS_ENCODER_MODEL
    if similarity_threshold is None:
        similarity_threshold = config.SIMILARITY_THRESHOLD
    
    print(f"Starting category-based clustering pipeline")
    print(f"Using model: {model_name}")
    print(f"Using parameters: metric={metric}, min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    print(f"Re-ranking: {use_reranking}, threshold: {similarity_threshold}")
    print(f"Loading prepared data from {prepared_data_path}")
    
    # Load prepared data
    prepared_df = pd.read_csv(prepared_data_path)
    print(f"Loaded {len(prepared_df)} products")
    
    # 1. Filter and group products by category
    filtered_df = filter_products_by_category(prepared_df)
    normalized_df = normalize_category_names(filtered_df)
    category_groups = group_products_by_category(normalized_df)
    
    # 2. Generate embeddings for each category
    category_embeddings = create_category_embeddings(
        category_groups,
        text_col='clustering_description',
        embedding_type=embedding_type,
        model_name=model_name
    )
    
    # 3. Cluster products within each category
    category_clusters = cluster_category_products(
        category_embeddings,
        metric=metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    
    # 4. Apply reranking if requested
    if use_reranking:
        print("\nRefining clusters with CrossEncoder reranking...")
        refined_category_clusters = refine_category_clusters(
            category_clusters,
            products_df=prepared_df,
            cross_encoder_model=cross_encoder_model,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size
        )
        
        # Save refined clusters
        refined_output_dir = os.path.join(output_dir, 'refined')
        clusters_path = save_category_clusters(
            refined_category_clusters,
            output_dir=refined_output_dir,
            filename='refined_category_clusters.json'
        )
    else:
        # Save original clusters
        clusters_path = save_category_clusters(
            category_clusters,
            output_dir=output_dir,
            filename='category_clusters.json'
        )
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    minutes, seconds = divmod(execution_time, 60)
    print(f"Category-based clustering completed in {int(minutes)}m {seconds:.1f}s")
    
    return clusters_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run category-based hierarchical clustering")
    parser.add_argument("--prepared_data", type=str, help="Path to prepared data CSV")
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2", help="Embedding model name")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric for clustering")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Minimum size of clusters")
    parser.add_argument("--min_samples", type=int, default=2, help="HDBSCAN min_samples parameter")
    parser.add_argument("--rerank", action="store_true", help="Use CrossEncoder reranking")
    parser.add_argument("--cross_encoder", type=str, default="cross-encoder/stsb-roberta-base", help="CrossEncoder model")
    parser.add_argument("--similarity", type=float, default=0.6, help="Similarity threshold for reranking")
    
    args = parser.parse_args()
    
    # Use default paths if not provided
    if args.prepared_data is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.prepared_data = os.path.join(script_dir, "data", "prepared_products.csv")
    
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, "data", "category_clustering")
    
    # Run the clustering pipeline
    run_category_clustering(
        prepared_data_path=args.prepared_data,
        output_dir=args.output_dir,
        model_name=args.model,
        metric=args.metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        use_reranking=args.rerank,
        cross_encoder_model=args.cross_encoder,
        similarity_threshold=args.similarity
    )
