#!/usr/bin/env python3
"""
Comprehensive Cluster Analysis

This script provides detailed analysis of product clustering results, including:
1. Cluster statistics (counts, sizes, distributions)
2. Clustering coverage (percentage of products in clusters)
3. Cluster coherence evaluation (semantic similarity within clusters)
4. Sample clusters for manual inspection
5. Visualization of cluster quality metrics
6. Identification of mixed product clusters
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import Counter, defaultdict
import random
import re

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_transaction_data(data_dir: Optional[str] = None):
    """
    Load original transaction data to get product descriptions.
    
    Args:
        data_dir: Optional directory containing data files
        
    Returns:
        Dictionary mapping product codes to descriptions
    """
    try:
        import pandas as pd
        import os
        
        # Find product data file
        if data_dir is None:
            # Check multiple possible locations with priority for CSV
            potential_paths = [
                # First check for prepared_products.csv (highest priority)
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared_products.csv"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prepared_products.csv"),
                
                # Then check for traditional transaction data formats
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Actuals", "Transaction_Report_Actual.xlsx"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Transaction_Report_Actual.xlsx"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Transaction_Report_Actual.xlsx")
            ]
            
            data_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    data_path = path
                    print(f"Found data file: {path}")
                    break
            
            if data_path is None:
                print("Error: Product or transaction data file not found in expected locations")
                
                # Try to find any product data files
                print("Searching for alternative product data files...")
                product_files = []
                for root, _, files in os.walk(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
                    for file in files:
                        if file.endswith(".csv") and ("product" in file.lower() or "prepared" in file.lower()):
                            product_files.append(os.path.join(root, file))
                        elif file.endswith(".xlsx") and "transaction" in file.lower():
                            product_files.append(os.path.join(root, file))
                
                if product_files:
                    print(f"Found potential product files: {product_files}")
                    data_path = product_files[0]
                    print(f"Using: {data_path}")
                else:
                    return {}
        else:
            # Look for prepared_products.csv first
            data_path = os.path.join(data_dir, "prepared_products.csv")
            if not os.path.exists(data_path):
                # Fall back to transaction data
                data_path = os.path.join(data_dir, "Transaction_Report_Actual.xlsx")
                if not os.path.exists(data_path):
                    print(f"Error: Neither prepared_products.csv nor Transaction_Report_Actual.xlsx found in {data_dir}")
                    
                    # Try to find any product data file
                    for root, _, files in os.walk(data_dir):
                        for file in files:
                            if "product" in file.lower() and (file.endswith(".csv") or file.endswith(".xlsx")):
                                data_path = os.path.join(root, file)
                                print(f"Found alternative product data: {data_path}")
                                break
                    
                    if not os.path.exists(data_path):
                        return {}
        
        print(f"Loading product data from: {data_path}")
        
        # Try to load based on file extension
        df = None
        if data_path.lower().endswith(".csv"):
            try:
                df = pd.read_csv(data_path)
            except Exception as e:
                print(f"Error loading CSV file: {str(e)}")
                return {}
        else:  # Excel format
            try:
                df = pd.read_excel(data_path, sheet_name="Sheet1")
            except Exception as e1:
                try:
                    df = pd.read_excel(data_path)
                except Exception as e2:
                    print(f"Error loading Excel file: {str(e1)}; {str(e2)}")
                    return {}
        
        # Create a product code to description mapping
        product_dict = {}
        
        # Handle different column name formats
        code_col = next((col for col in df.columns if col.lower() in ['product_code', 'productcode', 'code']), None)
        desc_col = next((col for col in df.columns if col.lower() in ['product_description', 'productdescription', 'description', 'desc']), None)
        
        if code_col and desc_col:
            print(f"Using columns: {code_col} (code) and {desc_col} (description)")
            for _, row in df.iterrows():
                if pd.notna(row[code_col]) and pd.notna(row[desc_col]):
                    product_dict[str(row[code_col])] = str(row[desc_col])
            
            print(f"Loaded {len(product_dict)} product descriptions from product data")
            return product_dict
        else:
            print(f"Required columns not found. Available columns: {list(df.columns)}")
            
            # If we can't find the exact column names, just use the first two columns as a fallback
            if len(df.columns) >= 2:
                print(f"Using first two columns as fallback: {df.columns[0]} and {df.columns[1]}")
                for _, row in df.iterrows():
                    if pd.notna(row[df.columns[0]]) and pd.notna(row[df.columns[1]]):
                        product_dict[str(row[df.columns[0]])] = str(row[df.columns[1]])
                
                print(f"Loaded {len(product_dict)} product descriptions using fallback method")
                return product_dict
            
            return {}
    except Exception as e:
        print(f"Error loading product data: {str(e)}")
        return {}

def load_embeddings(data_dir: str):
    """
    Load product embeddings for coherence analysis.
    
    Args:
        data_dir: Directory containing embeddings
        
    Returns:
        Dictionary mapping product codes to embeddings
    """
    embeddings_path = os.path.join(data_dir, "product_embeddings.npy")
    codes_path = os.path.join(data_dir, "product_codes.txt")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(codes_path):
        print(f"Embeddings or product codes file not found")
        return {}
    
    try:
        embeddings = np.load(embeddings_path)
        with open(codes_path, 'r') as f:
            product_codes = [line.strip() for line in f.readlines()]
        
        if len(embeddings) != len(product_codes):
            print(f"Warning: Mismatch between embeddings ({len(embeddings)}) and product codes ({len(product_codes)})")
        
        # Create mapping from product code to embedding
        embedding_dict = {}
        for i, code in enumerate(product_codes):
            if i < len(embeddings):
                embedding_dict[code] = embeddings[i]
        
        print(f"Loaded {len(embedding_dict)} product embeddings")
        return embedding_dict
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return {}

def load_clusters(clusters_path: str):
    """
    Load cluster data from JSON file.
    
    Args:
        clusters_path: Path to clusters JSON file
        
    Returns:
        Dictionary mapping cluster IDs to lists of product codes
    """
    if not os.path.exists(clusters_path):
        print(f"Clusters file not found at {clusters_path}")
        return {}
    
    try:
        with open(clusters_path, 'r') as f:
            clusters = json.load(f)
        
        print(f"Loaded {len(clusters)} clusters")
        return clusters
    except Exception as e:
        print(f"Error loading clusters: {str(e)}")
        return {}

def calculate_cluster_stats(clusters: Dict[str, List[str]], product_dict: Dict[str, str]):
    """
    Calculate basic statistics about clusters.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        product_dict: Dictionary mapping product codes to descriptions
        
    Returns:
        Dictionary with statistics
    """
    if not clusters:
        return {}
    
    # Calculate cluster sizes
    cluster_sizes = [len(products) for products in clusters.values()]
    
    # Count total unique products
    all_products = set()
    for products in clusters.values():
        all_products.update(products)
    
    # Calculate total products in dataset
    total_products_in_dataset = len(product_dict) if product_dict else 0
    
    # Calculate size distribution
    size_bins = [(1, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    size_distribution = {}
    
    for low, high in size_bins:
        range_str = f"{low}-{high if high != float('inf') else '∞'}"
        count = sum(1 for s in cluster_sizes if low <= s <= high)
        size_distribution[range_str] = {
            'count': count,
            'percentage': count / len(clusters) * 100
        }
    
    # Calculate statistics
    stats = {
        'total_clusters': len(clusters),
        'total_products': len(all_products),
        'total_products_in_dataset': total_products_in_dataset,
        'clustering_coverage_pct': (len(all_products) / total_products_in_dataset * 100) if total_products_in_dataset > 0 else 0,
        'products_per_cluster': {
            'mean': np.mean(cluster_sizes),
            'median': np.median(cluster_sizes),
            'min': min(cluster_sizes),
            'max': max(cluster_sizes),
            'std': np.std(cluster_sizes)
        },
        'size_distribution': size_distribution
    }
    
    return stats

def analyze_cluster_coherence(clusters: Dict[str, List[str]], 
                             embedding_dict: Dict[str, np.ndarray],
                             sample_clusters: bool = True,
                             max_clusters: int = 100):
    """
    Analyze the coherence of clusters using embeddings.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        embedding_dict: Dictionary mapping product codes to embeddings
        sample_clusters: Whether to sample clusters for analysis
        max_clusters: Maximum number of clusters to analyze
        
    Returns:
        Dictionary mapping cluster IDs to coherence scores
    """
    if not clusters or not embedding_dict:
        return {}
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Select clusters to analyze
    cluster_ids = list(clusters.keys())
    if sample_clusters and len(cluster_ids) > max_clusters:
        # Stratified sampling by cluster size
        size_bins = [(1, 5), (6, 10), (11, 20), (21, 50), (51, float('inf'))]
        sampled_ids = []
        
        for low, high in size_bins:
            bin_ids = [cid for cid in cluster_ids 
                      if low <= len(clusters[cid]) <= (high if high != float('inf') else float('inf'))]
            
            # Sample proportionally to bin size
            bin_sample_size = min(len(bin_ids), int(max_clusters * len(bin_ids) / len(cluster_ids)))
            if bin_ids and bin_sample_size > 0:
                sampled_ids.extend(random.sample(bin_ids, bin_sample_size))
        
        cluster_ids = sampled_ids
    
    # Calculate coherence for each selected cluster
    coherence_scores = {}
    missing_products = 0
    total_products = 0
    
    for cluster_id in cluster_ids:
        product_codes = clusters[cluster_id]
        if len(product_codes) <= 1:
            coherence_scores[cluster_id] = 0.0
            continue
        
        # Get embeddings for products in this cluster
        cluster_embeddings = []
        total_products += len(product_codes)
        
        for code in product_codes:
            if code in embedding_dict:
                cluster_embeddings.append(embedding_dict[code])
            else:
                missing_products += 1
        
        if len(cluster_embeddings) <= 1:
            coherence_scores[cluster_id] = 0.0
            continue
        
        # Calculate pairwise similarities
        cluster_embeddings = np.array(cluster_embeddings)
        sim_matrix = cosine_similarity(cluster_embeddings)
        
        # Remove self-similarity
        np.fill_diagonal(sim_matrix, 0)
        
        # Average pairwise similarity as coherence score
        avg_similarity = sim_matrix.sum() / (len(cluster_embeddings) * (len(cluster_embeddings) - 1))
        coherence_scores[cluster_id] = float(avg_similarity)
    
    if missing_products > 0:
        print(f"Warning: {missing_products} out of {total_products} products ({missing_products/total_products:.1%}) were not found in the embeddings")
    
    return coherence_scores

def identify_mixed_clusters(clusters: Dict[str, List[str]], 
                          product_dict: Dict[str, str],
                          coherence_scores: Dict[str, float]):
    """
    Identify potentially mixed clusters based on product descriptions.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        product_dict: Dictionary mapping product codes to descriptions
        coherence_scores: Dictionary mapping cluster IDs to coherence scores
        
    Returns:
        List of potentially mixed clusters
    """
    if not clusters or not product_dict:
        return []
    
    # Analyze descriptions to find common terms
    mixed_clusters = []
    
    for cluster_id, product_codes in clusters.items():
        if len(product_codes) < 5:  # Skip very small clusters
            continue
        
        # Get descriptions for this cluster
        descriptions = []
        for code in product_codes:
            if code in product_dict:
                descriptions.append(product_dict[code])
        
        if len(descriptions) < 5:  # Not enough descriptions to analyze
            continue
        
        # Extract key terms (nouns) from descriptions
        import re
        terms = []
        for desc in descriptions:
            # Simple heuristic: extract words that might be product types
            words = re.findall(r'\b[A-Z]+\b', desc.upper())
            terms.extend(words)
        
        # Count term frequencies
        term_counts = Counter(terms)
        
        # If there are multiple common terms with similar frequencies,
        # this might be a mixed cluster
        common_terms = [term for term, count in term_counts.most_common(5) 
                       if count >= len(descriptions) * 0.2]
        
        # Check if coherence score is low
        coherence = coherence_scores.get(cluster_id, 1.0)
        
        if len(common_terms) >= 3 and coherence < 0.7:
            mixed_clusters.append({
                'cluster_id': cluster_id,
                'size': len(product_codes),
                'coherence': coherence,
                'common_terms': common_terms,
                'sample_products': [(code, product_dict.get(code, "Unknown")) 
                                  for code in random.sample(product_codes, min(5, len(product_codes)))]
            })
    
    return mixed_clusters

def sample_clusters(clusters: Dict[str, List[str]], 
                   product_dict: Dict[str, str],
                   coherence_scores: Dict[str, float],
                   num_clusters: int = 10):
    """
    Sample clusters for manual inspection.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        product_dict: Dictionary mapping product codes to descriptions
        coherence_scores: Dictionary mapping cluster IDs to coherence scores
        num_clusters: Number of clusters to sample
        
    Returns:
        Dictionary with sampled clusters
    """
    if not clusters:
        return {}
    
    # Sample clusters of various sizes and coherence levels
    size_categories = {
        'small': [cid for cid, products in clusters.items() if len(products) <= 5],
        'medium': [cid for cid, products in clusters.items() if 6 <= len(products) <= 15],
        'large': [cid for cid, products in clusters.items() if len(products) > 15]
    }
    
    coherence_categories = {
        'high': [cid for cid, score in coherence_scores.items() if score >= 0.8],
        'medium': [cid for cid, score in coherence_scores.items() if 0.6 <= score < 0.8],
        'low': [cid for cid, score in coherence_scores.items() if score < 0.6]
    }
    
    # Sample from each category
    sampled_clusters = {}
    
    # Sample by size
    for category, cluster_ids in size_categories.items():
        if cluster_ids:
            sample_size = min(num_clusters // 3, len(cluster_ids))
            for cluster_id in random.sample(cluster_ids, sample_size):
                sampled_clusters[f"{category}_size_{cluster_id}"] = {
                    'cluster_id': cluster_id,
                    'size': len(clusters[cluster_id]),
                    'coherence': coherence_scores.get(cluster_id, 0.0),
                    'category': f"{category} size",
                    'products': [(code, product_dict.get(code, "Unknown")) 
                               for code in clusters[cluster_id]]
                }
    
    # Sample by coherence
    for category, cluster_ids in coherence_categories.items():
        if cluster_ids:
            sample_size = min(num_clusters // 3, len(cluster_ids))
            for cluster_id in random.sample(cluster_ids, sample_size):
                if cluster_id not in [c.split('_')[-1] for c in sampled_clusters.keys()]:
                    sampled_clusters[f"{category}_coherence_{cluster_id}"] = {
                        'cluster_id': cluster_id,
                        'size': len(clusters[cluster_id]),
                        'coherence': coherence_scores.get(cluster_id, 0.0),
                        'category': f"{category} coherence",
                        'products': [(code, product_dict.get(code, "Unknown")) 
                                   for code in clusters[cluster_id]]
                    }
    
    return sampled_clusters

def generate_report(output_path: str,
                   stats: Dict[str, Any],
                   coherence_scores: Dict[str, float],
                   mixed_clusters: List[Dict[str, Any]],
                   sampled_clusters: Dict[str, Dict[str, Any]],
                   product_dict: Dict[str, str]):
    """
    Generate a comprehensive analysis report.
    
    Args:
        output_path: Path to save the report
        stats: Cluster statistics
        coherence_scores: Dictionary mapping cluster IDs to coherence scores
        mixed_clusters: List of potentially mixed clusters
        sampled_clusters: Dictionary with sampled clusters
        product_dict: Dictionary mapping product codes to descriptions
        
    Returns:
        Path to the generated report
    """
    with open(output_path, 'w') as f:
        # Title and summary
        f.write("# Product Clustering Analysis Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total clusters: {stats['total_clusters']}\n")
        
        # Add clustering coverage information
        if stats.get('total_products_in_dataset', 0) > 0:
            f.write(f"- Total products in clusters: {stats['total_products']} out of {stats['total_products_in_dataset']} total products ({stats['clustering_coverage_pct']:.1f}%)\n")
        else:
            f.write(f"- Total products in clusters: {stats['total_products']}\n")
            
        f.write(f"- Average cluster size: {stats['products_per_cluster']['mean']:.2f}\n")
        f.write(f"- Median cluster size: {stats['products_per_cluster']['median']:.1f}\n")
        f.write(f"- Cluster size range: {stats['products_per_cluster']['min']} to {stats['products_per_cluster']['max']}\n")
        
        if coherence_scores:
            coherence_values = list(coherence_scores.values())
            f.write(f"- Average coherence score: {np.mean(coherence_values):.3f}\n")
            f.write(f"- Median coherence score: {np.median(coherence_values):.3f}\n")
        
        # Cluster size distribution
        f.write("\n## Cluster Size Distribution\n\n")
        f.write("| Size Range | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        
        for range_str, data in stats['size_distribution'].items():
            f.write(f"| {range_str} | {data['count']} | {data['percentage']:.1f}% |\n")
        
        # Coherence distribution
        if coherence_scores:
            f.write("\n## Coherence Distribution\n\n")
            coherence_ranges = [
                (0.0, 0.4, "Low"),
                (0.4, 0.6, "Moderate"),
                (0.6, 0.8, "Good"),
                (0.8, 1.0, "Excellent")
            ]
            
            f.write("| Coherence Range | Count | Percentage | Description |\n")
            f.write("|-----------------|-------|------------|-------------|\n")
            
            for low, high, desc in coherence_ranges:
                count = sum(1 for score in coherence_scores.values() if low <= score < high)
                percentage = count / len(coherence_scores) * 100
                f.write(f"| {low:.1f} - {high:.1f} | {count} | {percentage:.1f}% | {desc} |\n")
        
        # Mixed clusters
        if mixed_clusters:
            f.write("\n## Potentially Mixed Clusters\n\n")
            f.write(f"Found {len(mixed_clusters)} potentially mixed clusters.\n\n")
            
            for i, cluster in enumerate(mixed_clusters[:10]):  # Show top 10
                f.write(f"### Mixed Cluster {i+1}: {cluster['cluster_id']}\n\n")
                f.write(f"- Size: {cluster['size']} products\n")
                f.write(f"- Coherence: {cluster['coherence']:.3f}\n")
                f.write(f"- Common terms: {', '.join(cluster['common_terms'])}\n")
                f.write("\nSample products:\n\n")
                
                for code, desc in cluster['sample_products']:
                    f.write(f"- {code}: {desc}\n")
                
                f.write("\n")
        
        # Sample clusters
        if sampled_clusters:
            f.write("\n## Sample Clusters\n\n")
            
            # Group by category
            by_category = defaultdict(list)
            for info in sampled_clusters.values():
                by_category[info['category']].append(info)
            
            for category, clusters in by_category.items():
                f.write(f"### {category.title()} Clusters\n\n")
                
                for i, cluster in enumerate(clusters):
                    f.write(f"#### Cluster {cluster['cluster_id']}\n\n")
                    f.write(f"- Size: {cluster['size']} products\n")
                    f.write(f"- Coherence: {cluster['coherence']:.3f}\n")
                    f.write("\nProducts:\n\n")
                    
                    # Show all products
                    for code, desc in cluster['products']:
                        f.write(f"- {code}: {desc}\n")
                    
                    f.write("\n")
    
    print(f"Report generated at {output_path}")
    return output_path

def generate_visualizations(output_dir: str,
                          stats: Dict[str, Any],
                          coherence_scores: Dict[str, float]):
    """
    Generate visualizations of cluster analysis.
    
    Args:
        output_dir: Directory to save visualizations
        stats: Cluster statistics
        coherence_scores: Dictionary mapping cluster IDs to coherence scores
        
    Returns:
        List of generated visualization paths
    """
    os.makedirs(output_dir, exist_ok=True)
    viz_paths = []
    
    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Cluster size distribution
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    size_ranges = list(stats['size_distribution'].keys())
    counts = [data['count'] for data in stats['size_distribution'].values()]
    
    # Create bar chart
    plt.bar(size_ranges, counts, color='skyblue', alpha=0.7)
    plt.title('Distribution of Cluster Sizes', fontsize=15)
    plt.xlabel('Cluster Size Range', fontsize=12)
    plt.ylabel('Number of Clusters', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    size_dist_path = os.path.join(output_dir, 'cluster_size_distribution.png')
    plt.savefig(size_dist_path)
    plt.close()
    viz_paths.append(size_dist_path)
    
    # 2. Coherence score distribution
    if coherence_scores:
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(list(coherence_scores.values()), bins=20, alpha=0.7, color='green')
        plt.title('Distribution of Cluster Coherence Scores', fontsize=15)
        plt.xlabel('Coherence Score', fontsize=12)
        plt.ylabel('Number of Clusters', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for thresholds
        plt.axvline(x=0.6, color='orange', linestyle='--', label='Good (0.6)')
        plt.axvline(x=0.8, color='red', linestyle='--', label='Excellent (0.8)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        coherence_dist_path = os.path.join(output_dir, 'coherence_distribution.png')
        plt.savefig(coherence_dist_path)
        plt.close()
        viz_paths.append(coherence_dist_path)
        
        # 3. Scatterplot of cluster size vs. coherence
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        sizes = []
        scores = []
        
        for cluster_id, score in coherence_scores.items():
            if cluster_id in stats['clusters']:
                size = len(stats['clusters'][cluster_id])
                sizes.append(size)
                scores.append(score)
        
        # Create scatterplot
        plt.scatter(sizes, scores, alpha=0.5, s=30)
        plt.title('Cluster Size vs. Coherence', fontsize=15)
        plt.xlabel('Cluster Size (Number of Products)', fontsize=12)
        plt.ylabel('Coherence Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line for threshold
        plt.axhline(y=0.6, color='orange', linestyle='--', label='Good Coherence (0.6)')
        plt.legend()
        
        # Set log scale for x-axis if there's a wide range
        if max(sizes) / min(sizes) > 10:
            plt.xscale('log')
            plt.xlabel('Cluster Size (Log Scale)', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        scatter_path = os.path.join(output_dir, 'size_vs_coherence.png')
        plt.savefig(scatter_path)
        plt.close()
        viz_paths.append(scatter_path)
    
    return viz_paths

def analyze_clusters(clusters_path: str, 
                    data_dir: Optional[str] = None, 
                    refined: bool = True,
                    output_dir: Optional[str] = None,
                    transaction_data_path: Optional[str] = None):
    """
    Run comprehensive cluster analysis.
    
    Args:
        clusters_path: Path to clusters JSON file
        data_dir: Directory containing data files
        refined: Whether analyzing refined clusters
        output_dir: Directory to save analysis results
        
    Returns:
        Path to analysis report
    """
    # Set up directories
    if data_dir is None:
        data_dir = os.path.dirname(os.path.dirname(clusters_path))
    
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    print("Loading data...")
    clusters = load_clusters(clusters_path)
    if not clusters:
        print("Error: No clusters found")
        return None
    
    # If transaction_data_path is provided, use it directly
    if transaction_data_path and os.path.exists(transaction_data_path):
        print(f"Loading transaction data from provided path: {transaction_data_path}")
        try:
            import pandas as pd
            product_dict = {}
            df = pd.read_excel(transaction_data_path)
            
            # Handle different column name formats
            code_col = next((col for col in df.columns if col.lower() in ['product_code', 'productcode', 'code']), None)
            desc_col = next((col for col in df.columns if col.lower() in ['description', 'product_description', 'productdescription', 'desc']), None)
            
            if code_col and desc_col:
                print(f"Using columns: {code_col} (code) and {desc_col} (description)")
                for _, row in df.iterrows():
                    if pd.notna(row[code_col]) and pd.notna(row[desc_col]):
                        product_dict[str(row[code_col])] = str(row[desc_col])
                
                print(f"Loaded {len(product_dict)} product descriptions from transaction data")
            else:
                print(f"Required columns not found in {transaction_data_path}. Available columns: {list(df.columns)}")
                # Try fallback to first two columns
                if len(df.columns) >= 2:
                    print(f"Using first two columns as fallback: {df.columns[0]} and {df.columns[1]}")
                    for _, row in df.iterrows():
                        if pd.notna(row[df.columns[0]]) and pd.notna(row[df.columns[1]]):
                            product_dict[str(row[df.columns[0]])] = str(row[df.columns[1]])
                    print(f"Loaded {len(product_dict)} product descriptions using fallback method")
        except Exception as e:
            print(f"Error loading provided transaction data: {str(e)}")
            product_dict = load_transaction_data(data_dir)
    else:
        product_dict = load_transaction_data(data_dir)
    
    embedding_dict = load_embeddings(data_dir)
    
    # 2. Calculate basic statistics
    print("Calculating statistics...")
    stats = calculate_cluster_stats(clusters, product_dict)
    stats['clusters'] = clusters  # Add clusters to stats for later use
    
    # 3. Analyze coherence
    print("Analyzing cluster coherence...")
    coherence_scores = analyze_cluster_coherence(clusters, embedding_dict)
    
    # 4. Identify potentially mixed clusters
    print("Identifying potentially mixed clusters...")
    mixed_clusters = identify_mixed_clusters(clusters, product_dict, coherence_scores)
    
    # 5. Sample clusters for inspection
    print("Sampling clusters for inspection...")
    sampled_clusters = sample_clusters(clusters, product_dict, coherence_scores)
    
    # 6. Generate report
    print("Generating report...")
    report_filename = f"cluster_analysis{'_refined' if refined else ''}.md"
    report_path = os.path.join(output_dir, report_filename)
    generate_report(report_path, stats, coherence_scores, mixed_clusters, sampled_clusters, product_dict)
    
    # 7. Generate visualizations
    print("Generating visualizations...")
    viz_paths = generate_visualizations(output_dir, stats, coherence_scores)
    
    # 8. Perform USDA mapping analysis to evaluate how well expected groups are clustered
    print("Analyzing USDA mapping alignment...")
    try:
        from product_clustering.analyze_usda_mapping import load_usda_mapping, analyze_usda_grouping_alignment, generate_usda_mapping_report
        
        # Path to USDA mapping file
        usda_mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "data", "CorrectMapping", "product_mapping_semantic.xlsx")
        
        if os.path.exists(usda_mapping_path):
            # Load category to product mapping if available
            category_products = None
            category_products_path = os.path.join(os.path.dirname(clusters_path), "category_products.json")
            if os.path.exists(category_products_path):
                try:
                    with open(category_products_path, 'r') as f:
                        category_products = json.load(f)
                    print(f"Loaded category-product mapping from {category_products_path}")
                except Exception as e:
                    print(f"Error loading category-product mapping: {e}")
            
            # Load USDA mapping
            usda_mapping = load_usda_mapping(usda_mapping_path)
            print(f"Loaded {len(usda_mapping)} product groupings from USDA mapping file")
            
            # Load prepared products data for reference
            prepared_df_path = os.path.join(data_dir, "prepared_products.csv")
            if os.path.exists(prepared_df_path):
                prepared_df = pd.read_csv(prepared_df_path)
                
                # Extract flat clusters from the hierarchy
                flat_clusters = {}
                if isinstance(clusters, dict) and 'clusters' in next(iter(clusters.values()), {}):
                    # This is a category-based cluster format
                    for category, data in clusters.items():
                        for cluster_id, products in data.get('clusters', {}).items():
                            flat_clusters[f"{category}_{cluster_id}"] = products
                else:
                    # This is a flat cluster format
                    flat_clusters = clusters
                
                # Run the analysis
                usda_analysis = analyze_usda_grouping_alignment(
                    flat_clusters, usda_mapping, prepared_df, category_products)
                
                # Generate and save the report
                usda_report = generate_usda_mapping_report(usda_analysis)
                usda_report_path = os.path.join(output_dir, f"usda_mapping_analysis{'_refined' if refined else ''}.md")
                with open(usda_report_path, 'w') as f:
                    f.write(usda_report)
                
                print(f"USDA mapping analysis complete. Report saved to {usda_report_path}")
            else:
                print(f"Warning: Prepared products data not found at {prepared_df_path}. USDA mapping analysis skipped.")
        else:
            print(f"Warning: USDA mapping file not found at {usda_mapping_path}. USDA mapping analysis skipped.")
    except Exception as e:
        print(f"Warning: USDA mapping analysis failed: {e}")
    
    print(f"\nAnalysis complete!")
    print(f"- Report: {report_path}")
    print(f"- Visualizations: {', '.join(os.path.basename(p) for p in viz_paths)}")
    print(f"- USDA Mapping Analysis: {os.path.basename(usda_report_path) if 'usda_report_path' in locals() else 'Not available'}")
    
    return report_path

def main():
    """Main function to run cluster analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze product clustering results")
    parser.add_argument("--clusters_path", help="Path to clusters JSON file")
    parser.add_argument("--data_dir", help="Directory containing data files")
    parser.add_argument("--transaction_data", help="Path to transaction data Excel file")
    parser.add_argument("--refined", action="store_true", default=True, help="Analyze refined clusters (default)")
    parser.add_argument("--no-refined", action="store_false", dest="refined", help="Analyze original clusters")
    parser.add_argument("--output_dir", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Set default clusters path
    clusters_path = args.clusters_path
    if clusters_path is None:
        data_dir = args.data_dir
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        if args.refined:
            clusters_path = os.path.join(data_dir, "refined_clusters", "refined_clusters.json")
            if not os.path.exists(clusters_path):
                clusters_path = os.path.join(data_dir, "improved_clustering", "refined", "refined_clusters.json")
        else:
            clusters_path = os.path.join(data_dir, "improved_clustering", "clusters.json")
    
    # Run analysis
    analyze_clusters(
        clusters_path=clusters_path, 
        data_dir=args.data_dir,
        refined=args.refined,
        output_dir=args.output_dir,
        transaction_data_path=args.transaction_data
    )

if __name__ == "__main__":
    main()
