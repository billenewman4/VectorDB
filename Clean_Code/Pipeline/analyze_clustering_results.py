"""
Analysis script for hierarchical clustering results.

This script provides tools to analyze and visualize clustering results from
the hierarchical clustering pipeline, including:
- Cluster coherence analysis
- Inter-level relationship visualization
- Cluster size distribution analysis
- Representative product identification per cluster
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import argparse

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Now path is set up correctly for relative imports

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_clustering_results(results_csv: str) -> pd.DataFrame:
    """
    Load clustering results from CSV file.
    
    Args:
        results_csv: Path to clustering results CSV
        
    Returns:
        DataFrame with clustering results
    """
    logger.info(f"Loading clustering results from {results_csv}")
    return pd.read_csv(results_csv)


def analyze_cluster_sizes(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """
    Analyze cluster size distribution per level.
    
    Args:
        df: DataFrame with clustering results
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with cluster size statistics
    """
    logger.info("Analyzing cluster size distribution")
    
    # Find all level columns
    level_columns = [col for col in df.columns if col.startswith('level_') and col.endswith('_cluster')]
    
    stats = {}
    
    # Create figure for all level distributions
    plt.figure(figsize=(12, 8))
    
    for i, level_col in enumerate(level_columns):
        level_num = level_col.split('_')[1]
        
        # Count cluster sizes
        cluster_sizes = df[~df[level_col].isna()][level_col].value_counts().sort_values(ascending=False)
        
        # Skip if no clusters
        if len(cluster_sizes) == 0:
            logger.warning(f"No clusters found for {level_col}")
            continue
        
        # Calculate statistics
        stats[level_num] = {
            "num_clusters": len(cluster_sizes),
            "largest_cluster": cluster_sizes.iloc[0],
            "smallest_cluster": cluster_sizes.iloc[-1],
            "median_size": cluster_sizes.median(),
            "mean_size": cluster_sizes.mean(),
            "total_points": cluster_sizes.sum(),
            "noise_points": len(df[df[level_col].isna()]),
            "noise_percentage": 100 * len(df[df[level_col].isna()]) / len(df)
        }
        
        # Plot distribution
        plt.subplot(len(level_columns), 1, i+1)
        sns.histplot(cluster_sizes, kde=True)
        plt.title(f"Level {level_num} Cluster Size Distribution")
        plt.xlabel("Cluster Size")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        # Annotate with statistics
        text = (f"Clusters: {stats[level_num]['num_clusters']}, "
                f"Median: {stats[level_num]['median_size']:.1f}, "
                f"Noise: {stats[level_num]['noise_percentage']:.1f}%")
        plt.annotate(text, xy=(0.5, 0.9), xycoords='axes fraction', 
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "cluster_size_distribution.png"))
    
    # Save statistics to CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.index.name = 'level'
    stats_df.to_csv(os.path.join(output_dir, "cluster_size_statistics.csv"))
    
    logger.info(f"Cluster size analysis saved to {output_dir}")
    
    return stats


def analyze_inter_level_relationships(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """
    Analyze relationships between clusters at different levels.
    
    Args:
        df: DataFrame with clustering results
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with inter-level relationship statistics
    """
    logger.info("Analyzing inter-level cluster relationships")
    
    # Find all level columns
    level_columns = [col for col in df.columns if col.startswith('level_') and col.endswith('_cluster')]
    level_columns.sort()  # Ensure correct order
    
    if len(level_columns) < 2:
        logger.warning("Need at least 2 levels to analyze inter-level relationships")
        return {}
    
    stats = {}
    
    # Analyze relationships between adjacent levels
    for i in range(len(level_columns) - 1):
        upper_level = level_columns[i]
        lower_level = level_columns[i+1]
        
        upper_num = upper_level.split('_')[1]
        lower_num = lower_level.split('_')[1]
        
        # Create a mapping from upper level clusters to lower level clusters
        cluster_mapping = defaultdict(list)
        
        for _, row in df.dropna(subset=[upper_level]).iterrows():
            if pd.notna(row[upper_level]):
                upper_cluster = row[upper_level]
                lower_cluster = row[lower_level] if pd.notna(row[lower_level]) else "noise"
                
                if lower_cluster not in cluster_mapping[upper_cluster]:
                    cluster_mapping[upper_cluster].append(lower_cluster)
        
        # Calculate statistics
        child_counts = [len(children) for children in cluster_mapping.values()]
        
        if not child_counts:
            logger.warning(f"No valid mappings found between {upper_level} and {lower_level}")
            continue
        
        stats[f"{upper_num}_to_{lower_num}"] = {
            "upper_clusters": len(cluster_mapping),
            "avg_children_per_cluster": np.mean(child_counts),
            "max_children": max(child_counts),
            "min_children": min(child_counts),
            "total_connections": sum(child_counts)
        }
    
    # Create a Sankey diagram for level relationships
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Prepare data for Sankey diagram
        all_nodes = []
        links_source = []
        links_target = []
        links_value = []
        
        node_index = {}
        current_index = 0
        
        # Process each level pair
        for i in range(len(level_columns) - 1):
            upper_level = level_columns[i]
            lower_level = level_columns[i+1]
            
            # Count pairs
            pair_counts = Counter()
            for _, row in df.dropna(subset=[upper_level]).iterrows():
                if pd.notna(row[upper_level]) and pd.notna(row[lower_level]):
                    pair_counts[(f"L{upper_level.split('_')[1]}-{row[upper_level]}", 
                               f"L{lower_level.split('_')[1]}-{row[lower_level]}")] += 1
            
            # Add nodes and links
            for (source, target), value in pair_counts.items():
                if source not in node_index:
                    node_index[source] = current_index
                    all_nodes.append(source)
                    current_index += 1
                
                if target not in node_index:
                    node_index[target] = current_index
                    all_nodes.append(target)
                    current_index += 1
                
                links_source.append(node_index[source])
                links_target.append(node_index[target])
                links_value.append(value)
        
        # Create Sankey diagram if we have data
        if links_source:
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes
                ),
                link=dict(
                    source=links_source,
                    target=links_target,
                    value=links_value
                )
            )])
            
            fig.update_layout(
                title_text="Hierarchical Cluster Relationships",
                font_size=10,
                height=800
            )
            
            # Save as HTML
            fig.write_html(os.path.join(output_dir, "cluster_relationships.html"))
            logger.info(f"Sankey diagram saved to {os.path.join(output_dir, 'cluster_relationships.html')}")
    except ImportError:
        logger.warning("Plotly not installed. Skipping Sankey diagram.")
    
    # Save statistics to CSV
    if stats:
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        stats_df.index.name = 'level_pair'
        stats_df.to_csv(os.path.join(output_dir, "inter_level_statistics.csv"))
    
    logger.info(f"Inter-level relationship analysis saved to {output_dir}")
    
    return stats


def identify_representative_products(df: pd.DataFrame, output_dir: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Identify representative products for each cluster at each level.
    
    Args:
        df: DataFrame with clustering results
        output_dir: Directory to save results
        top_n: Number of representative products to identify per cluster
        
    Returns:
        Dictionary with representative products per cluster
    """
    logger.info(f"Identifying top {top_n} representative products per cluster")
    
    # Find all level columns
    level_columns = [col for col in df.columns if col.startswith('level_') and col.endswith('_cluster')]
    
    # Dictionary to store representative products
    representatives = {}
    
    # Process each level
    for level_col in level_columns:
        level_num = level_col.split('_')[1]
        representatives[f"level_{level_num}"] = {}
        
        # Get clusters
        clusters = df[~df[level_col].isna()][level_col].unique()
        
        # For each cluster, find representative products
        for cluster in clusters:
            # Get products in this cluster
            cluster_products = df[df[level_col] == cluster]
            
            # Sort by common attributes if available (like frequency, price, etc.)
            # Here we're using a simple approach - can be enhanced with other metrics
            if 'frequency' in cluster_products.columns:
                # Sort by frequency descending
                cluster_products = cluster_products.sort_values('frequency', ascending=False)
            elif 'description' in cluster_products.columns:
                # Sort by description length (as a proxy for informativeness)
                cluster_products['desc_len'] = cluster_products['description'].str.len()
                cluster_products = cluster_products.sort_values('desc_len', ascending=False)
            
            # Select top N representative products
            top_products = cluster_products.head(top_n)
            
            # Store in dictionary
            representatives[f"level_{level_num}"][f"cluster_{cluster}"] = []
            
            for _, product in top_products.iterrows():
                product_info = {}
                
                # Include all available product information
                for col in product.index:
                    if col not in level_columns and not col.endswith('_path'):
                        product_info[col] = product[col]
                
                representatives[f"level_{level_num}"][f"cluster_{cluster}"].append(product_info)
    
    # Save representatives to JSON
    with open(os.path.join(output_dir, "representative_products.json"), 'w') as f:
        json.dump(representatives, f, indent=2)
    
    # Create a readable HTML report
    html_content = []
    html_content.append("<html><head>")
    html_content.append("<style>")
    html_content.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html_content.append("h1 { color: #2c3e50; }")
    html_content.append("h2 { color: #3498db; margin-top: 30px; }")
    html_content.append("h3 { color: #e74c3c; margin-top: 20px; }")
    html_content.append("table { border-collapse: collapse; width: 100%; margin-top: 10px; }")
    html_content.append("th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }")
    html_content.append("th { background-color: #f2f2f2; }")
    html_content.append("tr:nth-child(even) { background-color: #f9f9f9; }")
    html_content.append("</style>")
    html_content.append("</head><body>")
    html_content.append("<h1>Representative Products by Cluster</h1>")
    
    for level, clusters in representatives.items():
        html_content.append(f"<h2>{level.replace('_', ' ').title()}</h2>")
        
        for cluster, products in clusters.items():
            html_content.append(f"<h3>{cluster.replace('_', ' ').title()}</h3>")
            
            # Create table of products
            html_content.append("<table>")
            
            # Get all possible columns from products
            all_columns = set()
            for product in products:
                all_columns.update(product.keys())
            
            # Prioritize certain columns to appear first
            priority_columns = ['description', 'code', 'product_name', 'category']
            column_order = []
            
            # First add priority columns that exist
            for col in priority_columns:
                if col in all_columns:
                    column_order.append(col)
                    all_columns.remove(col)
            
            # Add remaining columns
            column_order.extend(sorted(all_columns))
            
            # Create header row
            html_content.append("<tr>")
            for col in column_order:
                html_content.append(f"<th>{col}</th>")
            html_content.append("</tr>")
            
            # Add products
            for product in products:
                html_content.append("<tr>")
                for col in column_order:
                    value = product.get(col, "")
                    html_content.append(f"<td>{value}</td>")
                html_content.append("</tr>")
            
            html_content.append("</table>")
    
    html_content.append("</body></html>")
    
    # Save HTML report
    with open(os.path.join(output_dir, "representative_products.html"), 'w') as f:
        f.write("\n".join(html_content))
    
    logger.info(f"Representative products saved to {output_dir}")
    
    return representatives


def analyze_cluster_coherence(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """
    Analyze cluster coherence based on product attributes.
    
    Args:
        df: DataFrame with clustering results
        output_dir: Directory to save results
        
    Returns:
        Dictionary with coherence statistics
    """
    logger.info("Analyzing cluster coherence")
    
    # Find all level columns
    level_columns = [col for col in df.columns if col.startswith('level_') and col.endswith('_cluster')]
    
    # Identify categorical columns for coherence analysis
    possible_categorical = ['category', 'product_type', 'brand', 'department']
    categorical_cols = [col for col in possible_categorical if col in df.columns]
    
    if not categorical_cols:
        logger.warning("No categorical columns found for coherence analysis")
        return {}
    
    coherence_stats = {}
    
    # Analyze each level
    for level_col in level_columns:
        level_num = level_col.split('_')[1]
        level_stats = {}
        
        # Get clusters
        clusters = df[~df[level_col].isna()][level_col].unique()
        
        # Calculate coherence for each cluster and categorical attribute
        for cat_col in categorical_cols:
            cluster_coherence = []
            
            for cluster in clusters:
                cluster_products = df[df[level_col] == cluster]
                
                # Skip if empty
                if len(cluster_products) == 0:
                    continue
                
                # Count category values
                value_counts = cluster_products[cat_col].value_counts(normalize=True)
                
                # Calculate entropy as a measure of coherence
                # Lower entropy = higher coherence
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                
                # Calculate dominance of top category
                top_category_share = value_counts.iloc[0] if len(value_counts) > 0 else 0
                
                cluster_coherence.append({
                    'cluster': cluster,
                    'size': len(cluster_products),
                    'entropy': entropy,
                    'top_category': value_counts.index[0] if len(value_counts) > 0 else "N/A",
                    'top_category_share': top_category_share,
                    'unique_categories': len(value_counts)
                })
            
            # Compile statistics
            if cluster_coherence:
                coherence_df = pd.DataFrame(cluster_coherence)
                
                level_stats[cat_col] = {
                    'avg_entropy': coherence_df['entropy'].mean(),
                    'min_entropy': coherence_df['entropy'].min(),
                    'max_entropy': coherence_df['entropy'].max(),
                    'avg_top_category_share': coherence_df['top_category_share'].mean(),
                    'avg_unique_categories': coherence_df['unique_categories'].mean()
                }
                
                # Plot coherence distribution
                plt.figure(figsize=(10, 6))
                
                # Create scatter plot of entropy vs cluster size
                plt.scatter(coherence_df['size'], coherence_df['entropy'], 
                           alpha=0.7, s=coherence_df['top_category_share']*100)
                
                plt.title(f"Level {level_num} Cluster Coherence ({cat_col})")
                plt.xlabel("Cluster Size")
                plt.ylabel("Entropy (lower = more coherent)")
                plt.grid(True, alpha=0.3)
                
                # Add some labels for interesting points
                top_coherent = coherence_df.nsmallest(3, 'entropy')
                for _, row in top_coherent.iterrows():
                    plt.annotate(f"C{row['cluster']}\n{row['top_category']}",
                               xy=(row['size'], row['entropy']),
                               xytext=(10, 0), textcoords='offset points',
                               ha='left', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                plt.savefig(os.path.join(output_dir, f"coherence_level{level_num}_{cat_col}.png"))
                plt.close()
                
                # Save detailed coherence data
                coherence_df.to_csv(os.path.join(output_dir, f"coherence_level{level_num}_{cat_col}.csv"), index=False)
        
        coherence_stats[f"level_{level_num}"] = level_stats
    
    # Save overall coherence statistics
    with open(os.path.join(output_dir, "coherence_statistics.json"), 'w') as f:
        json.dump(coherence_stats, f, indent=2)
    
    logger.info(f"Coherence analysis saved to {output_dir}")
    
    return coherence_stats


def main():
    """
    Main entry point for analyzing clustering results.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze Hierarchical Clustering Results")
    parser.add_argument("--results_csv", type=str, required=True,
                       help="Path to clustering results CSV")
    parser.add_argument("--output_dir", type=str, default="clustering_analysis",
                       help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load clustering results
    df = load_clustering_results(args.results_csv)
    
    # Run analyses
    analyze_cluster_sizes(df, args.output_dir)
    analyze_inter_level_relationships(df, args.output_dir)
    identify_representative_products(df, args.output_dir)
    analyze_cluster_coherence(df, args.output_dir)
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")
    logger.info("Run with '--help' for more options")


if __name__ == "__main__":
    main()
