#!/usr/bin/env python3
"""
cluster_analyzer_llm.py - Use LLM to analyze product clusters and explain pricing inconsistencies

This script samples clusters from margin analysis results, presents the data to an LLM via LangChain,
and gets back detailed analysis explaining potential reasons for pricing anomalies within clusters.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Any
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

# Load environment variables from .env file
load_dotenv()



def load_margin_analysis_results(
    cluster_stats_path: str, 
    margin_outliers_path: str,
    clusters_path: str,
    pricing_data_path: str,
    product_id_col: str = 'ProductCode',
    price_col: str = 'SalesPrice',
    cost_col: str = 'AverageProductCost',
    description_col: str = 'ProductDescription'
) -> tuple:
    """
    Load the margin analysis results and supporting data.
    
    Args:
        cluster_stats_path: Path to cluster margin statistics CSV
        margin_outliers_path: Path to margin outliers CSV
        clusters_path: Path to clusters JSON file
        pricing_data_path: Path to pricing data file
        product_id_col: Name of product ID column in pricing data
        price_col: Name of price column in pricing data
        cost_col: Name of cost column in pricing data
        description_col: Name of description column in pricing data
        
    Returns:
        Tuple of (cluster_stats_df, outliers_df, clusters_dict, pricing_df)
    """
    # Load cluster stats
    cluster_stats_df = pd.read_csv(cluster_stats_path)
    
    # Load outliers
    outliers_df = pd.read_csv(margin_outliers_path)
    
    # Load clusters
    with open(clusters_path, 'r') as f:
        clusters_dict = json.load(f)
    
    # Load pricing data
    if pricing_data_path.lower().endswith('.csv'):
        pricing_df = pd.read_csv(pricing_data_path)
    else:
        # Assume Excel
        pricing_df = pd.read_excel(pricing_data_path)
    
    # Ensure product ID is string type
    pricing_df[product_id_col] = pricing_df[product_id_col].astype(str)
    
    return cluster_stats_df, outliers_df, clusters_dict, pricing_df


def sample_clusters_for_analysis(
    cluster_stats_df: pd.DataFrame,
    n_samples: int = 5,
    min_cluster_size: int = 3,
    prioritize_high_variance: bool = True
) -> List[str]:
    """
    Sample clusters for LLM analysis.
    
    Args:
        cluster_stats_df: DataFrame with cluster statistics
        n_samples: Number of clusters to sample
        min_cluster_size: Minimum cluster size to consider
        prioritize_high_variance: Whether to prioritize high variance clusters
        
    Returns:
        List of sampled cluster IDs
    """
    # Filter by minimum size
    eligible_clusters = cluster_stats_df[cluster_stats_df['product_count'] >= min_cluster_size]
    
    if eligible_clusters.empty:
        raise ValueError(f"No clusters with at least {min_cluster_size} products found.")
    
    if prioritize_high_variance:
        # Prioritize high variance clusters but ensure some diversity
        high_variance = eligible_clusters[eligible_clusters['high_variance']].copy()
        normal_variance = eligible_clusters[~eligible_clusters['high_variance']].copy()
        
        # If we have enough high variance clusters, take most from there
        if len(high_variance) >= n_samples * 0.8:
            # Sort by std_margin descending
            high_variance = high_variance.sort_values('std_margin', ascending=False)
            # Take 80% of samples from high variance clusters
            high_var_samples = min(int(n_samples * 0.8), len(high_variance))
            normal_var_samples = n_samples - high_var_samples
            
            # Get sample cluster IDs
            sampled_high_var = high_variance.head(high_var_samples)['cluster_id'].tolist()
            
            if normal_var_samples > 0 and not normal_variance.empty:
                sampled_normal_var = normal_variance.sample(
                    min(normal_var_samples, len(normal_variance))
                )['cluster_id'].tolist()
            else:
                sampled_normal_var = []
                
            sampled_clusters = sampled_high_var + sampled_normal_var
        else:
            # Not enough high variance clusters, take what we have and sample the rest randomly
            sampled_high_var = high_variance['cluster_id'].tolist()
            remaining_samples = n_samples - len(sampled_high_var)
            
            if remaining_samples > 0 and not normal_variance.empty:
                sampled_normal_var = normal_variance.sample(
                    min(remaining_samples, len(normal_variance))
                )['cluster_id'].tolist()
            else:
                sampled_normal_var = []
                
            sampled_clusters = sampled_high_var + sampled_normal_var
    else:
        # Random sampling
        sampled_clusters = eligible_clusters.sample(
            min(n_samples, len(eligible_clusters))
        )['cluster_id'].tolist()
    
    # If we still don't have enough samples, pad with random selection (with replacement)
    if len(sampled_clusters) < n_samples:
        additional_needed = n_samples - len(sampled_clusters)
        additional_samples = random.choices(
            eligible_clusters['cluster_id'].tolist(), 
            k=additional_needed
        )
        sampled_clusters.extend(additional_samples)
    
    return sampled_clusters


def prepare_cluster_data_for_llm(
    cluster_id: str,
    clusters_dict: Dict[str, List[str]],
    pricing_df: pd.DataFrame,
    product_id_col: str,
    price_col: str,
    cost_col: str,
    description_col: str
) -> Dict[str, Any]:
    """
    Prepare detailed data about a cluster for LLM analysis.
    
    Args:
        cluster_id: ID of the cluster to prepare
        clusters_dict: Dictionary mapping cluster IDs to product IDs
        pricing_df: DataFrame with pricing data
        product_id_col: Name of product ID column
        price_col: Name of price column
        cost_col: Name of cost column
        description_col: Name of description column
        
    Returns:
        Dictionary with detailed cluster data
    """
    # Get products in this cluster
    product_ids = clusters_dict.get(cluster_id, [])
    
    # Get pricing data for these products
    cluster_products = []
    
    for product_id in product_ids:
        # Find product in pricing data
        product_data = pricing_df[pricing_df[product_id_col] == str(product_id)]
        
        if not product_data.empty:
            # Get first matching row
            product_row = product_data.iloc[0]
            
            # Calculate margin
            price = product_row.get(price_col, 0)
            cost = product_row.get(cost_col, 0)
            
            if price > 0:
                margin = (price - cost) / price
            else:
                margin = float('nan')
            
            # Add to list
            cluster_products.append({
                'product_id': product_id,
                'description': product_row.get(description_col, 'Unknown'),
                'price': price,
                'cost': cost,
                'margin': margin
            })
    
    # Calculate cluster stats
    if cluster_products:
        margins = [p['margin'] for p in cluster_products if np.isfinite(p['margin'])]
        
        if margins:
            mean_margin = np.mean(margins)
            median_margin = np.median(margins)
            std_margin = np.std(margins)
            min_margin = np.min(margins)
            max_margin = np.max(margins)
        else:
            mean_margin = median_margin = std_margin = min_margin = max_margin = float('nan')
            
        # Find products with unusual margins
        unusual_products = []
        for product in cluster_products:
            if np.isfinite(product['margin']) and abs(product['margin'] - mean_margin) > std_margin:
                product['deviation'] = product['margin'] - mean_margin
                unusual_products.append(product)
    else:
        mean_margin = median_margin = std_margin = min_margin = max_margin = float('nan')
        unusual_products = []
    
    return {
        'cluster_id': cluster_id,
        'size': len(cluster_products),
        'products': cluster_products,
        'mean_margin': mean_margin,
        'median_margin': median_margin, 
        'std_margin': std_margin,
        'min_margin': min_margin,
        'max_margin': max_margin,
        'unusual_products': unusual_products
    }


def analyze_cluster_with_llm(
    cluster_data: Dict[str, Any],
    llm
) -> str:
    """
    Use LLM to analyze a cluster and explain pricing inconsistencies.
    
    Args:
        cluster_data: Dictionary with detailed cluster data
    Returns:
        Analysis results from the LLM
    """
    
    template = """You are a retail pricing analyst specializing in product pricing and margin optimization.
    
    # Product Cluster Analysis: Cluster {cluster_id}

    ## Cluster Statistics:
    - Number of Products: {size}
    - Mean Gross Margin: {mean_margin:.2f}%
    - Median Gross Margin: {median_margin:.2f}%
    - Min Gross Margin: {min_margin:.2f}%
    - Max Gross Margin: {max_margin:.2f}%
    - Standard Deviation: {std_margin:.2f}%

    ## Products in Cluster:
    {product_details}

    Analyze this cluster and address the following:
    1. Explain any inconsistencies in pricing or margins within this cluster.
    2. Identify potential reasons for these inconsistencies (e.g., size/weight variations, quality differences, etc.)
    3. Recommend whether these products should have more consistent pricing/margins or if the differences appear justified.
    4. Suggest if these descrepancies represent a real opportunity to improve pricing or if it is explained by other factors. Furthermore, describe weither the descrepancy is in the sell price or the procurment price.
    
    Provide a thorough and insightful analysis.
    """
    
    # Format product details as a table with emphasis on descriptions
    product_details = "| Description | Product ID | Price | Cost | Margin |\n"
    product_details += "|------------|------------|-------|------|--------|\n"
    for p in cluster_data['products']:
        margin_str = f"{p['margin']:.2%}" if np.isfinite(p['margin']) else "N/A"
        product_details += f"| {p['description']} | {p['product_id']} | ${p['price']:.2f} | ${p['cost']:.2f} | {margin_str} |\n"
    
    # Using the newer approach with | operator instead of LLMChain
    prompt = PromptTemplate(
        template=template, 
        input_variables=["cluster_id", "size", "mean_margin", "median_margin", 
                         "min_margin", "max_margin", "std_margin", "product_details"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # Run analysis with formatted data
    return chain.invoke({
        "cluster_id": cluster_data['cluster_id'],
        "size": cluster_data['size'],
        "mean_margin": cluster_data['mean_margin'],
        "median_margin": cluster_data['median_margin'],
        "min_margin": cluster_data['min_margin'],
        "max_margin": cluster_data['max_margin'],
        "std_margin": cluster_data['std_margin'],
        "product_details": product_details
    })


def main():
    parser = argparse.ArgumentParser(description='Analyze product clusters using LLM')
    parser.add_argument('--cluster_stats', type=str, default='product_clustering/data/margin_analysis/cluster_margin_stats.csv',
                       help='Path to cluster margin statistics CSV')
    parser.add_argument('--outliers', type=str, default='product_clustering/data/margin_analysis/margin_outliers.csv',
                       help='Path to margin outliers CSV')
    parser.add_argument('--clusters', type=str, default='product_clustering/data/refined_clustering/refined_clusters.json',
                       help='Path to clusters JSON file')
    parser.add_argument('--pricing', type=str, default='/Users/billnewman/Desktop/GitHub/VectorDB/data/Actuals/Transaction_Report_Actual.xlsx',
                       help='Path to pricing data file')
    parser.add_argument('--output_dir', type=str, default='product_clustering/data/llm_analysis',
                       help='Directory to save LLM analysis results')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of clusters to sample for analysis')
    parser.add_argument('--min_cluster_size', type=int, default=3,
                       help='Minimum cluster size to consider')
    parser.add_argument('--product_id_col', type=str, default='ProductCode',
                       help='Name of product ID column in pricing data')
    parser.add_argument('--price_col', type=str, default='SalesPrice',
                       help='Name of price column in pricing data')
    parser.add_argument('--cost_col', type=str, default='AverageProductCost',
                       help='Name of cost column in pricing data')
    parser.add_argument('--description_col', type=str, default='ProductDescription',
                       help='Name of description column in pricing data')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='LLM model to use for analysis')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (can also set via OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading margin analysis results and supporting data...")
    cluster_stats_df, outliers_df, clusters_dict, pricing_df = load_margin_analysis_results(
        args.cluster_stats,
        args.outliers,
        args.clusters,
        args.pricing,
        args.product_id_col,
        args.price_col,
        args.cost_col,
        args.description_col
    )
    
    # Sample clusters for analysis
    print(f"Sampling {args.n_samples} clusters for analysis...")
    sampled_clusters = sample_clusters_for_analysis(
        cluster_stats_df,
        n_samples=args.n_samples,
        min_cluster_size=args.min_cluster_size
    )
    
    # Initialize LLM
    print(f"Initializing LLM ({args.model})...")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided either via --api_key argument or OPENAI_API_KEY environment variable")
    
    llm = ChatOpenAI(model_name=args.model, temperature=0, openai_api_key=api_key)
    
    # Analyze each sampled cluster
    analyses = []
    for i, cluster_id in enumerate(sampled_clusters):
        print(f"Analyzing cluster {i+1}/{len(sampled_clusters)}: {cluster_id}...")
        
        # Prepare cluster data
        cluster_data = prepare_cluster_data_for_llm(
            cluster_id,
            clusters_dict,
            pricing_df,
            args.product_id_col,
            args.price_col,
            args.cost_col,
            args.description_col
        )
        
        # Skip clusters with insufficient data
        if cluster_data['size'] < args.min_cluster_size:
            print(f"  Skipping cluster {cluster_id} - insufficient products with pricing data ({cluster_data['size']} < {args.min_cluster_size})")
            continue
            
        # Analyze with LLM
        analysis = analyze_cluster_with_llm(cluster_data, llm)
        
        # Store results
        result = {
            'cluster_id': cluster_id,
            'cluster_data': cluster_data,
            'analysis': analysis
        }
        analyses.append(result)
        
        # Save individual analysis
        with open(os.path.join(args.output_dir, f"cluster_{cluster_id}_analysis.json"), 'w') as f:
            # Convert to JSON-serializable format
            serializable_result = {
                'cluster_id': result['cluster_id'],
                'cluster_data': {
                    k: (float(v) if isinstance(v, np.float64) else v)
                    for k, v in result['cluster_data'].items()
                    if k != 'products' and k != 'unusual_products'
                },
                'products': [
                    {k: (float(v) if isinstance(v, np.float64) else v) for k, v in p.items()}
                    for p in result['cluster_data']['products']
                ],
                'analysis': result['analysis']
            }
            json.dump(serializable_result, f, indent=2)
            
        # Also save as markdown for easier reading
        with open(os.path.join(args.output_dir, f"cluster_{cluster_id}_analysis.md"), 'w') as f:
            f.write(f"# Analysis of Cluster {cluster_id}\n\n")
            f.write("## Cluster Statistics\n\n")
            f.write(f"- Number of products: {cluster_data['size']}\n")
            f.write(f"- Average margin: {cluster_data['mean_margin']:.2%}\n")
            f.write(f"- Median margin: {cluster_data['median_margin']:.2%}\n")
            f.write(f"- Margin standard deviation: {cluster_data['std_margin']:.2%}\n")
            f.write(f"- Margin range: {cluster_data['min_margin']:.2%} to {cluster_data['max_margin']:.2%}\n\n")
            
            f.write("## Products in Cluster\n\n")
            f.write("| Product ID | Description | Price | Cost | Margin |\n")
            f.write("|------------|-------------|-------|------|--------|\n")
            for p in cluster_data['products']:
                margin_str = f"{p['margin']:.2%}" if np.isfinite(p['margin']) else "N/A"
                f.write(f"| {p['product_id']} | {p['description']} | ${p['price']:.2f} | ${p['cost']:.2f} | {margin_str} |\n")
            
            f.write("\n## LLM Analysis\n\n")
            f.write(analysis)
    
    # Generate summary report
    print(f"Generating summary report...")
    summary = f"# Product Cluster Pricing Analysis Summary\n\nAnalysis of {len(analyses)} product clusters using {args.model}\n\n"
    
    for i, result in enumerate(analyses, 1):
        cluster_id = result['cluster_id']
        size = result['cluster_data']['size']
        mean_margin = result['cluster_data']['mean_margin'] * 100
        std_margin = result['cluster_data']['std_margin'] * 100
        
        # Get a sample of product descriptions from this cluster
        sample_products = result['cluster_data']['products'][:3]  # Take up to 3 products as examples
        product_examples = [p['description'] for p in sample_products]
        product_examples_text = ", ".join(product_examples)
        if len(result['cluster_data']['products']) > 3:
            product_examples_text += ", ..."
            
        # Extract first few sentences of analysis as summary
        analysis_text = result['analysis']
        summary_text = ". ".join(analysis_text.split(". ")[:3]) + "..."
        
        summary += f"## {i}. Cluster {cluster_id}\n\n"
        summary += f"**Products:** {product_examples_text}\n\n"
        summary += f"**Size:** {size} products\n\n"
        summary += f"**Average margin:** {mean_margin:.2f}%\n\n"
        summary += f"**Margin standard deviation:** {std_margin:.2f}%\n\n"
        summary += f"**Summary:** {summary_text}\n\n"
        summary += f"[Full analysis](cluster_{cluster_id}_analysis.md)\n\n---\n\n"
    
    # Write summary report
    summary_path = os.path.join(args.output_dir, 'cluster_analysis_summary.md')
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")
    print(f"To view the summary report, open: {os.path.join(args.output_dir, 'cluster_analysis_summary.md')}")


if __name__ == "__main__":
    main()
