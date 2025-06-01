#!/usr/bin/env python3
"""
Generate Variance Report Script

This script analyzes the top clusters with highest variance in gross margin percentage,
determines whether price or cost is the main driver of inconsistency, and generates
a management-friendly markdown report with explanations and recommendations.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Set, Optional

def load_data(stats_file: str, products_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cluster statistics and product details.
    
    Args:
        stats_file: Path to cluster statistics CSV
        products_file: Path to cluster product details CSV
        
    Returns:
        Tuple of DataFrames: (cluster_stats, cluster_products)
    """
    # Load cluster statistics
    try:
        stats_df = pd.read_csv(stats_file)
        print(f"Loaded cluster statistics for {len(stats_df)} clusters")
    except Exception as e:
        print(f"Error loading cluster statistics: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Load product details
    try:
        products_df = pd.read_csv(products_file)
        print(f"Loaded details for {len(products_df)} products")
    except Exception as e:
        print(f"Error loading product details: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    return stats_df, products_df

def identify_top_variance_clusters(stats_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Identify the top n clusters with highest GP% variance.
    
    Args:
        stats_df: DataFrame with cluster statistics
        n: Number of top clusters to return
        
    Returns:
        DataFrame with top n clusters
    """
    # Check for appropriate variance metric column
    variance_metrics = [
        "Implied GP% (actual)_cv",
        "Implied GP% (actual)_std",
        "gp_percent_cv",
        "gp_percent_std"
    ]
    
    variance_metric = None
    for metric in variance_metrics:
        if metric in stats_df.columns:
            variance_metric = metric
            break
    
    if variance_metric is None:
        print("Error: No suitable variance metric found in cluster statistics")
        return pd.DataFrame()
    
    # Sort by variance metric and get top n
    top_clusters = stats_df.sort_values(by=variance_metric, ascending=False).head(n)
    
    if len(top_clusters) == 0:
        print("No clusters found with variance data")
        return pd.DataFrame()
        
    print(f"Identified top {len(top_clusters)} clusters with highest {variance_metric}")
    return top_clusters

def determine_variance_driver(cluster_df: pd.DataFrame) -> Dict:
    """
    Determine whether price or cost is driving the variance in GP%.
    
    Args:
        cluster_df: DataFrame with products in a cluster
        
    Returns:
        Dictionary with analysis results
    """
    # Calculate coefficient of variation for price and cost
    price_cols = ["SalesPrice", "sales_price"]
    cost_cols = ["AccountingCost", "accounting_cost"]
    gp_pct_cols = ["Implied GP% (actual)", "gp_percent"]
    
    # Find available columns
    price_col = next((col for col in price_cols if col in cluster_df.columns), None)
    cost_col = next((col for col in cost_cols if col in cluster_df.columns), None)
    gp_pct_col = next((col for col in gp_pct_cols if col in cluster_df.columns), None)
    
    if not all([price_col, cost_col, gp_pct_col]):
        return {
            "driver": "unknown", 
            "reason": "Missing required data columns",
            "price_cv": None,
            "cost_cv": None,
            "price_range_pct": None,
            "cost_range_pct": None
        }
    
    # Get non-null values
    price_values = cluster_df[price_col].dropna()
    cost_values = cluster_df[cost_col].dropna()
    
    # Skip if not enough data
    if len(price_values) < 2 or len(cost_values) < 2:
        return {
            "driver": "unknown", 
            "reason": "Not enough non-null values",
            "price_cv": None,
            "cost_cv": None,
            "price_range_pct": None,
            "cost_range_pct": None
        }
    
    # Calculate CV (coefficient of variation) - std/mean
    price_mean = price_values.mean()
    price_std = price_values.std()
    price_cv = price_std / price_mean if price_mean != 0 else float('inf')
    
    cost_mean = cost_values.mean()
    cost_std = cost_values.std()
    cost_cv = cost_std / cost_mean if cost_mean != 0 else float('inf')
    
    # Calculate range as percentage of mean
    price_range = price_values.max() - price_values.min()
    price_range_pct = price_range / price_mean * 100 if price_mean != 0 else float('inf')
    
    cost_range = cost_values.max() - cost_values.min()
    cost_range_pct = cost_range / cost_mean * 100 if cost_mean != 0 else float('inf')
    
    # Determine driver based on CV
    if price_cv > cost_cv:
        driver = "price"
        reason = f"Price variation ({price_cv:.2f}) is higher than cost variation ({cost_cv:.2f})"
    else:
        driver = "cost"
        reason = f"Cost variation ({cost_cv:.2f}) is higher than price variation ({price_cv:.2f})"
    
    return {
        "driver": driver,
        "reason": reason,
        "price_cv": price_cv,
        "cost_cv": cost_cv,
        "price_range_pct": price_range_pct,
        "cost_range_pct": cost_range_pct,
        "price_min": price_values.min(),
        "price_max": price_values.max(),
        "price_mean": price_mean,
        "cost_min": cost_values.min(),
        "cost_max": cost_values.max(),
        "cost_mean": cost_mean
    }

def analyze_cluster(cluster_id: str, stats: pd.Series, products: pd.DataFrame) -> Dict:
    """
    Perform detailed analysis of a cluster.
    
    Args:
        cluster_id: Cluster ID
        stats: Series with cluster statistics
        products: DataFrame with products in the cluster
        
    Returns:
        Dictionary with analysis results
    """
    # Basic cluster info
    analysis = {
        "cluster_id": cluster_id,
        "num_products": len(products),
        "metrics": {}
    }
    
    # Add product descriptions if available
    desc_cols = ["product_description", "ProductDescription"]
    desc_col = next((col for col in desc_cols if col in products.columns), None)
    
    if desc_col:
        analysis["product_descriptions"] = products[desc_col].tolist()
    
    # Get metric statistics
    gp_pct_cols = ["Implied GP% (actual)", "gp_percent"]
    gp_dollars_cols = ["Implied GP$ (actual)", "gp_dollars"]
    price_cols = ["SalesPrice", "sales_price"]
    cost_cols = ["AccountingCost", "accounting_cost"]
    
    # Find available columns
    gp_pct_col = next((col for col in gp_pct_cols if col in products.columns), None)
    gp_dollars_col = next((col for col in gp_dollars_cols if col in products.columns), None)
    price_col = next((col for col in price_cols if col in products.columns), None)
    cost_col = next((col for col in cost_cols if col in products.columns), None)
    
    # Add metric statistics
    for metric_name, col in [
        ("gp_percent", gp_pct_col),
        ("gp_dollars", gp_dollars_col),
        ("sales_price", price_col),
        ("accounting_cost", cost_col)
    ]:
        if col and col in products.columns:
            values = products[col].dropna()
            if len(values) >= 2:
                analysis["metrics"][metric_name] = {
                    "mean": values.mean(),
                    "median": values.median(),
                    "min": values.min(),
                    "max": values.max(),
                    "range": values.max() - values.min(),
                    "std": values.std(),
                    "cv": values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                }
    
    # Determine variance driver
    variance_driver = determine_variance_driver(products)
    analysis["variance_driver"] = variance_driver
    
    # Add individual product data
    analysis["products"] = []
    
    id_cols = ["product_code", "ProductCode"]
    id_col = next((col for col in id_cols if col in products.columns), None)
    
    if id_col:
        for _, row in products.iterrows():
            product = {
                "product_id": row[id_col]
            }
            
            # Add metrics
            for metric_name, col in [
                ("gp_percent", gp_pct_col),
                ("gp_dollars", gp_dollars_col),
                ("sales_price", price_col),
                ("accounting_cost", cost_col)
            ]:
                if col and col in products.columns:
                    product[metric_name] = row[col]
            
            # Add description if available
            if desc_col:
                product["description"] = row[desc_col]
            
            analysis["products"].append(product)
    
    return analysis

def generate_explanation(analysis: Dict) -> str:
    """
    Generate a human-readable explanation of the cluster variance.
    
    Args:
        analysis: Dictionary with cluster analysis results
        
    Returns:
        String with explanation text
    """
    cluster_id = analysis["cluster_id"]
    num_products = analysis["num_products"]
    
    # Get product descriptions if available
    product_descriptions = analysis.get("product_descriptions", [])
    product_desc_text = ""
    if product_descriptions:
        product_desc_text = "Products in this cluster include:\n\n"
        for i, desc in enumerate(product_descriptions[:5]):
            if desc and not pd.isna(desc):
                product_desc_text += f"- {desc}\n"
        
        if len(product_descriptions) > 5:
            product_desc_text += f"- and {len(product_descriptions) - 5} more...\n"
        
        product_desc_text += "\n"
    
    # Get metric statistics
    metrics = analysis.get("metrics", {})
    gp_pct = metrics.get("gp_percent", {})
    
    if not gp_pct:
        return f"## Cluster {cluster_id}\n\nInsufficient data for analysis."
    
    # Format GP% statistics
    gp_min = gp_pct.get("min", 0) * 100
    gp_max = gp_pct.get("max", 0) * 100
    gp_mean = gp_pct.get("mean", 0) * 100
    gp_range = gp_pct.get("range", 0) * 100
    gp_cv = gp_pct.get("cv", 0)
    
    # Get price and cost statistics
    price_stats = metrics.get("sales_price", {})
    cost_stats = metrics.get("accounting_cost", {})
    
    price_min = price_stats.get("min", 0)
    price_max = price_stats.get("max", 0)
    price_mean = price_stats.get("mean", 0)
    price_range = price_stats.get("range", 0)
    
    cost_min = cost_stats.get("min", 0)
    cost_max = cost_stats.get("max", 0)
    cost_mean = cost_stats.get("mean", 0)
    cost_range = cost_stats.get("range", 0)
    
    # Get variance driver
    variance_driver = analysis.get("variance_driver", {})
    driver = variance_driver.get("driver", "unknown")
    driver_reason = variance_driver.get("reason", "")
    
    # Generate explanation
    explanation = f"## Cluster {cluster_id}\n\n"
    explanation += f"This cluster contains {num_products} products with significant gross margin inconsistency.\n\n"
    explanation += product_desc_text
    
    explanation += f"### Margin Inconsistency Analysis\n\n"
    explanation += f"The gross margin percentage ranges from {gp_min:.1f}% to {gp_max:.1f}%, "
    explanation += f"a difference of {gp_range:.1f} percentage points. "
    explanation += f"The average GP% is {gp_mean:.1f}%.\n\n"
    
    # Add price and cost information
    explanation += f"### Price and Cost Analysis\n\n"
    explanation += f"- **Price Range:** ${price_min:.2f} to ${price_max:.2f} (avg. ${price_mean:.2f})\n"
    explanation += f"- **Cost Range:** ${cost_min:.2f} to ${cost_max:.2f} (avg. ${cost_mean:.2f})\n\n"
    
    # Add variance driver explanation
    explanation += f"### Root Cause Analysis\n\n"
    
    if driver == "price":
        explanation += f"**The primary driver of margin inconsistency is pricing.**\n\n"
        explanation += f"Price variation is significantly higher than cost variation. "
        explanation += f"Products in this cluster have similar costs but are being sold at different price points, "
        explanation += f"resulting in inconsistent margins.\n\n"
    elif driver == "cost":
        explanation += f"**The primary driver of margin inconsistency is product cost.**\n\n"
        explanation += f"Cost variation is significantly higher than price variation. "
        explanation += f"Products in this cluster are being sold at similar price points, but have different costs, "
        explanation += f"resulting in inconsistent margins.\n\n"
    else:
        explanation += f"**The driver of margin inconsistency could not be determined.**\n\n"
        explanation += f"Additional analysis is needed to identify the root cause.\n\n"
    
    # Add recommendations
    explanation += f"### Recommendations\n\n"
    
    if driver == "price":
        explanation += "1. **Standardize pricing strategy** for products in this cluster.\n"
        explanation += "2. Review sales data to understand if higher-priced items in this cluster sell at comparable volumes.\n"
        explanation += "3. Consider implementing pricing guidelines for sales teams to ensure consistency.\n"
        explanation += "4. Evaluate if the higher-margin products can inform pricing strategy for the lower-margin ones.\n"
    elif driver == "cost":
        explanation += "1. **Review vendor agreements** for products in this cluster.\n"
        explanation += "2. Identify if there are opportunities to negotiate better costs for the higher-cost items.\n"
        explanation += "3. Consider consolidating vendors for similar products to achieve better volume discounts.\n"
        explanation += "4. Investigate if there are quality or specification differences that justify the cost variations.\n"
    else:
        explanation += "1. Gather more data on both pricing and cost factors.\n"
        explanation += "2. Review product specifications to confirm these products should be clustered together.\n"
        explanation += "3. Consider a detailed review of each product's pricing and sourcing strategy.\n"
    
    return explanation

def generate_report(top_clusters: pd.DataFrame, cluster_analyses: List[Dict], output_file: str):
    """
    Generate a markdown report with explanations for top variance clusters.
    
    Args:
        top_clusters: DataFrame with top variance clusters
        cluster_analyses: List of cluster analysis dictionaries
        output_file: Path to output markdown file
    """
    # Create report content
    report = "# Pricing Inconsistency Analysis Report\n\n"
    report += "## Executive Summary\n\n"
    
    # Add executive summary
    num_clusters = len(top_clusters)
    report += f"This report analyzes the top {num_clusters} product clusters with the highest "
    report += f"gross margin percentage inconsistency. For each cluster, we identify whether "
    report += f"pricing or cost is the primary driver of the inconsistency and provide "
    report += f"targeted recommendations.\n\n"
    
    # Add summary table
    report += "### Summary of Top Variance Clusters\n\n"
    report += "| Cluster ID | # Products | GP% Range | Primary Driver |\n"
    report += "|------------|------------|-----------|----------------|\n"
    
    for analysis in cluster_analyses:
        cluster_id = analysis["cluster_id"]
        num_products = analysis["num_products"]
        
        gp_pct = analysis.get("metrics", {}).get("gp_percent", {})
        gp_range = gp_pct.get("range", 0) * 100 if gp_pct else 0
        
        driver = analysis.get("variance_driver", {}).get("driver", "unknown")
        driver_text = "Price" if driver == "price" else "Cost" if driver == "cost" else "Unknown"
        
        report += f"| {cluster_id} | {num_products} | {gp_range:.1f}% | {driver_text} |\n"
    
    report += "\n## Detailed Cluster Analyses\n\n"
    
    # Add cluster explanations
    for analysis in cluster_analyses:
        explanation = generate_explanation(analysis)
        report += explanation + "\n\n---\n\n"
    
    # Add footer
    report += "\n## Next Steps\n\n"
    report += "1. Review the recommendations for each cluster and prioritize actions based on potential margin impact.\n"
    report += "2. Implement the recommended changes for the highest-impact clusters first.\n"
    report += "3. Monitor the margin consistency after changes have been implemented.\n"
    report += "4. Consider running this analysis quarterly to identify new areas of pricing inconsistency.\n"
    
    # Write report to file
    with open(output_file, "w") as f:
        f.write(report)
    
    print(f"Generated report: {output_file}")

def main():
    """Main workflow."""
    # Set file paths
    stats_file = "cluster_variance_stats.csv"
    products_file = "cluster_product_details.csv"
    output_file = "top_variance_clusters_analysis.md"
    
    print("Loading analysis data...")
    stats_df, products_df = load_data(stats_file, products_file)
    
    if stats_df.empty or products_df.empty:
        print("Error: Required data not available. Exiting.")
        return
    
    print("Identifying top variance clusters...")
    top_clusters = identify_top_variance_clusters(stats_df)
    
    if top_clusters.empty:
        print("Error: No top variance clusters identified. Exiting.")
        return
    
    print(f"Analyzing top {len(top_clusters)} clusters with highest GP% variance...")
    
    # Analyze each top cluster
    cluster_analyses = []
    for _, cluster in top_clusters.iterrows():
        cluster_id = cluster["cluster_id"]
        print(f"Analyzing cluster {cluster_id}...")
        
        # Get cluster products
        cluster_products = products_df[products_df["cluster_id"] == cluster_id]
        
        # Analyze the cluster
        analysis = analyze_cluster(cluster_id, cluster, cluster_products)
        cluster_analyses.append(analysis)
    
    # Generate report
    print("Generating markdown report...")
    generate_report(top_clusters, cluster_analyses, output_file)
    
    print("\nAnalysis complete!")
    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    main()
