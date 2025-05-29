#!/usr/bin/env python3
"""
USDA Mapping Analysis Component

Evaluates how well our clusters align with the expected groupings based on the USDA code mappings.
This analysis helps identify why certain products that should be grouped together are not clustered properly.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_usda_mapping(mapping_path: str) -> Dict[str, List[str]]:
    """
    Load the USDA to product code mapping data from the Corrected_mapping.xlsx file.
    
    Args:
        mapping_path: Path to the mapping Excel file
        
    Returns:
        Dictionary mapping standardized product names to lists of product codes
    """
    try:
        # Load the Excel file
        df = pd.read_excel(mapping_path)
        
        # Create mapping from USDA code to product codes
        mapping = {}
        
        # The distributor code columns
        distributor_columns = ['Fulton_code', 'Pritzlaff_code', 'Queen_code', 'Moesle_code', 'Anmar_code']
        
        for _, row in df.iterrows():
            usda_code = str(row['USDA_Code']).strip()
            
            # Get all product codes from the distributor columns
            product_codes = []
            for col in distributor_columns:
                if pd.notna(row.get(col)) and str(row.get(col)).strip():
                    code = str(row.get(col)).strip()
                    # Strip the -1 or -2 suffix from product codes 
                    if code.endswith('-1') or code.endswith('-2'):
                        code = code.rsplit('-', 1)[0]
                    product_codes.append(code)
            
            # Filter out any empty strings
            product_codes = [code for code in product_codes if code]
            
            if usda_code and product_codes:
                mapping[usda_code] = product_codes
                
        return mapping
    
    except Exception as e:
        print(f"Error loading USDA mapping data: {e}")
        return {}

def load_clusters(clusters_path: str) -> Dict[str, List[str]]:
    """
    Load the clustering results.
    
    Args:
        clusters_path: Path to the clusters JSON file
        
    Returns:
        Dictionary mapping cluster IDs to lists of product codes
    """
    try:
        with open(clusters_path, 'r') as f:
            clusters = json.load(f)
        return clusters
    except Exception as e:
        print(f"Error loading clusters: {e}")
        return {}

def load_products_data(products_path: str) -> pd.DataFrame:
    """
    Load the prepared products data.
    
    Args:
        products_path: Path to the prepared products CSV file
        
    Returns:
        DataFrame containing product information
    """
    try:
        return pd.read_csv(products_path)
    except Exception as e:
        print(f"Error loading products data: {e}")
        return pd.DataFrame()

def load_category_products(category_products_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load the category-to-products mapping data.
    
    Args:
        category_products_path: Path to the category products JSON file
        
    Returns:
        Dictionary mapping categories to product code lists
    """
    try:
        with open(category_products_path, 'r') as f:
            category_products = json.load(f)
        return category_products
    except Exception as e:
        print(f"Error loading category products: {e}")
        return {}

def build_inverse_cluster_map(clusters: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build a map from product code to cluster ID.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        
    Returns:
        Dictionary mapping product codes to their cluster IDs
    """
    product_to_cluster = {}
    for cluster_id, products in clusters.items():
        for product in products:
            product_to_cluster[product] = cluster_id
    return product_to_cluster

def analyze_usda_grouping_alignment(
    clusters: Dict[str, List[str]],
    usda_mapping: Dict[str, List[str]],
    prepared_df: pd.DataFrame,
    category_products: Optional[Dict[str, Dict[str, List[str]]]] = None
) -> Dict[str, Any]:
    """
    Analyze how well clusters align with the expected USDA groupings.
    
    Args:
        clusters: Dictionary of clusters
        usda_mapping: Dictionary mapping standardized names to product codes
        prepared_df: DataFrame containing all prepared products
        category_products: Optional dictionary mapping categories to product codes
        
    Returns:
        Dictionary containing the analysis results
    """
    # Build product code to cluster map for quick lookups
    product_to_cluster = build_inverse_cluster_map(clusters)
    
    # Build product code to category map (if available)
    product_to_category = {}
    if category_products:
        for category, data in category_products.items():
            if 'clustered' in data:
                for product in data['clustered']:
                    product_to_category[product] = category
            if 'noise' in data:
                for product in data['noise']:
                    product_to_category[product] = f"{category} (noise)"
    
    # Create a lookup for product descriptions
    product_descriptions = {}
    for _, row in prepared_df.iterrows():
        if 'product_code' in row and 'product_description' in row:
            product_descriptions[str(row['product_code'])] = row['product_description']
    
    # Analyze each USDA grouping
    results = {
        'grouping_analysis': [],
        'summary': {
            'total_usda_groups': 0,
            'fully_aligned_groups': 0,
            'partially_aligned_groups': 0,
            'misaligned_groups': 0,
            'empty_groups': 0,
            'total_products_analyzed': 0,
            'products_in_same_cluster': 0,
            'products_in_different_clusters': 0,
            'products_not_clustered': 0,
            'reason_no_category': 0,
            'reason_refinement': 0,
            'reason_different_cluster': 0
        }
    }
    
    print(f"Analyzing alignment for {len(usda_mapping)} USDA product groupings...")
    
    for std_name, product_codes in tqdm(usda_mapping.items()):
        # Filter to products that exist in our dataset
        existing_products = [code for code in product_codes if code in product_descriptions]
        
        if not existing_products:
            results['summary']['empty_groups'] += 1
            continue
        
        # Find which clusters these products belong to
        cluster_assignments = {}
        not_clustered = []
        no_category_products = []
        refinement_removed = []
        different_cluster = []
        
        for product in existing_products:
            if product in product_to_cluster:
                cluster_id = product_to_cluster[product]
                if cluster_id not in cluster_assignments:
                    cluster_assignments[cluster_id] = []
                cluster_assignments[cluster_id].append(product)
            else:
                not_clustered.append(product)
                
                # Analyze why the product wasn't clustered
                if product not in product_to_category:
                    no_category_products.append(product)
                elif product_to_category.get(product, "").endswith("(noise)"):
                    refinement_removed.append(product)
                else:
                    different_cluster.append(product)
        
        # Calculate statistics
        total_products = len(existing_products)
        clustered_products = total_products - len(not_clustered)
        clustered_ratio = clustered_products / total_products if total_products > 0 else 0
        
        # Determine the dominant cluster (if any)
        dominant_cluster = None
        dominant_cluster_size = 0
        
        for cluster_id, products in cluster_assignments.items():
            if len(products) > dominant_cluster_size:
                dominant_cluster = cluster_id
                dominant_cluster_size = len(products)
        
        # Calculate how many products are in the dominant cluster
        in_dominant_cluster = dominant_cluster_size if dominant_cluster else 0
        dominant_cluster_ratio = in_dominant_cluster / total_products if total_products > 0 else 0
        
        # Determine grouping status
        status = "MISALIGNED"
        if dominant_cluster_ratio == 1.0:
            status = "FULLY_ALIGNED"
            results['summary']['fully_aligned_groups'] += 1
        elif dominant_cluster_ratio >= 0.5:
            status = "PARTIALLY_ALIGNED"
            results['summary']['partially_aligned_groups'] += 1
        else:
            results['summary']['misaligned_groups'] += 1
        
        # Record detailed information for this grouping
        grouping_info = {
            'standardized_name': std_name,
            'status': status,
            'total_products': total_products,
            'clustered_products': clustered_products,
            'clustered_ratio': clustered_ratio,
            'dominant_cluster': dominant_cluster,
            'dominant_cluster_size': dominant_cluster_size,
            'dominant_cluster_ratio': dominant_cluster_ratio,
            'cluster_assignments': {k: v for k, v in cluster_assignments.items()},
            'not_clustered': not_clustered,
            'reason_analysis': {
                'no_category': no_category_products,
                'refinement_removed': refinement_removed,
                'different_cluster': different_cluster
            }
        }
        
        results['grouping_analysis'].append(grouping_info)
        
        # Update summary statistics
        results['summary']['total_usda_groups'] += 1
        results['summary']['total_products_analyzed'] += total_products
        results['summary']['products_in_same_cluster'] += in_dominant_cluster
        results['summary']['products_in_different_clusters'] += (clustered_products - in_dominant_cluster)
        results['summary']['products_not_clustered'] += len(not_clustered)
        results['summary']['reason_no_category'] += len(no_category_products)
        results['summary']['reason_refinement'] += len(refinement_removed)
        results['summary']['reason_different_cluster'] += len(different_cluster)
    
    # Calculate overall percentages
    total_products = results['summary']['total_products_analyzed']
    if total_products > 0:
        results['summary']['percent_in_same_cluster'] = results['summary']['products_in_same_cluster'] / total_products * 100
        results['summary']['percent_in_different_clusters'] = results['summary']['products_in_different_clusters'] / total_products * 100
        results['summary']['percent_not_clustered'] = results['summary']['products_not_clustered'] / total_products * 100
    
    return results

def generate_usda_mapping_report(analysis_result: Dict[str, Any]) -> str:
    """
    Generate a detailed report on USDA mapping alignment.
    
    Args:
        analysis_result: Results from analyze_usda_grouping_alignment
        
    Returns:
        Markdown-formatted report
    """
    summary = analysis_result['summary']
    
    report = f"""# USDA Mapping Analysis Report

## Summary

- **Total USDA Groups Analyzed**: {summary['total_usda_groups']}
  - Fully Aligned: {summary['fully_aligned_groups']} ({summary['fully_aligned_groups'] / summary['total_usda_groups'] * 100:.1f}%)
  - Partially Aligned: {summary['partially_aligned_groups']} ({summary['partially_aligned_groups'] / summary['total_usda_groups'] * 100:.1f}%)
  - Misaligned: {summary['misaligned_groups']} ({summary['misaligned_groups'] / summary['total_usda_groups'] * 100:.1f}%)
  - Empty (No products found): {summary['empty_groups']}

- **Product Distribution**:
  - Total Products Analyzed: {summary['total_products_analyzed']}
  - Products in Same Cluster: {summary['products_in_same_cluster']} ({summary.get('percent_in_same_cluster', 0):.1f}%)
  - Products in Different Clusters: {summary['products_in_different_clusters']} ({summary.get('percent_in_different_clusters', 0):.1f}%)
  - Products Not Clustered: {summary['products_not_clustered']} ({summary.get('percent_not_clustered', 0):.1f}%)

- **Reasons for Non-Clustering**:
  - No Category Description: {summary['reason_no_category']} ({summary['reason_no_category'] / summary['products_not_clustered'] * 100:.1f}% of non-clustered)
  - Thrown Out During Refinement: {summary['reason_refinement']} ({summary['reason_refinement'] / summary['products_not_clustered'] * 100:.1f}% of non-clustered)
  - Put in Different Cluster: {summary['reason_different_cluster']} ({summary['reason_different_cluster'] / summary['products_not_clustered'] * 100:.1f}% of non-clustered)

## Detailed Analysis

The following section provides detailed analysis for each USDA product grouping.
"""

    # Sort groupings by status (misaligned first, then partially aligned, then fully aligned)
    sorted_groupings = sorted(
        analysis_result['grouping_analysis'],
        key=lambda x: (
            0 if x['status'] == 'MISALIGNED' else 
            (1 if x['status'] == 'PARTIALLY_ALIGNED' else 2),
            -x['total_products']  # Secondary sort by total products (descending)
        )
    )
    
    # Add detailed information for problematic groupings first
    report += "\n### Problematic USDA Groupings\n\n"
    problematic_count = 0
    
    for grouping in sorted_groupings:
        if grouping['status'] != 'FULLY_ALIGNED' and grouping['total_products'] >= 2:
            problematic_count += 1
            report += f"#### {problematic_count}. {grouping['standardized_name']}\n\n"
            report += f"- **Status**: {grouping['status']}\n"
            report += f"- **Products**: {grouping['total_products']} total, {grouping['clustered_products']} clustered ({grouping['clustered_ratio']*100:.1f}%)\n"
            
            if grouping['dominant_cluster']:
                report += f"- **Dominant Cluster**: {grouping['dominant_cluster']} with {grouping['dominant_cluster_size']} products ({grouping['dominant_cluster_ratio']*100:.1f}%)\n"
            
            # List products by cluster
            if grouping['cluster_assignments']:
                report += "\n**Products by Cluster**:\n\n"
                for cluster_id, products in grouping['cluster_assignments'].items():
                    report += f"- Cluster {cluster_id}: {len(products)} products\n"
            
            # List non-clustered products with reasons
            if grouping['not_clustered']:
                report += "\n**Non-Clustered Products**:\n\n"
                
                if grouping['reason_analysis']['no_category']:
                    report += f"- No Category Description: {len(grouping['reason_analysis']['no_category'])} products\n"
                
                if grouping['reason_analysis']['refinement_removed']:
                    report += f"- Thrown Out During Refinement: {len(grouping['reason_analysis']['refinement_removed'])} products\n"
                
                if grouping['reason_analysis']['different_cluster']:
                    report += f"- Put in Different Cluster: {len(grouping['reason_analysis']['different_cluster'])} products\n"
            
            report += "\n"
    
    if problematic_count == 0:
        report += "No problematic USDA groupings found.\n\n"
    
    # Add summary of fully aligned groupings
    fully_aligned = [g for g in sorted_groupings if g['status'] == 'FULLY_ALIGNED' and g['total_products'] >= 2]
    if fully_aligned:
        report += f"\n### Fully Aligned USDA Groupings ({len(fully_aligned)} groupings)\n\n"
        for i, grouping in enumerate(fully_aligned[:10]):  # Show only top 10
            report += f"{i+1}. **{grouping['standardized_name']}** - {grouping['total_products']} products in cluster {grouping['dominant_cluster']}\n"
        
        if len(fully_aligned) > 10:
            report += f"\n*... and {len(fully_aligned) - 10} more fully aligned groupings*\n"
    
    return report

def create_directory_if_not_exists(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze USDA mapping alignment with clusters")
    
    # Data paths
    parser.add_argument("--clusters_path", type=str, help="Path to clusters JSON file")
    parser.add_argument("--mapping_path", type=str, 
                        default=os.path.join("data", "CorrectMapping", "product_mapping_semantic.xlsx"),
                        help="Path to USDA mapping file")
    parser.add_argument("--products_path", type=str, help="Path to prepared products CSV file")
    parser.add_argument("--category_products_path", type=str, help="Path to category products JSON file")
    parser.add_argument("--output_dir", type=str, help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not args.mapping_path:
        args.mapping_path = os.path.join(project_root, "Source_data", "Actuals", "Corrected_mapping.xlsx")
    
    if not args.clusters_path:
        # Default to refined category clusters
        args.clusters_path = os.path.join(project_root, "product_clustering", "data", "category_clustering", 
                                        "refined", "refined_category_clusters.json")
    
    if not args.products_path:
        args.products_path = os.path.join(project_root, "product_clustering", "data", "prepared_products.csv")
    
    if not args.category_products_path:
        args.category_products_path = os.path.join(project_root, "product_clustering", "data", 
                                                 "category_clustering", "category_products.json")
    
    if not args.output_dir:
        args.output_dir = os.path.join(project_root, "product_clustering", "data", "analysis")
    
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(args.output_dir)
    
    print(f"Loading USDA mapping from {args.mapping_path}...")
    usda_mapping = load_usda_mapping(args.mapping_path)
    print(f"Loaded {len(usda_mapping)} standardized product groupings")
    
    print(f"Loading clusters from {args.clusters_path}...")
    clusters = load_clusters(args.clusters_path)
    print(f"Loaded {len(clusters)} clusters")
    
    print(f"Loading products data from {args.products_path}...")
    products_df = load_products_data(args.products_path)
    print(f"Loaded {len(products_df)} products")
    
    category_products = None
    if os.path.exists(args.category_products_path):
        print(f"Loading category products from {args.category_products_path}...")
        category_products = load_category_products(args.category_products_path)
        print(f"Loaded {len(category_products)} categories")
    
    print("Analyzing USDA grouping alignment...")
    analysis_result = analyze_usda_grouping_alignment(
        clusters=clusters,
        usda_mapping=usda_mapping,
        prepared_df=products_df,
        category_products=category_products
    )
    
    print("Generating report...")
    report = generate_usda_mapping_report(analysis_result)
    
    # Save report
    report_path = os.path.join(args.output_dir, "usda_mapping_analysis.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save raw analysis results
    results_path = os.path.join(args.output_dir, "usda_mapping_analysis.json")
    with open(results_path, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"Analysis complete. Report saved to {report_path}")
    print(f"Raw analysis results saved to {results_path}")
    
    # Provide a summary of the results
    summary = analysis_result['summary']
    print("\nUSDA Mapping Analysis Summary:")
    print(f"Total USDA Groups: {summary['total_usda_groups']}")
    print(f"Fully Aligned: {summary['fully_aligned_groups']} ({summary['fully_aligned_groups'] / summary['total_usda_groups'] * 100:.1f}%)")
    print(f"Partially Aligned: {summary['partially_aligned_groups']} ({summary['partially_aligned_groups'] / summary['total_usda_groups'] * 100:.1f}%)")
    print(f"Misaligned: {summary['misaligned_groups']} ({summary['misaligned_groups'] / summary['total_usda_groups'] * 100:.1f}%)")
    
    return report_path, results_path

if __name__ == "__main__":
    main()
