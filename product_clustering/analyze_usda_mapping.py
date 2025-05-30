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
        
        # Check if we have the expected distributor columns or a simplified structure
        distributor_columns = ['Fulton_code', 'Pritzlaff_code', 'Queen_code', 'Moesle_code', 'Anmar_code']
        has_distributor_columns = any(col in df.columns for col in distributor_columns)
        
        for _, row in df.iterrows():
            usda_code = str(row['USDA_Code']).strip()
            
            # Get all product codes from either distributor columns or product_code column
            product_codes = []
            
            if has_distributor_columns:
                # Original approach with distributor columns
                for col in distributor_columns:
                    if pd.notna(row.get(col)) and str(row.get(col)).strip():
                        code = str(row.get(col)).strip()
                        # Strip the -1 or -2 suffix from product codes 
                        if code.endswith('-1') or code.endswith('-2'):
                            code = code.rsplit('-', 1)[0]
                        product_codes.append(code)
            elif 'product_code' in df.columns:
                # Alternative approach with direct product_code column
                if pd.notna(row.get('product_code')) and str(row.get('product_code')).strip():
                    code = str(row.get('product_code')).strip()
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

def analyze_usda_mapping(clusters, usda_mapping, prepared_df, category_products=None, original_clusters=None):
    """
    Analyze how well clusters align with USDA product groupings.
    
    Args:
        clusters: Dictionary of refined clusters (after cross-encoder)
        usda_mapping: Dictionary mapping standardized names to product codes
        prepared_df: DataFrame containing all prepared products
        category_products: Optional dictionary mapping categories to product codes
        original_clusters: Optional dictionary of original clusters (before cross-encoder)
        
    Returns:
        Dictionary containing the analysis results
    """
    # Build product code to cluster map for quick lookups (for refined clusters)
    product_to_refined_cluster = build_inverse_cluster_map(clusters)
    
    # Build product code to original cluster map if available
    product_to_original_cluster = {}
    if original_clusters:
        product_to_original_cluster = build_inverse_cluster_map(original_clusters)
    
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
    
    # Initialize result structure with enhanced diagnostics
    results = {
        "summary": {
            "total_usda_groups": len(usda_mapping),
            "fully_aligned_groups": 0,
            "partially_aligned_groups": 0,
            "misaligned_groups": 0,
            "empty_groups": 0,
            "total_products_analyzed": 0,
            "products_in_same_cluster": 0,
            "products_in_different_clusters": 0,
            "products_not_clustered": 0,
            # Original reason categories (kept for backward compatibility)
            "reason_no_category": 0,
            "reason_refinement": 0,
            "reason_different_cluster": 0,
            # Enhanced diagnostic categories
            "in_original_not_refined": 0,  # Present in original clusters but removed by refinement
            "never_prepared": 0,          # Not in the prepared data
            "missing_from_all_clusters": 0 # Not in any cluster stage
        },
        "grouping_analysis": [],  # Add this key for storing detailed group analysis
        "groups": {}
    }
    
    # Count all individual products across all USDA groups that actually exist in our dataset
    all_products = []
    valid_usda_products = []
    
    for usda_code, product_codes in usda_mapping.items():
        # Only include products that exist in our prepared data
        existing_products = [code for code in product_codes if code in product_descriptions]
        all_products.extend(product_codes)  # All products from mapping (for debugging)
        valid_usda_products.extend(existing_products)  # Only valid products
    
    # Update total products analyzed - count unique valid products to avoid duplicates
    results['summary']['total_products_analyzed'] = len(set(valid_usda_products))
    
    print(f"Total unique products in USDA mapping: {len(set(all_products))}")
    print(f"Valid unique products in dataset: {results['summary']['total_products_analyzed']}")
    print(f"Total products including duplicates: {len(all_products)}")
    print(f"Number of USDA categories: {len(usda_mapping)}")
    
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
        
        # Initialize detailed tracking categories
        in_original_not_refined = []
        missing_from_all_clusters = []
        never_prepared = []
        
        for product in existing_products:
            # Check if product is in refined clusters
            if product in product_to_refined_cluster:
                cluster_id = product_to_refined_cluster[product]
                if cluster_id not in cluster_assignments:
                    cluster_assignments[cluster_id] = []
                cluster_assignments[cluster_id].append(product)
            else:
                # Product not in refined clusters, check where it was filtered out
                not_clustered.append(product)
                
                # Check if it was in original clusters but removed by refinement
                if original_clusters and product in product_to_original_cluster:
                    in_original_not_refined.append(product)
                    refinement_removed.append(product)  # For backward compatibility
                # Check if it was never processed into the prepared data
                elif product not in product_descriptions:
                    never_prepared.append(product)
                    no_category_products.append(product)  # Map to traditional category
                # Check if it's actually in the dataset but missing from clusters
                # This is likely due to not having a category description
                elif product in product_descriptions:
                    missing_from_all_clusters.append(product)
                    no_category_products.append(product)  # Map to traditional category
                # If we can't determine the reason, mark as different cluster
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
            status = "MISALIGNED"
            results['summary']['misaligned_groups'] += 1
            
        # Update product-level summary stats - count actual products, not just groups
        results['summary']['products_in_same_cluster'] += in_dominant_cluster
        
        # Count products in non-dominant clusters
        products_in_other_clusters = 0
        for cluster_id, products in cluster_assignments.items():
            if cluster_id != dominant_cluster:
                products_in_other_clusters += len(products)
        
        results['summary']['products_in_different_clusters'] += products_in_other_clusters
        results['summary']['products_not_clustered'] += len(not_clustered)
        
        # Track reasons for non-clustering at the product level
        # Update traditional reason categories (for backward compatibility)
        results['summary']['reason_no_category'] += len(no_category_products)
        results['summary']['reason_refinement'] += len(refinement_removed)
        results['summary']['reason_different_cluster'] += len(different_cluster)
        
        # Update enhanced diagnostic categories
        results['summary']['in_original_not_refined'] += len(in_original_not_refined)
        results['summary']['never_prepared'] += len(never_prepared)
        results['summary']['missing_from_all_clusters'] += len(missing_from_all_clusters)
        
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
        # These group-level metrics are aggregated for detailed analysis
        results['summary']['products_in_same_cluster'] += in_dominant_cluster
        results['summary']['products_in_different_clusters'] += (clustered_products - in_dominant_cluster)
        results['summary']['products_not_clustered'] += len(not_clustered)
        results['summary']['reason_no_category'] += len(no_category_products)
        results['summary']['reason_refinement'] += len(refinement_removed)
        results['summary']['reason_different_cluster'] += len(different_cluster)
    
    # Track unique products in each category to avoid double-counting
    unique_products_in_same_cluster = set()
    unique_products_in_different_clusters = set()
    unique_products_not_clustered = set()
    
    # Go through each grouping analysis to collect unique products
    for grouping in results['grouping_analysis']:
        # Get products in the dominant cluster
        dominant_cluster = grouping.get('dominant_cluster')
        if dominant_cluster and dominant_cluster in grouping['cluster_assignments']:
            for product in grouping['cluster_assignments'][dominant_cluster]:
                unique_products_in_same_cluster.add(product)
        
        # Get products in different clusters
        for cluster_id, products in grouping['cluster_assignments'].items():
            if cluster_id != dominant_cluster:
                for product in products:
                    unique_products_in_different_clusters.add(product)
        
        # Get non-clustered products
        for product in grouping['not_clustered']:
            unique_products_not_clustered.add(product)
    
    # Add unique product counts to summary
    results['summary']['unique_products_in_same_cluster'] = len(unique_products_in_same_cluster)
    results['summary']['unique_products_in_different_clusters'] = len(unique_products_in_different_clusters)
    results['summary']['unique_products_not_clustered'] = len(unique_products_not_clustered)
    
    return results

def export_detailed_csv(clusters, usda_mapping, prepared_df, output_path):
    """
    Export a detailed CSV file with product details, cluster assignments, and USDA codes.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of product codes
        usda_mapping: Dictionary mapping USDA codes to lists of product codes
        prepared_df: DataFrame with product details
        output_path: Path to save the CSV file
    """
    # Import abbreviation expansion functionality
    from src.abbreviation_translator import expand_abbreviations
    
    # Create inverse mappings for quick lookups
    product_to_cluster = {}
    for cluster_id, products in clusters.items():
        for product in products:
            product_to_cluster[product] = cluster_id
    
    # Create reverse mapping from product to USDA code
    product_to_usda = {}
    for usda_code, products in usda_mapping.items():
        for product in products:
            product_to_usda[product] = usda_code
    
    # Create a list to store the data
    csv_data = []
    
    # Process each product in the prepared data
    for _, row in prepared_df.iterrows():
        if 'product_code' not in row or pd.isna(row['product_code']):
            continue
            
        product_code = str(row['product_code'])
        description = row.get('product_description', '') if not pd.isna(row.get('product_description', '')) else ''
        company = row.get('distributor', '') if not pd.isna(row.get('distributor', '')) else ''
        
        # Expand abbreviations in the description
        expanded_description = expand_abbreviations(description) if description else ''
        
        # Get cluster assignment
        cluster = product_to_cluster.get(product_code, 'Not Clustered')
        
        # Get USDA code
        usda_code = product_to_usda.get(product_code, 'No USDA Code')
        
        # Add to data list
        csv_data.append({
            'product_id': product_code,
            'description': description,
            'expanded_description': expanded_description,
            'company': company,
            'cluster': cluster,
            'usda_code': usda_code
        })
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    
    print(f"Exported {len(df)} products to {output_path}")
    return output_path

def generate_usda_mapping_report(analysis_result: Dict[str, Any]) -> str:
    """
    Generate a detailed report on USDA mapping alignment.
    
    Args:
        analysis_result: Results from analyze_usda_mapping
        
    Returns:
        Markdown-formatted report
    """
    summary = analysis_result['summary']
    
    report = f"""# USDA Mapping Analysis Report

## Summary

- **Total USDA Groups Analyzed**: {summary['total_usda_groups']}
  - Fully Aligned: {summary['fully_aligned_groups']} ({(summary['fully_aligned_groups'] / summary['total_usda_groups'] * 100) if summary['total_usda_groups'] > 0 else 0:.1f}%)
  - Partially Aligned: {summary['partially_aligned_groups']} ({(summary['partially_aligned_groups'] / summary['total_usda_groups'] * 100) if summary['total_usda_groups'] > 0 else 0:.1f}%)
  - Misaligned: {summary['misaligned_groups']} ({(summary['misaligned_groups'] / summary['total_usda_groups'] * 100) if summary['total_usda_groups'] > 0 else 0:.1f}%)
  - Empty (No products found): {summary['empty_groups']}

- **Product Distribution** (Individual Products, not Groups):
  - Total Products Analyzed: {summary['total_products_analyzed']}
  - Products in Same Cluster as their Group: {summary.get('unique_products_in_same_cluster', 0)} ({(summary.get('unique_products_in_same_cluster', 0) / summary['total_products_analyzed'] * 100) if summary['total_products_analyzed'] > 0 else 0:.1f}%)
  - Products in Different Clusters: {summary.get('unique_products_in_different_clusters', 0)} ({(summary.get('unique_products_in_different_clusters', 0) / summary['total_products_analyzed'] * 100) if summary['total_products_analyzed'] > 0 else 0:.1f}%)
  - Products Not Found in Any Cluster: {summary.get('unique_products_not_clustered', 0)} ({(summary.get('unique_products_not_clustered', 0) / summary['total_products_analyzed'] * 100) if summary['total_products_analyzed'] > 0 else 0:.1f}%)

- **Reasons for Non-Clustering (Enhanced Diagnostics)**:
  - **Found in Original Clusters, Removed by Refinement**: {summary['in_original_not_refined']} ({(summary['in_original_not_refined'] / summary['products_not_clustered'] * 100) if summary['products_not_clustered'] > 0 else 0:.1f}% of non-clustered)
  - **Never Prepared for Clustering**: {summary['never_prepared']} ({(summary['never_prepared'] / summary['products_not_clustered'] * 100) if summary['products_not_clustered'] > 0 else 0:.1f}% of non-clustered)
  - **Missing from All Clustering Stages**: {summary['missing_from_all_clusters']} ({(summary['missing_from_all_clusters'] / summary['products_not_clustered'] * 100) if summary['products_not_clustered'] > 0 else 0:.1f}% of non-clustered)
  
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
    parser.add_argument('--clusters_path', help='Path to refined clusters JSON file')
    parser.add_argument('--original_clusters_path', help='Path to original clusters JSON file (before refinement)')
    parser.add_argument('--mapping_path', help='Path to USDA mapping Excel file')
    parser.add_argument('--products_path', help='Path to prepared products CSV file')
    parser.add_argument('--category_products_path', help='Path to category products JSON file')
    parser.add_argument('--output_dir', help='Directory to save analysis results')
    
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
    
    print(f"Loading refined clusters from {args.clusters_path}...")
    clusters = load_clusters(args.clusters_path)
    print(f"Loaded {len(clusters)} refined clusters")
    
    # Load original clusters if path provided
    original_clusters = None
    if args.original_clusters_path and os.path.exists(args.original_clusters_path):
        print(f"Loading original clusters from {args.original_clusters_path}...")
        original_clusters = load_clusters(args.original_clusters_path)
        print(f"Loaded {len(original_clusters)} original clusters")
    
    print(f"Loading products data from {args.products_path}...")
    products_df = load_products_data(args.products_path)
    print(f"Loaded {len(products_df)} products")
    
    category_products = None
    if os.path.exists(args.category_products_path):
        print(f"Loading category products from {args.category_products_path}...")
        category_products = load_category_products(args.category_products_path)
        print(f"Loaded {len(category_products)} categories")
    
    print("Analyzing USDA grouping alignment with enhanced diagnostics...")
    analysis_result = analyze_usda_mapping(
        clusters=clusters,
        usda_mapping=usda_mapping,
        prepared_df=products_df,
        category_products=category_products,
        original_clusters=original_clusters
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
        
    # Export detailed CSV with cluster, product_id, description, company, and USDA code
    csv_path = os.path.join(args.output_dir, "usda_cluster_mapping.csv")
    export_detailed_csv(clusters, usda_mapping, products_df, csv_path)
    
    print(f"Analysis complete. Report saved to {report_path}")
    print(f"Raw analysis results saved to {results_path}")
    print(f"Detailed CSV with cluster, product_id, description, company, and USDA code saved to {csv_path}")
    
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
