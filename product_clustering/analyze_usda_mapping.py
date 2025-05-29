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

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_usda_mapping(mapping_path: str) -> Dict[str, List[str]]:
    """
    Load the USDA to product code mapping data.
    
    Args:
        mapping_path: Path to the mapping Excel file
        
    Returns:
        Dictionary mapping standardized product names to lists of product codes
    """
    try:
        # Load the Excel file
        df = pd.read_excel(mapping_path)
        
        # Create mapping from standardized name to product codes
        mapping = {}
        for _, row in df.iterrows():
            std_name = row['Standardized Product Name']
            # Convert product codes to strings and strip any whitespace
            product_codes = [str(code).strip() for code in str(row['Product Codes']).split(';')]
            # Filter out any empty strings
            product_codes = [code for code in product_codes if code]
            
            if std_name and product_codes:
                mapping[std_name] = product_codes
                
        return mapping
    
    except Exception as e:
        print(f"Error loading USDA mapping data: {e}")
        return {}

def find_product_category(product_code: str, category_products: Dict[str, List[str]]) -> Optional[str]:
    """
    Find the category a product belongs to.
    
    Args:
        product_code: Product code to look up
        category_products: Dictionary mapping categories to lists of product codes
        
    Returns:
        Category name or None if not found
    """
    for category, products in category_products.items():
        if product_code in products:
            return category
    return None

def analyze_usda_grouping_alignment(
    clusters: Dict[str, Any],
    usda_mapping: Dict[str, List[str]],
    prepared_df: pd.DataFrame,
    category_products: Optional[Dict[str, List[str]]] = None
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
    # Initialize result structure
    result = {
        'total_usda_groups': len(usda_mapping),
        'analyzed_usda_groups': 0,
        'fully_clustered_groups': 0,
        'partially_clustered_groups': 0,
        'not_clustered_groups': 0,
        'group_details': [],
        'summary': {}
    }
    
    # Get all product codes in clusters
    clustered_products = set()
    product_to_cluster = {}
    for cluster_id, product_codes in clusters.items():
        for code in product_codes:
            clustered_products.add(code)
            product_to_cluster[code] = cluster_id
    
    # Create a set of all product codes from the prepared data
    all_products = set(prepared_df['product_code'].astype(str).tolist())
    
    # Get a set of products that have category information
    products_with_category = set()
    category_missing_counts = 0
    if category_products:
        for category, products in category_products.items():
            products_with_category.update(products)
    
    # Analysis tracking variables
    reasons_not_clustered = defaultdict(int)
    cluster_distribution = defaultdict(list)  # Map USDA group to list of clusters its products are in
    
    # Analyze each USDA group
    for std_name, product_codes in usda_mapping.items():
        # Filter out product codes not in our prepared data
        valid_codes = [code for code in product_codes if code in all_products]
        
        if not valid_codes:
            # Skip groups with no valid products
            continue
            
        result['analyzed_usda_groups'] += 1
        group_result = {
            'name': std_name,
            'total_products': len(valid_codes),
            'clustered_products': 0,
            'clusters': set(),
            'coverage': 0.0,
            'products': [],
            'status': 'not_clustered'
        }
        
        # Track products and their clustering status
        clusters_for_group = set()
        for code in valid_codes:
            product_info = {
                'code': code,
                'clustered': code in clustered_products,
                'cluster_id': product_to_cluster.get(code, 'N/A'),
                'has_category': code in products_with_category
            }
            
            # Find the category if available
            if category_products:
                product_info['category'] = find_product_category(code, category_products)
            else:
                product_info['category'] = None
                
            if product_info['clustered']:
                group_result['clustered_products'] += 1
                clusters_for_group.add(product_info['cluster_id'])
                
                # Add to cluster distribution
                cluster_distribution[std_name].append(product_info['cluster_id'])
            else:
                # Track reasons for not being clustered
                if not product_info['has_category']:
                    reasons_not_clustered['no_category'] += 1
                    product_info['reason'] = 'no_category'
                else:
                    reasons_not_clustered['filtered_out'] += 1
                    product_info['reason'] = 'filtered_out'
            
            group_result['products'].append(product_info)
        
        # Calculate coverage
        if valid_codes:
            group_result['coverage'] = group_result['clustered_products'] / len(valid_codes)
        
        # Store clusters that products in this group appear in
        group_result['clusters'] = list(clusters_for_group)
        group_result['distinct_clusters'] = len(clusters_for_group)
        
        # Determine status
        if group_result['clustered_products'] == 0:
            group_result['status'] = 'not_clustered'
            result['not_clustered_groups'] += 1
        elif group_result['clustered_products'] < len(valid_codes):
            group_result['status'] = 'partially_clustered'
            result['partially_clustered_groups'] += 1
        else:
            if len(clusters_for_group) == 1:
                group_result['status'] = 'fully_clustered_together'
                result['fully_clustered_groups'] += 1
            else:
                group_result['status'] = 'fully_clustered_split'
                result['partially_clustered_groups'] += 1
        
        result['group_details'].append(group_result)
    
    # Compute clustering quality
    cluster_quality = {}
    for std_name, cluster_ids in cluster_distribution.items():
        if not cluster_ids:
            continue
        
        # Count occurrences of each cluster
        cluster_counts = defaultdict(int)
        for c_id in cluster_ids:
            cluster_counts[c_id] += 1
            
        # Find the most common cluster
        most_common = max(cluster_counts.items(), key=lambda x: x[1])
        total = len(cluster_ids)
        
        # Calculate precision - what % of the products in this group are in the most common cluster
        precision = most_common[1] / total if total > 0 else 0
        
        cluster_quality[std_name] = {
            'primary_cluster': most_common[0],
            'precision': precision,
            'split_across': len(cluster_counts)
        }
    
    # Calculate summary statistics
    if result['analyzed_usda_groups'] > 0:
        result['summary'] = {
            'analyzed_groups': result['analyzed_usda_groups'],
            'fully_clustered_percent': (result['fully_clustered_groups'] / result['analyzed_usda_groups']) * 100,
            'partially_clustered_percent': (result['partially_clustered_groups'] / result['analyzed_usda_groups']) * 100,
            'not_clustered_percent': (result['not_clustered_groups'] / result['analyzed_usda_groups']) * 100,
            'reasons_not_clustered': dict(reasons_not_clustered),
            'cluster_quality': cluster_quality
        }
    
    return result

def generate_usda_mapping_report(analysis_result: Dict[str, Any]) -> str:
    """
    Generate a detailed report on USDA mapping alignment.
    
    Args:
        analysis_result: Results from analyze_usda_grouping_alignment
        
    Returns:
        Markdown-formatted report
    """
    if not analysis_result or not analysis_result.get('analyzed_usda_groups', 0):
        return "# USDA Mapping Analysis\n\nNo USDA mapping data available for analysis."
    
    summary = analysis_result['summary']
    
    report = [
        "# USDA Mapping Analysis",
        "\n## Summary",
        f"\nTotal product groups analyzed: {analysis_result['analyzed_usda_groups']}",
        f"- **Fully clustered together**: {analysis_result['fully_clustered_groups']} groups ({summary.get('fully_clustered_percent', 0):.1f}%)",
        f"- **Partially clustered or split**: {analysis_result['partially_clustered_groups']} groups ({summary.get('partially_clustered_percent', 0):.1f}%)",
        f"- **Not clustered**: {analysis_result['not_clustered_groups']} groups ({summary.get('not_clustered_percent', 0):.1f}%)",
        
        "\n## Why Products Aren't Properly Clustered",
        "\nTop reasons products from the same group weren't clustered together:"
    ]
    
    # Format reasons not clustered
    reasons = summary.get('reasons_not_clustered', {})
    if reasons:
        report.append("\n| Reason | Count |")
        report.append("| ------ | ----- |")
        for reason, count in reasons.items():
            report.append(f"| {reason.replace('_', ' ').title()} | {count} |")
    
    # Add group details for sampling
    report.append("\n## Sample Group Analysis")
    
    # Get a sample of groups for each status
    fully_clustered = [g for g in analysis_result['group_details'] if g['status'] == 'fully_clustered_together']
    split_clustered = [g for g in analysis_result['group_details'] if g['status'] == 'fully_clustered_split']
    partially_clustered = [g for g in analysis_result['group_details'] if g['status'] == 'partially_clustered']
    not_clustered = [g for g in analysis_result['group_details'] if g['status'] == 'not_clustered']
    
    # Sample groups (up to 3 of each type)
    import random
    samples = []
    for group_list, status in [
        (fully_clustered, "Fully Clustered Together"),
        (split_clustered, "Fully Clustered But Split"),
        (partially_clustered, "Partially Clustered"),
        (not_clustered, "Not Clustered")
    ]:
        if group_list:
            # Take up to 3 samples, but if fewer are available, take all
            sample_count = min(3, len(group_list))
            sampled = random.sample(group_list, sample_count)
            
            report.append(f"\n### {status} - {sample_count} Examples")
            
            for i, group in enumerate(sampled, 1):
                report.append(f"\n#### Example {i}: {group['name']}")
                report.append(f"- Total Products: {group['total_products']}")
                report.append(f"- Clustered Products: {group['clustered_products']} ({group['coverage']*100:.1f}%)")
                
                if group['clusters']:
                    report.append(f"- Found In Clusters: {', '.join(sorted(group['clusters']))}")
                    report.append(f"- Number of Distinct Clusters: {group['distinct_clusters']}")
                
                # Table of products
                report.append("\nProduct Details:")
                report.append("| Product Code | Clustered | Cluster ID | Has Category | Category |")
                report.append("| ------------ | --------- | ---------- | ------------ | -------- |")
                
                # Show up to 10 products per group
                for product in group['products'][:10]:
                    clustered = "✅" if product['clustered'] else "❌"
                    has_category = "✅" if product['has_category'] else "❌"
                    category = product['category'] if product['category'] else "N/A"
                    cluster_id = product['cluster_id'] if product['clustered'] else "N/A"
                    
                    report.append(f"| {product['code']} | {clustered} | {cluster_id} | {has_category} | {category} |")
                
                if len(group['products']) > 10:
                    report.append("| ... | ... | ... | ... | ... |")
    
    return "\n".join(report)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze USDA mapping alignment with clusters")
    parser.add_argument("--mapping", required=True, help="Path to USDA mapping file")
    parser.add_argument("--clusters", required=True, help="Path to clusters JSON")
    parser.add_argument("--products", required=True, help="Path to prepared products CSV")
    parser.add_argument("--output", required=True, help="Path to output report")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.clusters, 'r') as f:
        clusters = json.load(f)
    
    usda_mapping = load_usda_mapping(args.mapping)
    prepared_df = pd.read_csv(args.products)
    
    # Run analysis
    result = analyze_usda_grouping_alignment(clusters, usda_mapping, prepared_df)
    
    # Generate report
    report = generate_usda_mapping_report(result)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"USDA mapping analysis complete. Report saved to {args.output}")
