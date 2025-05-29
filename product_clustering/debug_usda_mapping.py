#!/usr/bin/env python3
"""
Debug USDA Mapping

This script helps diagnose issues with USDA mapping by inspecting product codes,
their matches, and providing detailed debugging information.
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

def load_usda_mapping(mapping_path: str) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Load the USDA to product code mapping data.
    
    Args:
        mapping_path: Path to the mapping Excel file
        
    Returns:
        Tuple of (mapping dictionary, raw DataFrame)
    """
    try:
        # Load the Excel file
        df = pd.read_excel(mapping_path)
        
        # Create mapping from standardized name to product codes
        mapping = {}
        for _, row in df.iterrows():
            std_name = row['Standardized Product Name']
            # Convert product codes to strings and split by semicolon
            product_codes = [str(code).strip() for code in str(row['Product Codes']).split(';')]
            # Filter out any empty strings
            product_codes = [code for code in product_codes if code]
            
            if std_name and product_codes:
                mapping[std_name] = product_codes
                
        return mapping, df
    
    except Exception as e:
        print(f"Error loading USDA mapping data: {e}")
        return {}, pd.DataFrame()

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

def validate_product_codes(
    usda_mapping: Dict[str, List[str]], 
    products_df: pd.DataFrame,
    detailed: bool = False
) -> Dict[str, Any]:
    """
    Validate product codes in the USDA mapping against the products dataframe.
    
    Args:
        usda_mapping: Dictionary mapping standardized names to product codes
        products_df: DataFrame containing product information
        detailed: Whether to include detailed information in the result
        
    Returns:
        Dictionary containing validation results
    """
    if 'product_code' not in products_df.columns:
        raise ValueError("products_df must contain a 'product_code' column")
    
    # Convert all product codes to strings for consistent comparison
    products_df['product_code'] = products_df['product_code'].astype(str)
    
    # Get set of all product codes in products_df
    all_product_codes = set(products_df['product_code'].tolist())
    
    # Validate each USDA mapping entry
    validation_results = {
        'summary': {
            'total_usda_groups': len(usda_mapping),
            'total_usda_product_codes': 0,
            'matched_product_codes': 0,
            'unmatched_product_codes': 0,
            'groups_with_matches': 0,
            'groups_without_matches': 0
        },
        'group_results': []
    }
    
    total_usda_product_codes = 0
    matched_product_codes = 0
    unmatched_product_codes = 0
    
    for std_name, product_codes in usda_mapping.items():
        total_usda_product_codes += len(product_codes)
        
        # Check which product codes match with products_df
        matched_codes = [code for code in product_codes if code in all_product_codes]
        unmatched_codes = [code for code in product_codes if code not in all_product_codes]
        
        matched_product_codes += len(matched_codes)
        unmatched_product_codes += len(unmatched_codes)
        
        # Record results for this group
        group_result = {
            'standardized_name': std_name,
            'total_product_codes': len(product_codes),
            'matched_codes': len(matched_codes),
            'unmatched_codes': len(unmatched_codes),
            'match_ratio': len(matched_codes) / len(product_codes) if product_codes else 0
        }
        
        if detailed:
            # Include the actual codes for detailed analysis
            group_result['product_codes'] = product_codes
            group_result['matched_product_codes'] = matched_codes
            group_result['unmatched_product_codes'] = unmatched_codes
            
            # Include product descriptions for matched codes
            if matched_codes:
                matched_descriptions = {}
                for code in matched_codes:
                    description = products_df.loc[products_df['product_code'] == code, 'product_description'].iloc[0] \
                                  if not products_df.loc[products_df['product_code'] == code].empty else "N/A"
                    matched_descriptions[code] = description
                group_result['matched_descriptions'] = matched_descriptions
        
        validation_results['group_results'].append(group_result)
    
    # Update summary statistics
    validation_results['summary']['total_usda_product_codes'] = total_usda_product_codes
    validation_results['summary']['matched_product_codes'] = matched_product_codes
    validation_results['summary']['unmatched_product_codes'] = unmatched_product_codes
    validation_results['summary']['groups_with_matches'] = sum(1 for result in validation_results['group_results'] if result['matched_codes'] > 0)
    validation_results['summary']['groups_without_matches'] = sum(1 for result in validation_results['group_results'] if result['matched_codes'] == 0)
    
    # Calculate percentages
    if total_usda_product_codes > 0:
        validation_results['summary']['matched_product_codes_percent'] = matched_product_codes / total_usda_product_codes * 100
        validation_results['summary']['unmatched_product_codes_percent'] = unmatched_product_codes / total_usda_product_codes * 100
    
    if validation_results['summary']['total_usda_groups'] > 0:
        validation_results['summary']['groups_with_matches_percent'] = validation_results['summary']['groups_with_matches'] / validation_results['summary']['total_usda_groups'] * 100
        validation_results['summary']['groups_without_matches_percent'] = validation_results['summary']['groups_without_matches'] / validation_results['summary']['total_usda_groups'] * 100
    
    return validation_results

def debug_usda_mapping_issues(
    mapping_path: str,
    products_path: str,
    output_dir: str,
    detailed: bool = True
) -> str:
    """
    Debug issues with USDA mapping.
    
    Args:
        mapping_path: Path to the USDA mapping Excel file
        products_path: Path to the prepared products CSV file
        output_dir: Directory to save debug results
        detailed: Whether to include detailed information in the results
        
    Returns:
        Path to the debug report file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading USDA mapping from {mapping_path}...")
    usda_mapping, mapping_df = load_usda_mapping(mapping_path)
    print(f"Loaded {len(usda_mapping)} standardized product groupings")
    
    print(f"Loading products data from {products_path}...")
    products_df = load_products_data(products_path)
    print(f"Loaded {len(products_df)} products")
    
    print("Validating product codes...")
    validation_results = validate_product_codes(usda_mapping, products_df, detailed)
    
    # Save validation results to JSON
    results_path = os.path.join(output_dir, "usda_mapping_validation.json")
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Generate report
    report = f"""# USDA Mapping Debug Report

## Summary

- **Total USDA Groups**: {validation_results['summary']['total_usda_groups']}
- **Total USDA Product Codes**: {validation_results['summary']['total_usda_product_codes']}
- **Matched Product Codes**: {validation_results['summary']['matched_product_codes']} ({validation_results['summary'].get('matched_product_codes_percent', 0):.1f}%)
- **Unmatched Product Codes**: {validation_results['summary']['unmatched_product_codes']} ({validation_results['summary'].get('unmatched_product_codes_percent', 0):.1f}%)
- **Groups with Matches**: {validation_results['summary']['groups_with_matches']} ({validation_results['summary'].get('groups_with_matches_percent', 0):.1f}%)
- **Groups without Matches**: {validation_results['summary']['groups_without_matches']} ({validation_results['summary'].get('groups_without_matches_percent', 0):.1f}%)

## Issues and Recommendations

"""
    
    # Analyze issues and provide recommendations
    if validation_results['summary']['matched_product_codes'] == 0:
        report += """
### Critical Issue: No Product Codes Match

None of the product codes in the USDA mapping file match with the products in your dataset. This could be due to:

1. **Format Mismatch**: The product code format in the mapping file doesn't match the format in your products data.
2. **Different Product Sets**: The mapping file might refer to a completely different set of products.
3. **Data Preparation Issue**: There might be an issue in how the product codes are prepared or normalized.

**Recommendation**: 
- Use the `create_realistic_mapping.py` script to generate a mapping file using actual product codes from your dataset.
- Check the format of product codes in both the mapping file and your products data.
- Ensure the mapping file contains relevant product codes for your specific dataset.
"""
    elif validation_results['summary']['matched_product_codes_percent'] < 25:
        report += f"""
### Issue: Low Match Rate ({validation_results['summary'].get('matched_product_codes_percent', 0):.1f}%)

Only a small percentage of product codes in the USDA mapping file match with products in your dataset. This suggests:

1. **Partial Data Overlap**: The mapping file might be from a larger or different dataset with partial overlap.
2. **Code Format Inconsistencies**: Some product codes might have format differences (leading zeros, hyphens, etc.).
3. **Outdated Mapping**: The mapping file might include obsolete product codes.

**Recommendation**:
- Use the `create_realistic_mapping.py` script to generate a mapping file with better coverage.
- Consider normalizing product codes consistently across both datasets.
- If certain categories have better match rates, focus your analysis on those categories.
"""
    
    # List top groups with and without matches
    report += "\n## Groups with Highest Match Rates\n\n"
    
    # Sort groups by match ratio (descending)
    sorted_groups = sorted(validation_results['group_results'], 
                           key=lambda x: x['match_ratio'], 
                           reverse=True)
    
    # Show top 10 groups with matches
    groups_with_matches = [g for g in sorted_groups if g['matched_codes'] > 0]
    if groups_with_matches:
        for i, group in enumerate(groups_with_matches[:10]):
            report += f"{i+1}. **{group['standardized_name']}**: {group['matched_codes']}/{group['total_product_codes']} codes matched ({group['match_ratio']*100:.1f}%)\n"
    else:
        report += "No groups with matches found.\n"
    
    # Show top 10 groups without matches (if detailed is True)
    if detailed:
        report += "\n## Groups with No Matches\n\n"
        groups_without_matches = [g for g in validation_results['group_results'] if g['matched_codes'] == 0]
        if groups_without_matches:
            for i, group in enumerate(groups_without_matches[:10]):
                report += f"{i+1}. **{group['standardized_name']}**: 0/{group['total_product_codes']} codes matched\n"
        else:
            report += "All groups have at least one match.\n"
    
    # Save report
    report_path = os.path.join(output_dir, "usda_mapping_debug_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Debug complete. Report saved to {report_path}")
    print(f"Raw validation results saved to {results_path}")
    
    # Print key findings
    print("\nKey Findings:")
    print(f"- Total USDA Groups: {validation_results['summary']['total_usda_groups']}")
    print(f"- Matched Product Codes: {validation_results['summary']['matched_product_codes']} ({validation_results['summary'].get('matched_product_codes_percent', 0):.1f}%)")
    print(f"- Groups with Matches: {validation_results['summary']['groups_with_matches']} ({validation_results['summary'].get('groups_with_matches_percent', 0):.1f}%)")
    
    return report_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug USDA mapping issues")
    
    # Data paths
    parser.add_argument("--mapping_path", type=str, 
                        default=os.path.join("data", "CorrectMapping", "product_mapping_semantic.xlsx"),
                        help="Path to USDA mapping file")
    parser.add_argument("--products_path", type=str, help="Path to prepared products CSV file")
    parser.add_argument("--output_dir", type=str, help="Directory to save debug results")
    parser.add_argument("--detailed", action="store_true", help="Include detailed information in results")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not args.mapping_path:
        args.mapping_path = os.path.join(project_root, "data", "CorrectMapping", "product_mapping_semantic.xlsx")
    
    if not args.products_path:
        args.products_path = os.path.join(project_root, "product_clustering", "data", "prepared_products.csv")
    
    if not args.output_dir:
        args.output_dir = os.path.join(project_root, "product_clustering", "data", "analysis")
    
    debug_usda_mapping_issues(
        mapping_path=args.mapping_path,
        products_path=args.products_path,
        output_dir=args.output_dir,
        detailed=args.detailed
    )

if __name__ == "__main__":
    main()
