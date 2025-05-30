#!/usr/bin/env python3
"""
exact_match_analyzer.py - Analyze product clusters to identify exact matches

This script analyzes the refined clusters from the product clustering algorithm
and identifies exact matches within each cluster based on specific criteria.
An exact match is defined as products that have:
1. Almost exactly the same name
2. Same size (if provided)
3. Same brand (if provided)
4. From the same company

The script outputs a CSV file with exact match groups, including:
1. Match ID
2. Match group name
3. SKU ID
4. SKU name
5. Company
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Set, Tuple, Any
from difflib import SequenceMatcher
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_product_name(name: str) -> str:
    """
    Clean and normalize product name for comparison.
    
    Args:
        name: Raw product name
        
    Returns:
        Cleaned product name
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove special characters
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Remove common filler words
    filler_words = [
        'inc', 'llc', 'co', 'company', 'corporation', 'corp', 
        'products', 'product', 'brand', 'foods', 'food'
    ]
    for word in filler_words:
        name = re.sub(r'\b' + word + r'\b', '', name)
    
    return name.strip()

def extract_size(name: str) -> Tuple[str, str]:
    """
    Extract size information from product name.
    
    Args:
        name: Product name
        
    Returns:
        Tuple of (name without size, size)
    """
    if pd.isna(name) or not isinstance(name, str):
        return "", ""
    
    # Common size patterns like "12 oz", "1.5 lb", "750 ml", "5x5"
    size_patterns = [
        # Weight patterns
        r'(\d+(\.\d+)?\s*(oz|ounce|pound|lb|kg|g|gram)s?)',
        # Volume patterns
        r'(\d+(\.\d+)?\s*(ml|l|liter|gal|gallon|qt|quart|pt|pint|fl\.?\s*oz)s?)',
        # Dimension patterns
        r'(\d+(\.\d+)?\s*[xX]\s*\d+(\.\d+)?(\s*[xX]\s*\d+(\.\d+)?)?)',
        # Count patterns
        r'(\d+\s*(ct|count|pc|piece|pack|pk)s?)',
        # Standard packaging sizes
        r'(\d+\s*(case|box|bag|bottle|jar|can|container)s?)'
    ]
    
    # Search for size patterns
    size = ""
    name_without_size = name
    
    for pattern in size_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            size = match.group(0)
            name_without_size = name.replace(size, "").strip()
            break
    
    return name_without_size, size

def extract_brand(name: str) -> Tuple[str, str]:
    """
    Extract brand information from product name if present.
    
    Args:
        name: Product name
        
    Returns:
        Tuple of (name without brand, brand)
    """
    if pd.isna(name) or not isinstance(name, str):
        return "", ""
    
    # Simple approach: check if there's a possessive name (ending with 's)
    possessive_match = re.search(r"([A-Z][a-zA-Z]+)'s", name)
    if possessive_match:
        brand = possessive_match.group(1)
        return name.replace(possessive_match.group(0), "").strip(), brand
    
    # Check for names that appear at the beginning with capitalization
    brand_match = re.match(r"^([A-Z][a-zA-Z]+)", name)
    if brand_match:
        potential_brand = brand_match.group(1)
        # Don't extract common words as brands
        common_words = ["the", "fresh", "frozen", "organic", "whole", "sliced"]
        if potential_brand.lower() not in common_words and len(potential_brand) > 2:
            return name[len(potential_brand):].strip(), potential_brand
    
    return name, ""

def calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two product names.
    
    Args:
        name1: First product name
        name2: Second product name
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not name1 or not name2:
        return 0.0
    
    return SequenceMatcher(None, name1, name2).ratio()

def load_transaction_data(filepath: str) -> pd.DataFrame:
    """
    Load transaction data from Excel file.
    
    Args:
        filepath: Path to Transaction_Report_Actual.xlsx
        
    Returns:
        DataFrame with transaction data
    """
    try:
        # Try different engines to ensure compatibility
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
        except Exception as e:
            logger.warning(f"Error using openpyxl: {e}. Trying with xlrd...")
            try:
                df = pd.read_excel(filepath, engine='xlrd')
            except Exception as e2:
                logger.warning(f"Error using xlrd: {e2}. Trying with odf...")
                df = pd.read_excel(filepath, engine='odf')
        
        logger.info(f"Successfully loaded transaction data with {len(df)} rows")
        
        # Check if required columns exist
        required_cols = ['ProductCode', 'ProductDescription']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in transaction data. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Add company information if possible
        if 'Company' not in df.columns:
            # Try to infer company from filename or other sources
            filename = os.path.basename(filepath)
            if "Anmar" in filename:
                df['Company'] = "Anmar"
            elif "Fulton" in filename:
                df['Company'] = "Fulton"
            elif "Moesle" in filename:
                df['Company'] = "Moesle"
            elif "Pritzlaff" in filename:
                df['Company'] = "Pritzlaff"
            elif "Queen" in filename:
                df['Company'] = "Queen"
            else:
                # If we can't infer, use "Unknown"
                df['Company'] = "Unknown"
        
        # Ensure all ProductCode values are strings
        df['ProductCode'] = df['ProductCode'].astype(str)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading transaction data: {e}")
        return pd.DataFrame()

def load_clusters(filepath: str) -> Dict[str, List[str]]:
    """
    Load cluster data from JSON file.
    
    Args:
        filepath: Path to refined_clusters.json
        
    Returns:
        Dictionary mapping cluster IDs to lists of product IDs
    """
    try:
        with open(filepath, 'r') as f:
            clusters = json.load(f)
        
        logger.info(f"Successfully loaded {len(clusters)} clusters")
        return clusters
    
    except Exception as e:
        logger.error(f"Error loading clusters: {e}")
        return {}

def find_exact_matches(
    cluster_id: str,
    product_ids: List[str],
    transaction_df: pd.DataFrame,
    name_similarity_threshold: float = 0.85,
    size_match_required: bool = True,
    brand_match_required: bool = True,
    company_match_required: bool = True
) -> List[Dict[str, Any]]:
    """
    Find exact matches within a cluster.
    
    Args:
        cluster_id: ID of the current cluster
        product_ids: List of product IDs in the cluster
        transaction_df: DataFrame with transaction data
        name_similarity_threshold: Minimum similarity score to consider names as matching
        size_match_required: Whether size matching is required for exact matches
        brand_match_required: Whether brand matching is required for exact matches
        company_match_required: Whether company matching is required for exact matches
        
    Returns:
        List of exact match groups, each as a dictionary
    """
    # Filter transaction data to only include products in this cluster
    cluster_products = transaction_df[transaction_df['ProductCode'].isin(product_ids)].copy()
    
    if len(cluster_products) == 0:
        logger.warning(f"No transaction data found for cluster {cluster_id}")
        return []
    
    # Preprocess product names and extract features
    cluster_products['cleaned_name'] = cluster_products['ProductDescription'].apply(clean_product_name)
    cluster_products['name_without_size'], cluster_products['size'] = zip(
        *cluster_products['ProductDescription'].apply(extract_size)
    )
    cluster_products['name_without_brand'], cluster_products['brand'] = zip(
        *cluster_products['name_without_size'].apply(extract_brand)
    )
    
    # Create a comparison matrix for all products in the cluster
    product_indices = list(range(len(cluster_products)))
    match_groups = []
    processed_indices = set()
    
    # For each product, find its exact matches
    for i in product_indices:
        if i in processed_indices:
            continue
        
        current_product = cluster_products.iloc[i]
        match_group = [i]
        
        # Compare with all other products
        for j in product_indices:
            if j == i or j in processed_indices:
                continue
            
            comparison_product = cluster_products.iloc[j]
            
            # Check name similarity
            name_similarity = calculate_name_similarity(
                current_product['cleaned_name'],
                comparison_product['cleaned_name']
            )
            
            # Check size match if required
            size_match = True
            if size_match_required and current_product['size'] and comparison_product['size']:
                size_match = current_product['size'] == comparison_product['size']
            
            # Check brand match if required
            brand_match = True
            if brand_match_required and current_product['brand'] and comparison_product['brand']:
                brand_match = current_product['brand'] == comparison_product['brand']
            
            # Check company match if required
            company_match = True
            if company_match_required:
                company_match = current_product['Company'] == comparison_product['Company']
            
            # Determine if this is an exact match
            is_match = (
                name_similarity >= name_similarity_threshold and
                size_match and
                brand_match and
                company_match
            )
            
            if is_match:
                match_group.append(j)
        
        # If we found a group of matches (at least 2 products), save it
        if len(match_group) >= 2:
            match_products = cluster_products.iloc[match_group].copy()
            
            # Generate a good name for the match group
            # Use the most common elements in the names
            name_elements = defaultdict(int)
            for name in match_products['cleaned_name']:
                for word in name.split():
                    if len(word) > 2:  # Ignore short words
                        name_elements[word] += 1
            
            # Get the most common elements
            common_elements = sorted(
                name_elements.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            match_name = " ".join([elem[0] for elem in common_elements]).title()
            
            # Add to match groups
            match_groups.append({
                "cluster_id": cluster_id,
                "match_name": match_name,
                "product_indices": match_group,
                "products": match_products
            })
            
            # Mark these products as processed
            processed_indices.update(match_group)
    
    return match_groups

def generate_match_output(
    all_match_groups: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Generate CSV output with exact matches.
    
    Args:
        all_match_groups: List of all match groups across all clusters
        output_path: Path to write the output CSV
    """
    # Prepare data for output
    output_rows = []
    match_id = 1
    
    for match_group in all_match_groups:
        match_name = match_group["match_name"]
        
        # Add each product in the match group
        for _, product_row in match_group["products"].iterrows():
            output_rows.append({
                "Match_ID": f"MATCH_{match_id:04d}",
                "Match_Group_Name": match_name,
                "SKU_ID": product_row["ProductCode"],
                "SKU_Name": product_row["ProductDescription"],
                "Company": product_row["Company"]
            })
        
        match_id += 1
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_rows)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(output_rows)} exact match products to {output_path}")
    
    # Print summary statistics
    logger.info(f"Found {match_id - 1} exact match groups across all clusters")
    logger.info(f"Average group size: {len(output_rows) / (match_id - 1):.2f} products per group")

def main():
    parser = argparse.ArgumentParser(description='Analyze product clusters to find exact matches')
    
    parser.add_argument('--clusters', type=str, 
                        default='/Users/eshantarneja/Documents/Git/VectorDB/product_clustering/data/refined_clustering/refined_clusters.json',
                        help='Path to refined_clusters.json')
    
    parser.add_argument('--transaction_data', type=str, 
                        default='/Users/eshantarneja/Documents/Git/VectorDB/Source_data/Actuals/Transaction_Report_Actual.xlsx',
                        help='Path to transaction data Excel file')
    
    parser.add_argument('--output', type=str, 
                        default='/Users/eshantarneja/Documents/Git/VectorDB/product_clustering/Analysis_Scripts/exact_matches.csv',
                        help='Path to output CSV file')
    
    parser.add_argument('--name_similarity', type=float, default=0.85,
                        help='Threshold for name similarity (0.0-1.0)')
    
    parser.add_argument('--ignore_size', action='store_true',
                        help='Ignore size differences when finding exact matches')
    
    parser.add_argument('--ignore_brand', action='store_true',
                        help='Ignore brand differences when finding exact matches')
    
    parser.add_argument('--ignore_company', action='store_true',
                        help='Ignore company differences when finding exact matches')
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading cluster data...")
    clusters = load_clusters(args.clusters)
    
    logger.info("Loading transaction data...")
    transaction_df = load_transaction_data(args.transaction_data)
    
    if len(clusters) == 0 or len(transaction_df) == 0:
        logger.error("Failed to load required data. Exiting.")
        return
    
    # Process each cluster
    logger.info("Processing clusters to find exact matches...")
    all_match_groups = []
    
    for cluster_id, product_ids in clusters.items():
        logger.info(f"Processing cluster {cluster_id} with {len(product_ids)} products")
        
        match_groups = find_exact_matches(
            cluster_id,
            product_ids,
            transaction_df,
            name_similarity_threshold=args.name_similarity,
            size_match_required=not args.ignore_size,
            brand_match_required=not args.ignore_brand,
            company_match_required=not args.ignore_company
        )
        
        all_match_groups.extend(match_groups)
        
        logger.info(f"Found {len(match_groups)} exact match groups in cluster {cluster_id}")
    
    # Generate output
    logger.info(f"Processing complete. Found {len(all_match_groups)} total exact match groups.")
    generate_match_output(all_match_groups, args.output)

if __name__ == "__main__":
    main()
