#!/usr/bin/env python3
"""
cluster_llm_matcher.py - Identify exact product matches within clusters using LLM

This script processes product clusters by sending all product descriptions within
a cluster to an LLM (GPT-3.5-turbo) and having it identify exact matches based on
product name, brand, size, and count. This approach reduces complexity by analyzing
entire clusters at once rather than making pairwise comparisons.

The script limits processing to a sample of clusters to keep costs low while testing.
"""

import os
import json
import time
import logging
import argparse
import re
import random
from typing import List, Dict, Tuple
from collections import defaultdict

import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

# Define common stopwords to filter out from match group names
COMMON_STOPWORDS = {
    'the', 'and', 'for', 'with', 'has', 'its', 'are', 'not', 'this', 'that',
    'was', 'from', 'but', 'have', 'all', 'they', 'been', 'were', 'when', 'who',
    'will', 'more', 'out', 'use', 'any', 'than', 'can', 'into', 'some', 'other',
    'which', 'their', 'time', 'only', 'them', 'would', 'about', 'there', 'what',
    'our', 'your', 'also', 'how', 'then', 'first', 'just', 'should', 'these',
    'two', 'make', 'over', 'could', 'may', 'such', 'used', 'being', 'must',
    'very', 'new', 'after', 'most', 'before', 'through', 'where', 'each', 'well',
    'did', 'off', 'like', 'had', 'get', 'said', 'back', 'way', 'now', 'even'
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (for API keys)
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o-mini"  # More efficient GPT-4 variant for better reasoning at lower cost
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SAMPLE_SIZE = 20  # Number of clusters to sample

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

def sample_clusters(clusters: Dict[str, List[str]], n: int) -> Dict[str, List[str]]:
    """
    Sample n clusters for analysis, using a balanced approach that includes small, medium and large clusters.
    
    Args:
        clusters: Dictionary mapping cluster IDs to product IDs
        n: Number of clusters to sample
        
    Returns:
        Dictionary with sampled clusters
    """
    # Categorize clusters by size
    small_clusters = {}
    medium_clusters = {}
    large_clusters = {}
    
    for cluster_id, products in clusters.items():
        size = len(products)
        if 3 <= size <= 6:  # Small clusters
            small_clusters[cluster_id] = products
        elif 7 <= size <= 15:  # Medium clusters
            medium_clusters[cluster_id] = products
        elif size > 15:  # Large clusters
            large_clusters[cluster_id] = products
    
    result = {}
    
    # Allocate sampling proportionally
    # 50% small, 30% medium, 20% large if possible
    small_target = min(int(n * 0.5), len(small_clusters))
    medium_target = min(int(n * 0.3), len(medium_clusters))
    large_target = min(n - small_target - medium_target, len(large_clusters))
    
    # Adjust if we can't meet targets
    remaining = n - small_target - medium_target - large_target
    if remaining > 0 and len(medium_clusters) > medium_target:
        medium_target += min(remaining, len(medium_clusters) - medium_target)
        remaining = n - small_target - medium_target - large_target
    
    if remaining > 0 and len(small_clusters) > small_target:
        small_target += min(remaining, len(small_clusters) - small_target)
        remaining = n - small_target - medium_target - large_target
    
    if remaining > 0 and len(large_clusters) > large_target:
        large_target += min(remaining, len(large_clusters) - large_target)
    
    # Sample from each category
    if small_target > 0:
        small_keys = list(small_clusters.keys())
        for cluster_id in random.sample(small_keys, small_target):
            result[cluster_id] = small_clusters[cluster_id]
    
    if medium_target > 0:
        medium_keys = list(medium_clusters.keys())
        for cluster_id in random.sample(medium_keys, medium_target):
            result[cluster_id] = medium_clusters[cluster_id]
    
    if large_target > 0:
        large_keys = list(large_clusters.keys())
        for cluster_id in random.sample(large_keys, large_target):
            result[cluster_id] = large_clusters[cluster_id]
    
    # If we still don't have enough, take from any remaining clusters
    if len(result) < n:
        all_remaining = list(set(clusters.keys()) - set(result.keys()))
        if all_remaining:
            additional = random.sample(all_remaining, min(n - len(result), len(all_remaining)))
            for cluster_id in additional:
                result[cluster_id] = clusters[cluster_id]
    
    logger.info(f"Selected {len(result)} clusters for analysis (approx. 50% small, 30% medium, 20% large)")
    return result

def call_llm_for_cluster(
    cluster_id: str, 
    product_descriptions: List[Tuple[str, str, str]], 
    model: str = DEFAULT_MODEL
) -> List[List[int]]:
    """
    Call LLM to identify exact matches within a cluster.
    
    Args:
        cluster_id: ID of the cluster
        product_descriptions: List of tuples (product_id, description, company)
        model: LLM model to use
        
    Returns:
        List of lists, where each inner list contains indices of products that are exact matches
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set. Cannot use LLM functionality.")
        return []
    
    # Debug: Show we're processing this specific cluster
    print(f"\n{'='*80}")
    print(f"PROCESSING CLUSTER: {cluster_id} with {len(product_descriptions)} products")
    print(f"{'='*80}\n")
    
    # Skip tiny clusters (1-2 products) as they're unlikely to have exact matches
    if len(product_descriptions) <= 2:
        logger.info(f"Skipping cluster {cluster_id} with only {len(product_descriptions)} products")
        return []
    
    # Construct the prompt with appropriately strict matching criteria but clear examples
    system_message = """You work for a food distributor that sells many SKUs of products. You are a product matching expert with high standards. Your task is to identify which SKUs in a cluster are exact matches of each other based on their descriptions/names - these are skus that are likely duplicates and can be consolidated. They may have the exact same name or a small variation of eachother. I will give you a cluster of SKUs (with their ID # and description) that are already grouped as similar. However I want you to be even more discerning to figure out which SKU's within the cluster are exactly the same product.

IMPORTANT RULES:
1. Products with the SAME SKU NUMBER should NEVER be matched together - this is an absolute requirement. It should be two different skus that are duplicates of eachother
2. Products must be the same fundamental food item, not just similar or related products
4. Two products are considered exact matches if they share these critical attributes:
   - Same product type and category
   - Same brand (if specified)
   - Same size/count/weight information
   - Same key product specifications/attributes
   - Same flavor, cut, or variety of the same product

5. Minor differences that do NOT prevent a match:
   - CAPITALIZATION DIFFERENCES (uppercase vs lowercase)
   - Different ordering of words or formatting
   - Abbreviations vs. spelled-out terms ("lbs" vs "pounds")
   - Punctuation differences ("5-up" vs "5 up" vs "5up")
   - Minor typographical differences
   - Different SKU numbers (must have different SKUs to be a valid match)

6. Significant differences that DO prevent a match:
   - Different product types or categories
   - Different brands
   - Different sizes/weights/counts
   - Different key specifications
   - Different flavors, cuts, or varieties of the same product

EXAMPLES OF MATCHES AND NON-MATCHES:

MATCH EXAMPLE 1:
- [SKU: 40011] Beef Tenderloin PSMO 5up Prime
- [SKU: 141050] BEEF TENDERLOIN, PSMO, 5UP, PRIME
These are matches because they are the same product (Beef Tenderloin), same preparation (PSMO), same grade (Prime), and same size specification (5up), despite capitalization and punctuation differences.

MATCH EXAMPLE 2:
- [SKU: 620831] 3-1 MAC & CHEESE BRAT LINKS
- [SKU: 620841] 4-1 MAC & CHEESE BRAT LINKS
- [SKU: 620861] 5-1 MAC & CHEESE BRAT LINKS
These are matches because they are the same product (Mac & Cheese Bratwurst) with the same flavor, despite having different link counts which is a packaging difference.

MATCH EXAMPLE 3:
- [SKU: 71130] HEAD ON Shrimp 20/30 10/4#
- [SKU: 71131] HEAD ON Shrimp 30/40 10/4#
These are matches because they are the same product (Head-on Shrimp) with the same packaging (10/4#), despite having different size counts.

NON-MATCH EXAMPLE 1:
- [SKU: 17864025] CHK WING WHL-JUM AMICK
- [SKU: 17864016] CHK WING-z WHL-JUM SAN
These are NOT matches because they have different brand names (AMICK vs SAN) despite having the same product type.

NON-MATCH EXAMPLE 2:
- [SKU: 240094] PORK CHOP FRENCHED, 9OZ
- [SKU: 240089] PORK CHOP FRENCHED, 8OZ
These are NOT matches when the weight difference is important for inventory management.

Look carefully at the key product specifications before making your decision. IGNORE capitalization, punctuation, and word order differences.

Output your answer as a numbered list of match groups, with each group containing the indices of products that exactly match. Use the format:
Group 1: [0, 3, 5]
Group 2: [1, 7]
...

Only include products that have at least one exact match with a DIFFERENT SKU. Products without exact matches should not be included.
If there are no exact matches in the list (which is completely valid), respond with "No exact matches found."
"""
    
    user_message = f"Identify exact product matches in this cluster (Cluster {cluster_id}):\n\n"
    user_message += "Be appropriately discerning in your matching. It's perfectly fine if you find no exact matches in this cluster.\n\n"
    
    # Add numbered product descriptions with SKU numbers but no company info
    for i, (product_id, description, company) in enumerate(product_descriptions):
        user_message += f"{i}: [SKU: {product_id}] {description}\n"
    
    # Debug: Show what we're sending to the LLM
    print(f"\nSENDING TO LLM:\n")
    print(f"SYSTEM MESSAGE:\n{'-'*40}\n{system_message}\n{'-'*40}\n")
    print(f"USER MESSAGE:\n{'-'*40}\n{user_message}\n{'-'*40}\n")
    
    # Call OpenAI API
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.2  # Low temperature for more consistent results
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Error calling OpenAI API: {response.text}")
            return []
            
        response_json = response.json()
        llm_response = response_json["choices"][0]["message"]["content"]
        
        # Debug: Show the raw LLM response
        print(f"\nLLM RESPONSE:\n{'-'*40}\n{llm_response}\n{'-'*40}\n")
        
        # Process the LLM response to extract match groups
        match_groups = []
        
        if "No exact matches found" in llm_response:
            print("RESULT: No exact matches found in this cluster")
            return []
        
        # Extract match groups using regex
        print("\nEXTRACTING MATCH GROUPS:\n")
        for line in llm_response.split('\n'):
            match = re.search(r'Group \d+: \[(.*?)\]', line)
            if match:
                indices_str = match.group(1)
                try:
                    indices = [int(idx.strip()) for idx in indices_str.split(',')]
                    if len(indices) >= 2:  # Only consider groups with at least 2 products
                        match_groups.append(indices)
                        print(f"Found match group: {indices}")
                except ValueError:
                    print(f"WARNING: Could not parse indices from line: {line}")
                    continue  # Skip if indices can't be converted to integers
        
        if match_groups:
            print(f"\nRESULT: Found {len(match_groups)} match groups with indices: {match_groups}")
        else:
            print("\nRESULT: No valid match groups extracted")
                    
        return match_groups
        
    except Exception as e:
        logger.error(f"Error calling LLM API for cluster {cluster_id}: {e}")
        print(f"ERROR: Failed to process cluster {cluster_id}: {str(e)}")
        return []

def process_clusters_with_llm(
    clusters: Dict[str, List[str]],
    transaction_df: pd.DataFrame
) -> Dict[str, List[List[str]]]:
    """
    Process clusters using LLM to identify exact matches.
    
    Args:
        clusters: Dictionary mapping cluster IDs to product IDs
        transaction_df: DataFrame with transaction data
        
    Returns:
        Dictionary mapping cluster IDs to lists of exact match groups
    """
    print("\n" + "*"*80)
    print(f"STARTING LLM PROCESSING OF {len(clusters)} CLUSTERS")
    print("*"*80 + "\n")
    
    results = {}
    
    for cluster_id, product_ids in clusters.items():
        logger.info(f"Processing cluster {cluster_id} with {len(product_ids)} products")
        
        # Get product descriptions for this cluster
        cluster_products = transaction_df[transaction_df['ProductCode'].isin(product_ids)]
        
        if len(cluster_products) == 0:
            logger.warning(f"No transaction data found for cluster {cluster_id}")
            continue
        
        # Deduplicate products with the same SKU before sending to LLM
        # This prevents the LLM from receiving duplicate entries
        deduplicated_products = cluster_products.drop_duplicates(subset=['ProductCode'])
        logger.info(f"Deduplicated cluster {cluster_id} from {len(cluster_products)} to {len(deduplicated_products)} products")
        
        # If we have fewer than 3 unique products after deduplication, skip this cluster
        if len(deduplicated_products) < 3:
            logger.info(f"Skipping cluster {cluster_id} - fewer than 3 unique products after deduplication")
            continue
        
        # Prepare product descriptions from deduplicated data, ensuring company names are excluded
        product_descriptions = []
        for _, row in deduplicated_products.iterrows():
            # Get the product code and description
            product_code = row['ProductCode']
            description = row['ProductDescription']
            company = row['Company']
            
            # Make sure company name isn't embedded in the description
            # Remove company name from description if it appears there
            if company and company.strip() and len(company) > 1:
                # Replace company name with empty string, case insensitive
                description = re.sub(r'\b' + re.escape(company) + r'\b', '', description, flags=re.IGNORECASE)
                
            # Remove any trailing commas, dashes, or excessive whitespace
            description = re.sub(r'[,-]\s*$', '', description).strip()
            
            # Add to product descriptions
            product_descriptions.append((product_code, description, company))
        
        # Call LLM to identify exact matches
        match_groups_indices = call_llm_for_cluster(cluster_id, product_descriptions)
        
        if match_groups_indices:
            # Convert indices back to product IDs
            print(f"\nPOST-PROCESSING: Converting indices to product IDs and filtering")
            match_groups = []
            for indices in match_groups_indices:
                product_group = [product_descriptions[idx][0] for idx in indices]
                print(f"\nChecking match group with indices {indices}:")
                print(f"  Product IDs: {product_group}")
                
                # Additional check: ensure all SKUs in the group are unique
                if len(product_group) == len(set(product_group)):
                    print(f"  VALID: All SKUs are unique in this group")
                    match_groups.append(product_group)
                else:
                    duplicate_skus = [sku for sku in product_group if product_group.count(sku) > 1]
                    print(f"  INVALID: Group contains duplicate SKUs: {duplicate_skus}")
                    logger.warning(f"Skipping a match group in cluster {cluster_id} that contains duplicate SKUs")
            
            # Only add to results if there are valid match groups after filtering
            if match_groups:
                results[cluster_id] = match_groups
                print(f"\nFINAL RESULT: Found {len(match_groups)} valid match groups in cluster {cluster_id}")
                for i, group in enumerate(match_groups):
                    print(f"  Group {i+1}: {group}")
                logger.info(f"Found {len(match_groups)} exact match groups in cluster {cluster_id}")
            else:
                print(f"\nFINAL RESULT: No valid match groups in cluster {cluster_id} after filtering")
                logger.info(f"No valid exact match groups found in cluster {cluster_id} after filtering")
        else:
            logger.info(f"No exact match groups found in cluster {cluster_id}")
        
        # Add a small delay to avoid rate limits
        time.sleep(1)
    
    return results

def generate_output_csv(match_results: Dict[str, List[List[str]]], transaction_df: pd.DataFrame, output_path: str):
    """
    Generate CSV output from match results.
    
    Args:
        match_results: Dictionary mapping cluster IDs to lists of exact match groups
        transaction_df: DataFrame with transaction data
        output_path: Path to save output CSV
    """
    print("\n" + "+"*80)
    print(f"GENERATING OUTPUT CSV FROM {len(match_results)} CLUSTERS WITH MATCHES")
    print("+"*80 + "\n")
    
    output_rows = []
    match_group_count = 0
    
    # Create a deduplicated version of the transaction dataframe for output generation
    # This ensures we only include unique SKUs in our results
    deduplicated_df = transaction_df.drop_duplicates(subset=['ProductCode']).copy()
    
    for cluster_id, match_groups in match_results.items():
        for match_group in match_groups:
            # Skip if fewer than 2 products in match group
            if len(match_group) < 2:
                continue
                
            match_group_count += 1
            match_id = f"MATCH_LLM_{match_group_count:04d}"
            
            # Get product details for this match group using the deduplicated dataframe
            match_products = deduplicated_df[deduplicated_df['ProductCode'].isin(match_group)]
            
            # Verify we still have at least 2 products after deduplication
            if len(match_products) < 2:
                logger.warning(f"Skipping match group in cluster {cluster_id} - fewer than 2 unique products after final deduplication")
                continue
            
            # Generate a good name for the match group by finding common words
            descriptions = match_products['ProductDescription'].tolist()
            words_count = defaultdict(int)
            
            for desc in descriptions:
                words = re.findall(r'\b[A-Za-z]+\b', desc)
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        words_count[word.title()] += 1
            
            # Find common words (appearing in at least half of the descriptions)
            common_threshold = max(2, len(descriptions) // 2)
            common_words = [word for word, count in words_count.items() 
                           if count >= common_threshold and word.lower() not in COMMON_STOPWORDS]
            
            # Fallback if no common words found
            if not common_words:
                # Use first 2-3 words of the first description
                first_desc_words = re.findall(r'\b[A-Za-z]+\b', descriptions[0])
                common_words = [word.title() for word in first_desc_words[:3] 
                                if len(word) > 2 and word.lower() not in COMMON_STOPWORDS]
            
            match_group_name = ' '.join(common_words[:3])  # Limit to 3 words
            
            # Add a row for each product in this match group
            for _, row in match_products.iterrows():
                output_rows.append({
                    'Match_ID': match_id,
                    'Match_Group_Name': match_group_name,
                    'SKU_ID': row['ProductCode'],
                    'SKU_Name': row['ProductDescription'],
                    'Company': row['Company'],
                    'Cluster_ID': cluster_id
                })
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_rows)
    
    print(f"\nOUTPUT SUMMARY:")
    print(f"  - Total match groups: {match_group_count}")
    print(f"  - Total products in matches: {len(output_df)}")
    print(f"  - Average products per match group: {len(output_df) / match_group_count if match_group_count > 0 else 0:.2f}")
    
    # Save to CSV - append to existing file if it exists
    import os
    if os.path.exists(output_path):
        # Read existing file to get current match IDs
        existing_df = pd.read_csv(output_path)
        
        # Find the highest match ID number to continue the sequence
        if not existing_df.empty and 'Match_ID' in existing_df.columns:
            existing_ids = existing_df['Match_ID'].tolist()
            max_id = 0
            for id_str in existing_ids:
                if id_str.startswith('MATCH_LLM_'):
                    try:
                        id_num = int(id_str.split('_')[-1])
                        max_id = max(max_id, id_num)
                    except ValueError:
                        pass
            
            # Update match IDs in the new data to continue the sequence
            if max_id > 0:
                # Create a mapping of original match IDs to new match IDs
                match_id_mapping = {}
                next_id = max_id + 1
                
                # First pass: create the mapping
                for row in output_rows:
                    original_id = row['Match_ID']
                    if original_id not in match_id_mapping:
                        match_id_mapping[original_id] = f"MATCH_LLM_{next_id:04d}"
                        next_id += 1
                
                # Second pass: apply the mapping
                for row in output_rows:
                    row['Match_ID'] = match_id_mapping[row['Match_ID']]
                # Recreate the DataFrame with updated IDs
                output_df = pd.DataFrame(output_rows)
        
        # Combine with existing data
        combined_df = pd.concat([existing_df, output_df], ignore_index=True)
        
        # Remove duplicates based on SKU_ID and Match_Group_Name
        combined_df = combined_df.drop_duplicates(subset=['SKU_ID', 'Match_Group_Name'])
        
        # Write the combined data
        combined_df.to_csv(output_path, index=False)
        print(f"\nAppended new matches to existing file: {output_path}")
    else:
        # Create new file if it doesn't exist
        output_df.to_csv(output_path, index=False)
        print(f"\nCreated new output file: {output_path}")
    
    logger.info(f"Saved {len(output_df)} exact match products to {output_path}")
    logger.info(f"Found {match_group_count} exact match groups across {len(match_results)} clusters")
    
    # Calculate average group size
    avg_group_size = len(output_df) / match_group_count if match_group_count > 0 else 0
    logger.info(f"Average group size: {avg_group_size:.2f} products per group")


def main():
    parser = argparse.ArgumentParser(description='Analyze product clusters to find exact matches using LLM')
    
    parser.add_argument('--clusters', type=str, 
                        default='/Users/eshantarneja/Documents/Git/VectorDB/product_clustering/data/refined_clustering/refined_clusters.json',
                        help='Path to refined_clusters.json')
    
    parser.add_argument('--transaction_data', type=str, 
                        default='/Users/eshantarneja/Documents/Git/VectorDB/Source_data/Actuals/Transaction_Report_Actual.xlsx',
                        help='Path to transaction data Excel file')
    
    parser.add_argument('--output', type=str, 
                        default='/Users/eshantarneja/Documents/Git/VectorDB/product_clustering/Analysis_Scripts/llm_exact_matches.csv',
                        help='Path to output CSV file')
    
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Number of clusters to sample for testing')
    
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='LLM model to use')
    
    parser.add_argument('--cluster_id', type=str,
                        help='Specific cluster ID to test (overrides sample_size)')
    
    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set. Please set it before running this script.")
        return

    # Load data
    logger.info("Loading cluster data...")
    all_clusters = load_clusters(args.clusters)
    
    logger.info("Loading transaction data...")
    transaction_df = load_transaction_data(args.transaction_data)
    
    if len(all_clusters) == 0 or len(transaction_df) == 0:
        logger.error("Failed to load required data. Exiting.")
        return
    
    # Sample clusters for testing or use specific cluster
    if args.cluster_id:
        if args.cluster_id in all_clusters:
            sampled_clusters = {args.cluster_id: all_clusters[args.cluster_id]}
            logger.info(f"Using specific cluster {args.cluster_id}")
        else:
            logger.error(f"Cluster ID {args.cluster_id} not found")
            return
    else:
        logger.info(f"Sampling {args.sample_size} clusters for testing...")
        sampled_clusters = sample_clusters(all_clusters, args.sample_size)
        
    logger.info(f"Selected {len(sampled_clusters)} clusters for analysis")
    
    # Process clusters with LLM
    logger.info("Processing clusters with LLM to find exact matches...")
    match_results = process_clusters_with_llm(sampled_clusters, transaction_df)
    
    # Generate output
    logger.info("Generating output CSV...")
    generate_output_csv(match_results, transaction_df, args.output)

if __name__ == "__main__":
    main()
