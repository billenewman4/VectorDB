#!/usr/bin/env python3
"""
Evaluate the accuracy of the vector database against reference mappings
"""
import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import vector database functions
from src.vectordb import ProductVectorDB, find_similar_products

# Import Excel functions
from src.excel import process_transaction_data, load_mapping_data

def parse_possible_descriptions(desc_str: str) -> List[str]:
    """Parse semicolon-separated descriptions and normalize them"""
    descriptions = [d.strip().lower() for d in desc_str.split(';')]
    return descriptions

def load_reference_mappings() -> Dict[str, List[str]]:
    """Load reference mappings from the Excel file"""
    print("Loading reference mappings...")
    # Load the reference mapping data
    mapping_path = Path(__file__).parent.parent / "data/CorrectMapping/product_mapping_semantic.xlsx"
    mapping_df = load_mapping_data(mapping_path)
    
    # Create a dictionary of standardized product names to possible descriptions
    reference_map = {}
    for _, row in mapping_df.iterrows():
        std_name = row['Standardized Product Name']
        descriptions = parse_possible_descriptions(row['Possible Descriptions'])
        reference_map[std_name] = descriptions
    
    print(f"Loaded {len(reference_map)} standardized products with {sum(len(d) for d in reference_map.values())} possible descriptions")
    return reference_map

def find_product_in_transaction_data(description: str, transaction_data: pd.DataFrame) -> bool:
    """Check if a product description exists in the transaction data"""
    # Normalize description
    description = description.lower().strip()
    
    # Check if description exists in transaction data
    exists = any(
        re.search(rf"\b{re.escape(description)}\b", prod.lower()) or 
        re.search(rf"\b{re.escape(prod.lower())}\b", description)
        for prod in transaction_data['product_description']
    )
    return exists

def evaluate_vector_db_accuracy() -> Dict[str, float]:
    """
    Evaluate the accuracy of the vector database against reference mappings
    
    Returns:
        Dictionary of accuracy metrics
    """
    # Load reference mappings
    reference_map = load_reference_mappings()
    
    # Load transaction data
    _, unique_products = process_transaction_data()
    
    # Initialize vector database
    print("Initializing vector database...")
    vector_db = ProductVectorDB()
    
    # Filter reference descriptions to only those found in transaction data
    filtered_reference_map = {}
    product_to_std_name = {}
    
    for std_name, descriptions in reference_map.items():
        filtered_descriptions = []
        for desc in descriptions:
            if find_product_in_transaction_data(desc, unique_products):
                filtered_descriptions.append(desc)
                product_to_std_name[desc] = std_name
        
        if filtered_descriptions:
            filtered_reference_map[std_name] = filtered_descriptions
    
    print(f"Found {len(filtered_reference_map)} standardized products with {sum(len(d) for d in filtered_reference_map.values())} descriptions in transaction data")
    
    # Track metrics
    total_queries = 0
    correct_matches = 0
    false_positives = 0
    false_negatives = 0
    
    # Evaluate each reference mapping
    print("\nEvaluating similarity matching accuracy...")
    
    for std_name, descriptions in filtered_reference_map.items():
        print(f"\nTesting standardized product: '{std_name}'")
        
        # For each description, use it as a query and check if other descriptions are in top N results
        for query_idx, query_desc in enumerate(descriptions):
            # Define correct matches (other descriptions for the same standard product)
            correct_set = set(d.lower() for d in descriptions if d.lower() != query_desc.lower())
            
            if not correct_set:
                continue  # Skip if there are no other descriptions to match
            
            total_queries += 1
            
            # Find similar products
            n_results = min(10, len(unique_products))  # Use top 10 or max available
            results = find_similar_products(query_desc, n_results=n_results, vector_db=vector_db)
            
            # Extract matched descriptions
            matched_products = set(result.lower() for result in results['product_description'])
            
            # Calculate correct matches
            matches_found = [prod for prod in matched_products if any(
                # Check if any correct description is in the matched product
                re.search(rf"\b{re.escape(correct.lower())}\b", prod.lower()) or
                re.search(rf"\b{re.escape(prod.lower())}\b", correct.lower())
                for correct in correct_set
            )]
            
            # Calculate metrics for this query
            query_correct_matches = len(matches_found)
            
            # All matched products that aren't in the correct set are false positives
            query_false_positives = len(matched_products) - query_correct_matches
            
            # All correct products that weren't matched are false negatives
            query_false_negatives = len(correct_set) - query_correct_matches
            
            # Update total metrics
            correct_matches += query_correct_matches
            false_positives += query_false_positives
            false_negatives += query_false_negatives
            
            # Print results for this query
            print(f"  Query: '{query_desc}'")
            print(f"    Expected matches: {len(correct_set)}")
            print(f"    Found matches: {query_correct_matches}")
            print(f"    Correct matches: {query_correct_matches}")
            print(f"    False positives: {query_false_positives}")
            print(f"    False negatives: {query_false_negatives}")
    
    # Calculate final metrics
    precision = correct_matches / (correct_matches + false_positives) if (correct_matches + false_positives) > 0 else 0
    recall = correct_matches / (correct_matches + false_negatives) if (correct_matches + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    accuracy_metrics = {
        'total_queries': total_queries,
        'correct_matches': correct_matches,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    # Print summary
    print(f"\n=== Accuracy Metrics ===")
    print(f"Total queries: {total_queries}")
    print(f"Correct matches: {correct_matches}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Interpret the results
    print("\n=== Performance Rating ===")
    
    if f1_score >= 0.8:
        print("Excellent: The vector database achieves high accuracy in matching similar products.")
    elif f1_score >= 0.6:
        print("Good: The vector database performs well but has some room for improvement.")
    elif f1_score >= 0.4:
        print("Fair: The vector database has moderate accuracy. Consider tuning the embedding model or parameters.")
    else:
        print("Poor: The vector database has low accuracy. Significant improvements are needed.")
    
    # Print additional information
    print("\nNote: Performance is evaluated based on how well the vector database groups")
    print("products that should be considered the same according to the reference mappings.")
    
    return accuracy_metrics

if __name__ == "__main__":
    print("Evaluating vector database accuracy...")
    metrics = evaluate_vector_db_accuracy()
