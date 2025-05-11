#!/usr/bin/env python3
"""
Analyze the lowest performing clusters and why they have low recall
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.vectordb import ProductVectorDB, find_similar_products

# Load the mapping file
mapping_df = pd.read_excel('CorrectMapping/product_mapping_semantic.xlsx')

# The lowest performing clusters identified in our test
worst_clusters = [
    'Organic Chicken Breast',
    'Frozen Jumbo Shrimp', 
    'Ground Turkey Frozen',
    'Almond Milk Unsweetened',
    'Brown Rice Long Grain',
    'Orange Juice Fresh Squeezed'
]

# Initialize vector DB
vector_db = ProductVectorDB()

print("Analyzing low-performing clusters...")

# Try different similarity thresholds
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]

for cluster_name in worst_clusters:
    print(f"\n{'='*50}")
    print(f"CLUSTER: {cluster_name}")
    
    # Get possible variations from mapping file
    row = mapping_df[mapping_df['Standardized Product Name'] == cluster_name]
    if row.empty:
        print(f"No mapping found for {cluster_name}")
        continue
    
    possible_desc = row.iloc[0]['Possible Descriptions']
    variations = [desc.strip() for desc in possible_desc.split(';')]
    
    print(f"Possible variations ({len(variations)}):")
    for v in variations:
        print(f"  - {v}")
    
    # Test different thresholds
    print("\nSearching with different thresholds:")
    threshold_results = {}
    
    for threshold in thresholds:
        results = find_similar_products(
            cluster_name, 
            n_results=20, 
            similarity_threshold=threshold,
            vector_db=vector_db
        )
        
        # Count how many variations we found
        found = set()
        for _, result_row in results.iterrows():
            product_desc = result_row["product_description"]
            similarity = result_row["similarity"]
            
            # Check if this result matches any variation
            for var in variations:
                if product_desc.lower() == var.lower():
                    found.add(var)
                    break
        
        recall = len(found) / len(variations) if variations else 0
        threshold_results[threshold] = (len(found), found)
        
        print(f"  Threshold {threshold}: Found {len(found)}/{len(variations)} variations (recall: {recall:.2f})")
        for f in found:
            idx = next((i for i, res in enumerate(results.iterrows()) if res[1]['product_description'].lower() == f.lower()), -1)
            sim_score = results.iloc[idx]['similarity'] if idx >= 0 else "N/A"
            print(f"    - {f} (similarity: {sim_score})")
        
        # Show what we're missing
        missing = set(variations) - found
        if missing:
            print(f"  Missing {len(missing)} variations:")
            for m in missing:
                print(f"    - {m}")
            
            # Find similarity scores for missing variations
            print("  Similarity scores for missing variations:")
            for m in missing:
                # Search for this specific variation
                m_results = find_similar_products(
                    m,  # Use missing variation as query
                    n_results=1,
                    vector_db=vector_db
                )
                if not m_results.empty:
                    m_sim = m_results.iloc[0]['similarity']
                    print(f"    - '{m}' as query -> '{m_results.iloc[0]['product_description']}' (similarity: {m_sim:.4f})")
