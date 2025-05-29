#!/usr/bin/env python3
"""
Create a realistic USDA mapping file using actual product codes from the dataset.
This will let us evaluate how well our clustering matches expected product groupings.
"""

import os
import sys
import pandas as pd
import json
import random
from collections import defaultdict

def create_realistic_mapping():
    """
    Create a realistic mapping file with actual product codes.
    """
    # Load the prepared products data
    products_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "data", "prepared_products.csv")
    
    # Load category mapping if available
    category_products_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "data", "category_clustering", "refined", "category_products.json")
    
    try:
        print(f"Loading product data from: {products_path}")
        products_df = pd.read_csv(products_path)
        print(f"Loaded {len(products_df)} products")
        
        # Get unique categories
        categories = products_df['category_description'].unique()
        print(f"Found {len(categories)} unique categories")
        
        # Create category to products mapping
        category_mapping = defaultdict(list)
        for _, row in products_df.iterrows():
            if pd.notna(row['category_description']) and pd.notna(row['product_code']):
                category_mapping[row['category_description']].append(str(row['product_code']))
        
        # Load category products if available
        category_products = None
        if os.path.exists(category_products_path):
            try:
                with open(category_products_path, 'r') as f:
                    category_products = json.load(f)
                print(f"Loaded category-product mapping from {category_products_path}")
            except Exception as e:
                print(f"Error loading category-product mapping: {e}")
        
        # Create mapping for categories with sufficient products
        mapping_data = []
        
        # Use the loaded category mapping directly
        if category_products:
            print(f"Using category_products mapping with {len(category_products)} categories")
            # Select categories with enough products
            categories_with_sufficient_products = 0
            for category, data in category_products.items():
                # The structure has a 'clustered' key with the product codes
                if 'clustered' in data:
                    products = data['clustered']
                    if len(products) >= 5:  # Only include categories with enough products
                        categories_with_sufficient_products += 1
                        # Sample products if there are too many
                        sampled_products = products if len(products) <= 15 else random.sample(products, 15)
                        mapping_data.append({
                            'Standardized Product Name': category,
                            'Product Codes': ';'.join(sampled_products),
                            'Possible Descriptions': f'Category with {len(products)} products'
                        })
                else:
                    print(f"Category {category} does not have a 'clustered' key")
            print(f"Found {categories_with_sufficient_products} categories with 5+ products")
        else:
            # Use the mapping created from the dataframe
            print(f"Using category_mapping with {len(category_mapping)} categories")
            categories_with_sufficient_products = 0
            for category, products in category_mapping.items():
                if len(products) >= 5:  # Only include categories with enough products
                    categories_with_sufficient_products += 1
                    # Sample products if there are too many
                    sampled_products = products if len(products) <= 15 else random.sample(products, 15)
                    mapping_data.append({
                        'Standardized Product Name': category,
                        'Product Codes': ';'.join(sampled_products),
                        'Possible Descriptions': f'Category with {len(products)} products'
                    })
            print(f"Found {categories_with_sufficient_products} categories with 5+ products")
        
        # Create DataFrame and save to Excel
        mapping_df = pd.DataFrame(mapping_data)
        
        # Limit to a reasonable number of categories for the analysis
        if len(mapping_df) > 50:
            mapping_df = mapping_df.sample(50, random_state=42)
            
        print(f"Created mapping with {len(mapping_df)} categories")
        
        # Save the mapping
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "CorrectMapping", "realistic_product_mapping.xlsx")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        mapping_df.to_excel(output_path, index=False)
        print(f"Saved realistic mapping to: {output_path}")
        
        # Also save as CSV for easier viewing
        csv_path = output_path.replace('.xlsx', '.csv')
        mapping_df.to_csv(csv_path, index=False)
        print(f"Also saved as CSV: {csv_path}")
        
    except Exception as e:
        print(f"Error creating realistic mapping: {e}")

if __name__ == "__main__":
    create_realistic_mapping()
