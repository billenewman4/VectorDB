"""
Analyze the coverage of warehouse codes and product categories in the unified product data.
"""
import os
import sys
import pandas as pd

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_prep.processor import prepare_unified_product_data

def analyze_coverage():
    """
    Analyze what percentage of products have warehouse codes and product categories.
    """
    print("Loading unified product data...")
    # Get the unified product data
    unified_data = prepare_unified_product_data()
    
    # Total number of products
    total_products = len(unified_data)
    print(f"Total unique products: {total_products}")
    
    # Count products with warehouse information
    products_with_warehouse = unified_data['product_warehouse'].notna().sum()
    warehouse_percentage = (products_with_warehouse / total_products) * 100
    
    # Count products with category information
    products_with_category = unified_data['category_description'].notna().sum()
    category_percentage = (products_with_category / total_products) * 100
    
    # Count products with both warehouse and category
    products_with_both = unified_data[unified_data['product_warehouse'].notna() & 
                                      unified_data['category_description'].notna()].shape[0]
    both_percentage = (products_with_both / total_products) * 100
    
    # Print results
    print(f"\nCoverage Analysis:")
    print(f"Products with warehouse information: {products_with_warehouse} ({warehouse_percentage:.2f}%)")
    print(f"Products with category information: {products_with_category} ({category_percentage:.2f}%)")
    print(f"Products with both warehouse and category: {products_with_both} ({both_percentage:.2f}%)")
    
    # Additional analysis - most common warehouses and categories
    if products_with_warehouse > 0:
        print("\nTop 5 Most Common Warehouses:")
        print(unified_data['product_warehouse'].value_counts().head(5).to_string())
    
    if products_with_category > 0:
        print("\nTop 5 Most Common Categories:")
        print(unified_data['category_description'].value_counts().head(5).to_string())

if __name__ == "__main__":
    analyze_coverage()
