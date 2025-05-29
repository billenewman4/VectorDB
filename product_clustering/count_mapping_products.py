#!/usr/bin/env python3
"""
Count the number of product codes in the USDA mapping file and
check how many of those exist in the prepared products dataset.
"""

import os
import pandas as pd
import numpy as np

def main():
    # Paths
    mapping_path = os.path.join("Source_data", "Actuals", "Corrected_mapping.xlsx")
    products_path = os.path.join("product_clustering", "data", "prepared_products.csv")
    
    # Load the mapping file
    mapping_df = pd.read_excel(mapping_path)
    print(f"Total USDA codes in mapping file: {len(mapping_df)}")
    
    # Define distributor columns 
    distributor_cols = ['Fulton_code', 'Pritzlaff_code', 'Queen_code', 'Moesle_code', 'Anmar_code']
    
    # Count non-empty product codes in each row and column
    all_product_codes = []
    for _, row in mapping_df.iterrows():
        for col in distributor_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                all_product_codes.append(str(row[col]).strip())
    
    print(f"Total non-empty product codes in mapping file: {len(all_product_codes)}")
    print(f"Unique product codes in mapping file: {len(set(all_product_codes))}")
    
    # Count per distributor
    print("\nNon-empty product codes per distributor:")
    for col in distributor_cols:
        if col in mapping_df.columns:
            count = sum(1 for _, row in mapping_df.iterrows() if pd.notna(row.get(col)) and str(row.get(col)).strip())
            print(f"  - {col}: {count}")
    
    # Load the prepared products data
    try:
        products_df = pd.read_csv(products_path)
        print(f"\nTotal products in prepared_products.csv: {len(products_df)}")
        
        # Count how many mapped product codes exist in the prepared products
        product_codes_in_df = set(str(code) for code in products_df['product_code'])
        mapped_codes_in_products = [code for code in all_product_codes if code in product_codes_in_df]
        
        print(f"Mapped product codes found in prepared_products.csv: {len(mapped_codes_in_products)}")
        print(f"Mapped product codes NOT found in prepared_products.csv: {len(all_product_codes) - len(mapped_codes_in_products)}")
        
        # Show some examples of mapped codes not found in products
        missing_codes = [code for code in all_product_codes if code not in product_codes_in_df]
        if missing_codes:
            print("\nExamples of mapped codes not found in products (first 10):")
            for i, code in enumerate(missing_codes[:10]):
                print(f"  {i+1}. {code}")
        
    except Exception as e:
        print(f"Error loading products data: {e}")

if __name__ == "__main__":
    main()
