#!/usr/bin/env python3
"""
Debug USDA mapping data and its structure.
"""

import os
import sys
import pandas as pd
import json

def debug_usda_mapping():
    """
    Load and examine the USDA mapping file to debug issues.
    """
    mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "data", "CorrectMapping", "product_mapping_semantic.xlsx")
    
    print(f"Loading USDA mapping data from: {mapping_path}")
    
    # Load the prepared products to check codes
    products_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "data", "prepared_products.csv")
    
    try:
        # Try to load the Excel file
        df = pd.read_excel(mapping_path)
        print(f"Successfully loaded mapping with {len(df)} rows")
        print("\nFirst 5 rows:")
        print(df.head(5).to_string())
        
        print("\nColumns:")
        print(df.columns.tolist())
        
        # Try to parse product codes
        print("\nAttempting to parse product codes from 'Product Codes' column:")
        all_codes = []
        
        for idx, row in df.iterrows():
            # Determine the delimiter by trying different options
            if pd.isna(row['Product Codes']):
                print(f"  Row {idx}: No product codes found (NaN)")
                continue
                
            raw_codes = str(row['Product Codes'])
            
            # Try different delimiters
            potential_delimiters = [';', ',', ' ']
            for delimiter in potential_delimiters:
                if delimiter in raw_codes:
                    codes = [code.strip() for code in raw_codes.split(delimiter)]
                    print(f"  Row {idx}: Found {len(codes)} codes using delimiter '{delimiter}': {codes}")
                    all_codes.extend(codes)
                    break
            else:
                # No delimiter found, treat as a single code
                print(f"  Row {idx}: Found single code: {raw_codes}")
                all_codes.append(raw_codes.strip())
        
        print(f"\nTotal unique product codes found: {len(set(all_codes))}")
        
        # Load prepared products to check for matches
        print(f"\nLoading product data from: {products_path}")
        if os.path.exists(products_path):
            products_df = pd.read_csv(products_path)
            print(f"Loaded {len(products_df)} products")
            
            # Convert product codes to strings for comparison
            product_codes = set(products_df['product_code'].astype(str).tolist())
            
            # Find matches between mapping codes and actual product codes
            matches = set(all_codes).intersection(product_codes)
            print(f"\nMatches found: {len(matches)} out of {len(set(all_codes))} unique mapping codes")
            print(f"Match percentage: {len(matches)/len(set(all_codes))*100:.1f}%")
            
            # Print some example matches and non-matches
            print("\nExample matches (first 5):")
            for code in list(matches)[:5]:
                print(f"  {code}")
                
            non_matches = set(all_codes) - matches
            print("\nExample non-matches (first 5):")
            for code in list(non_matches)[:5]:
                print(f"  {code}")
                
            # Try to find similar codes in case of formatting differences
            print("\nChecking for potential formatting differences:")
            for non_match in list(non_matches)[:10]:
                similar = []
                for prod_code in product_codes:
                    # Try removing leading zeros
                    if non_match.lstrip('0') == prod_code.lstrip('0'):
                        similar.append(prod_code)
                    # Try different number formats
                    elif non_match.isdigit() and prod_code.isdigit() and int(non_match) == int(prod_code):
                        similar.append(prod_code)
                
                if similar:
                    print(f"  Non-matching code {non_match} has similar codes in product data: {similar}")
            
            # Save the mapping to a more structured format for debugging
            mapping_dict = {}
            for idx, row in df.iterrows():
                std_name = row['Standardized Product Name']
                raw_codes = str(row['Product Codes']) if not pd.isna(row['Product Codes']) else ""
                
                # Try to split using detected delimiter
                codes = []
                for delimiter in potential_delimiters:
                    if delimiter in raw_codes:
                        codes = [code.strip() for code in raw_codes.split(delimiter)]
                        break
                else:
                    if raw_codes:
                        codes = [raw_codes.strip()]
                
                mapping_dict[std_name] = {
                    'codes': codes,
                    'matching_codes': list(set(codes).intersection(product_codes))
                }
            
            # Save the structured mapping
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "data", "usda_mapping_debug.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(mapping_dict, f, indent=2)
            
            print(f"\nSaved debug mapping to: {output_path}")
            
    except Exception as e:
        print(f"Error debugging USDA mapping data: {e}")

if __name__ == "__main__":
    debug_usda_mapping()
