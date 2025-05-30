#!/usr/bin/env python3
"""
Combine Product Data Script

This script extracts unique product codes from all inventory Excel files and transaction 
Excel files, then combines them with product descriptions, category descriptions, and company information.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Set, Tuple
import xlrd
import openpyxl
import json
from tqdm import tqdm

def load_excel_file(file_path: str) -> pd.DataFrame:
    """
    Load an Excel file, supporting both .xls and .xlsx formats
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame containing the Excel data
    """
    try:
        if file_path.endswith('.xls'):
            # For older .xls files
            return pd.read_excel(file_path, engine='xlrd')
        else:
            # For newer .xlsx files
            return pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def extract_product_codes_from_inventory(file_path: str, company: str) -> List[Dict[str, Any]]:
    """
    Extract product codes and descriptions from inventory files
    
    Args:
        file_path: Path to the inventory Excel file
        company: Name of the company (distributor)
        
    Returns:
        List of dictionaries containing product information
    """
    print(f"Processing inventory file: {file_path}")
    
    try:
        df = load_excel_file(file_path)
        
        # Common column names to check for product codes
        code_columns = ['Product_Code', 'Product Code', 'Code', 'Item Number', 'Item_Number', 'SKU', 
                        'Item', 'ItemCode', 'Item_Code', 'ProductCode', 'Product']
        
        # Common column names to check for descriptions
        desc_columns = ['Description', 'Product_Description', 'Product Description', 'Item Description', 
                        'Item_Description', 'ProductDescription', 'Desc']
        
        # Common column names to check for categories
        category_columns = ['Category', 'Cat', 'Department', 'Dept', 'Group', 'Product Group', 
                            'Product_Group', 'ProductGroup', 'Item_Category', 'Item Category']
        
        # Find the actual column names in the file
        code_col = next((col for col in code_columns if col in df.columns), None)
        desc_col = next((col for col in desc_columns if col in df.columns), None)
        category_col = next((col for col in category_columns if col in df.columns), None)
        
        # If we couldn't find the columns, make an educated guess based on position and content
        if code_col is None and len(df.columns) > 0:
            # Try the first column for product code
            code_col = df.columns[0]
            print(f"Using column '{code_col}' for product codes")
        
        # For description and category, try to make a better assessment
        # Sample some values to determine which column is likely the description
        description_found = False
        category_found = False
        
        # First check for columns that clearly indicate description or category
        for col in df.columns:
            if col not in [code_col] and not description_found:
                # Sample values to see if they look like descriptions
                sample_values = df[col].dropna().astype(str).head(20).tolist()
                # Look for patterns suggesting descriptions (multiple words, consistent with food products)
                if any(len(str(val).split()) > 2 for val in sample_values) or \
                   any(food_keyword in ' '.join(sample_values).lower() for food_keyword in 
                       ['beef', 'chicken', 'pork', 'fish', 'meat', 'vegetable', 'fruit', 'dairy']):
                    desc_col = col
                    description_found = True
                    print(f"Using column '{desc_col}' for descriptions (identified by content)")
                # If it looks like codes with suffixes, don't use as description
                elif all('-' in str(val) or str(val).isdigit() for val in sample_values if val and str(val).strip()):
                    print(f"Column '{col}' appears to contain codes, not descriptions")
            
            # Look for category columns
            if col not in [code_col, desc_col] and not category_found:
                if any(cat_term in col.lower() for cat_term in ['cat', 'group', 'dept', 'class']):
                    category_col = col
                    category_found = True
                    print(f"Using column '{category_col}' for categories (identified by name)")
        
        # If we still haven't found description or category, use position as a fallback
        if not description_found and len(df.columns) > 1:
            for col in df.columns:
                if col not in [code_col] and not description_found:
                    sample_values = df[col].dropna().astype(str).head(5).tolist()
                    if not all(str(val).strip() == '' for val in sample_values):
                        desc_col = col
                        description_found = True
                        print(f"Using column '{desc_col}' for descriptions (fallback)")
                        break
        
        if not category_found and len(df.columns) > 2:
            potential_category_cols = [col for col in df.columns if col not in [code_col, desc_col]]
            if potential_category_cols:
                category_col = potential_category_cols[0]
                print(f"Using column '{category_col}' for categories (fallback)")
                
        # Special case for food product files - if there's a column named 'Category Description' and
        # it contains food-related terms, use it as the product description
        if 'Category Description' in df.columns:
            sample_cat = df['Category Description'].dropna().astype(str).head(20).tolist()
            cat_text = ' '.join(sample_cat).lower()
            food_terms = ['beef', 'chicken', 'pork', 'fish', 'meat', 'poultry', 'vegetable', 'fruit', 'dairy']
            
            if any(term in cat_text for term in food_terms):
                print(f"Using 'Category Description' for product descriptions based on food term analysis")
                desc_col = 'Category Description'
                
                # Find another column for category if possible
                for col in df.columns:
                    if col not in [code_col, desc_col] and any(cat_term in col.lower() for cat_term in ['category', 'group', 'dept', 'class', 'type']):
                        if col != 'Category Description':
                            category_col = col
                            print(f"Using '{category_col}' for categories")
                            break
        
        # Extract data
        products = []
        for _, row in df.iterrows():
            product_code = str(row.get(code_col, '')).strip() if code_col else ''
            
            # Skip empty product codes
            if not product_code or pd.isna(product_code) or product_code == 'nan':
                continue
            
            description = str(row.get(desc_col, '')).strip() if desc_col else ''
            category = str(row.get(category_col, '')).strip() if category_col else ''
            
            products.append({
                'product_code': product_code,
                'product_description': description,
                'category_description': category,
                'company': company
            })
        
        return products
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def extract_product_codes_from_transactions(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract product codes and descriptions from transaction files
    
    Args:
        file_path: Path to the transaction Excel file
        
    Returns:
        List of dictionaries containing product information
    """
    print(f"Processing transaction file: {file_path}")
    
    try:
        df = load_excel_file(file_path)
        
        # Check for specific columns in the transaction file based on what we saw in the data exploration
        if 'ProductDescription' in df.columns and 'ProductCode' in df.columns:
            # This is likely the Transaction_Report_Actual.xlsx format
            code_col = 'ProductCode'
            desc_col = 'ProductDescription'
            company_col = 'Company' if 'Company' in df.columns else None
            print(f"Using standard transaction report format with '{desc_col}' for descriptions")
        elif 'Product Code' in df.columns and 'Product Description' in df.columns:
            # This is likely the product_transactions_semantic.xlsx format
            code_col = 'Product Code'
            desc_col = 'Product Description'
            company_col = None
            print(f"Using semantic transaction format with '{desc_col}' for descriptions")
        else:
            # Common column names to check for product codes
            code_columns = ['Product_Code', 'Product Code', 'Code', 'Item Number', 'Item_Number', 'SKU', 
                            'Item', 'ItemCode', 'Item_Code', 'ProductCode', 'Product']
            
            # Common column names to check for descriptions
            desc_columns = ['Description', 'Product_Description', 'Product Description', 'Item Description', 
                            'Item_Description', 'ProductDescription', 'Desc']
            
            # Common column names to check for company/distributor
            company_columns = ['Company', 'Distributor', 'Vendor', 'Supplier', 'Source']
            
            # Find the actual column names in the file
            code_col = next((col for col in code_columns if col in df.columns), None)
            desc_col = next((col for col in desc_columns if col in df.columns), None)
            company_col = next((col for col in company_columns if col in df.columns), None)
            
            # If we still couldn't find the columns, try to infer them from the data
            if code_col is None and len(df.columns) > 0:
                # Look for columns that might contain product codes
                for col in df.columns:
                    sample_values = df[col].dropna().astype(str).head(10).tolist()
                    # Product codes are often alphanumeric with specific patterns
                    if all(len(str(val)) < 20 for val in sample_values):
                        code_col = col
                        print(f"Using column '{code_col}' for product codes")
                        break
            
            if desc_col is None and len(df.columns) > 1:
                # Look for columns that might contain descriptions
                for col in df.columns:
                    if col != code_col:
                        sample_values = df[col].dropna().astype(str).head(10).tolist()
                        # Descriptions are usually longer and contain spaces
                        if any(len(str(val).split()) > 1 for val in sample_values):
                            desc_col = col
                            print(f"Using column '{desc_col}' for descriptions")
                            break
        
        # Extract data
        products = []
        for _, row in df.iterrows():
            product_code = str(row.get(code_col, '')).strip() if code_col else ''
            
            # Skip empty product codes
            if not product_code or pd.isna(product_code) or product_code == 'nan':
                continue
            
            description = str(row.get(desc_col, '')).strip() if desc_col else ''
            
            # Skip items with no description or just numbers/codes as descriptions
            if not description or description == 'nan' or (description.replace('-', '').isdigit() and len(description) < 10):
                continue
                
            company = str(row.get(company_col, '')).strip() if company_col else ''
            
            products.append({
                'product_code': product_code,
                'product_description': description,
                'category_description': '',  # Transactions might not have category info
                'company': company
            })
        
        return products
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def load_usda_mapping(mapping_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the USDA to product code mapping data
    
    Args:
        mapping_path: Path to the mapping Excel file
        
    Returns:
        Dictionary mapping product codes to USDA information
    """
    try:
        df = load_excel_file(mapping_path)
        
        # Create mapping from product codes to USDA information
        mapping = {}
        
        # The distributor code columns (these might vary)
        distributor_columns = ['Fulton_code', 'Pritzlaff_code', 'Queen_code', 'Moesle_code', 'Anmar_code']
        
        # Adjust based on actual columns in the file
        actual_dist_columns = [col for col in distributor_columns if col in df.columns]
        if not actual_dist_columns:
            # Try to find columns that might contain distributor codes
            potential_dist_columns = [col for col in df.columns if 'code' in col.lower()]
            if potential_dist_columns:
                actual_dist_columns = potential_dist_columns
        
        for _, row in df.iterrows():
            usda_code = str(row.get('USDA_Code', '')).strip() if 'USDA_Code' in df.columns else ''
            usda_description = ''
            
            # Try to find a column with USDA description
            for col in df.columns:
                if 'description' in col.lower() or 'name' in col.lower() or 'product' in col.lower():
                    usda_description = str(row.get(col, '')).strip()
                    break
            
            # Get all product codes from the distributor columns
            for col in actual_dist_columns:
                if pd.notna(row.get(col)) and str(row.get(col)).strip():
                    code = str(row.get(col)).strip()
                    
                    # Strip the -1 or -2 suffix from product codes if present
                    if code.endswith('-1') or code.endswith('-2'):
                        code = code.rsplit('-', 1)[0]
                    
                    mapping[code] = {
                        'usda_code': usda_code,
                        'usda_description': usda_description
                    }
        
        return mapping
    
    except Exception as e:
        print(f"Error loading USDA mapping data: {e}")
        return {}

def merge_product_data(inventory_products: List[Dict[str, Any]], 
                       transaction_products: List[Dict[str, Any]],
                       usda_mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge product data from inventory and transaction files
    
    Args:
        inventory_products: List of products from inventory files
        transaction_products: List of products from transaction files
        usda_mapping: USDA mapping data
        
    Returns:
        List of merged product information
    """
    # Create a dictionary to store unique products by code
    merged_products = {}
    
    # Process transaction products FIRST (since they have better descriptions)
    for product in transaction_products:
        code = product['product_code']
        if code not in merged_products:
            merged_products[code] = product
        else:
            # Merge with existing product
            # If the transaction product has a description, use it (overwrite)
            if product.get('product_description'):
                merged_products[code]['product_description'] = product['product_description']
            
            # Keep track of all companies
            companies = set([merged_products[code].get('company', '')])
            if product.get('company'):
                companies.add(product['company'])
            merged_products[code]['company'] = ', '.join(filter(None, companies))
    
    # Process inventory products SECOND (as fallback)
    for product in inventory_products:
        code = product['product_code']
        if code not in merged_products:
            merged_products[code] = product
        else:
            # For description, only use inventory description if we don't already have one
            # from transaction data, and if the inventory description is meaningful
            if not merged_products[code].get('product_description') and product.get('product_description'):
                # Check if the description is just a product code with suffix
                desc = product.get('product_description', '')
                if not (desc.replace('-', '').isdigit() or desc == code or desc == code + '-1'):
                    merged_products[code]['product_description'] = desc
            
            # Always capture category descriptions
            if product.get('category_description') and not merged_products[code].get('category_description'):
                merged_products[code]['category_description'] = product['category_description']
            
            # Keep track of all companies
            companies = set([merged_products[code].get('company', '')])
            if product.get('company'):
                companies.add(product['company'])
            merged_products[code]['company'] = ', '.join(filter(None, companies))
    
    # Add USDA mapping information
    for code, product in merged_products.items():
        if code in usda_mapping:
            product['usda_code'] = usda_mapping[code].get('usda_code', '')
            product['usda_description'] = usda_mapping[code].get('usda_description', '')
        else:
            product['usda_code'] = ''
            product['usda_description'] = ''
        
        # Fallback: If we still don't have a proper description, use category description as product description
        if (not product.get('product_description') or 
            product.get('product_description') == code or 
            product.get('product_description') == code + '-1') and product.get('category_description'):
            product['product_description'] = product['category_description']
    
    return list(merged_products.values())

def main():
    # Set project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define source data directories
    source_data_dir = os.path.join(project_root, "Source_data")
    actuals_dir = os.path.join(source_data_dir, "Actuals")
    transactions_dir = os.path.join(source_data_dir, "Transactions")
    
    # Define output directory
    output_dir = os.path.join(project_root, "product_clustering", "data", "combined_products")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the distributor company name mapping
    company_mapping = {
        "Anmar Inventory Valuation.xls": "Anmar",
        "Fulton Inventory Valuation.xls": "Fulton",
        "Moesle Inventory Valuation - email.xls": "Moesle",
        "Pritzlaff Inventory Report.xls": "Pritzlaff",
        "Queen Inventory Valuation.xls": "Queen"
    }
    
    # Collect inventory products
    inventory_products = []
    for filename in os.listdir(actuals_dir):
        if filename.endswith('.xls') or filename.endswith('.xlsx'):
            if filename.startswith('~$'):  # Skip temporary Excel files
                continue
            
            if "Transaction" in filename:  # Skip transaction files in this loop
                continue
            
            if filename == "Corrected_mapping.xlsx":  # Skip mapping file
                continue
            
            company = company_mapping.get(filename, os.path.splitext(filename)[0])
            file_path = os.path.join(actuals_dir, filename)
            
            products = extract_product_codes_from_inventory(file_path, company)
            inventory_products.extend(products)
    
    print(f"Extracted {len(inventory_products)} products from inventory files")
    
    # Collect transaction products
    transaction_products = []
    
    # First check the Actuals directory for transaction files
    for filename in os.listdir(actuals_dir):
        if (filename.endswith('.xls') or filename.endswith('.xlsx')) and "Transaction" in filename:
            if filename.startswith('~$'):  # Skip temporary Excel files
                continue
            
            file_path = os.path.join(actuals_dir, filename)
            products = extract_product_codes_from_transactions(file_path)
            transaction_products.extend(products)
    
    # Then check the Transactions directory
    if os.path.exists(transactions_dir):
        for filename in os.listdir(transactions_dir):
            if filename.endswith('.xls') or filename.endswith('.xlsx'):
                if filename.startswith('~$'):  # Skip temporary Excel files
                    continue
                
                file_path = os.path.join(transactions_dir, filename)
                products = extract_product_codes_from_transactions(file_path)
                transaction_products.extend(products)
    
    print(f"Extracted {len(transaction_products)} products from transaction files")
    
    # Load USDA mapping
    mapping_path = os.path.join(actuals_dir, "Corrected_mapping.xlsx")
    usda_mapping = load_usda_mapping(mapping_path)
    print(f"Loaded {len(usda_mapping)} USDA mappings")
    
    # Merge all product data
    merged_products = merge_product_data(inventory_products, transaction_products, usda_mapping)
    print(f"Created {len(merged_products)} unique product entries after merging")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(merged_products)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "combined_product_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved combined product data to {csv_path}")
    
    # Save as Excel
    excel_path = os.path.join(output_dir, "combined_product_data.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Saved combined product data to {excel_path}")
    
    # Create a summary of the data
    summary = {
        "total_products": int(len(df)),
        "products_with_description": int(df['product_description'].notna().sum()),
        "products_with_category": int(df['category_description'].notna().sum()),
        "products_with_company": int(df['company'].notna().sum()),
        "products_with_usda_mapping": int(df['usda_code'].notna().sum()),
        "unique_companies": int(df['company'].nunique()),
        "unique_categories": int(df['category_description'].nunique()),
        "unique_usda_codes": int(df['usda_code'].nunique())
    }
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, "combined_product_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Generate a brief report in markdown format
    report = f"""# Combined Product Data Report

## Summary
- Total unique products: {summary['total_products']}
- Products with descriptions: {summary['products_with_description']} ({summary['products_with_description']/summary['total_products']*100:.1f}%)
- Products with category info: {summary['products_with_category']} ({summary['products_with_category']/summary['total_products']*100:.1f}%)
- Products with company info: {summary['products_with_company']} ({summary['products_with_company']/summary['total_products']*100:.1f}%)
- Products with USDA mapping: {summary['products_with_usda_mapping']} ({summary['products_with_usda_mapping']/summary['total_products']*100:.1f}%)

## Source Files
- Inventory files processed: {len([f for f in os.listdir(actuals_dir) if f.endswith(('.xls', '.xlsx')) and not f.startswith('~$') and 'Transaction' not in f and f != 'Corrected_mapping.xlsx'])}
- Transaction files processed: {len([f for f in os.listdir(actuals_dir) if f.endswith(('.xls', '.xlsx')) and not f.startswith('~$') and 'Transaction' in f]) + (len([f for f in os.listdir(transactions_dir) if f.endswith(('.xls', '.xlsx')) and not f.startswith('~$')]) if os.path.exists(transactions_dir) else 0)}

## Companies
- Unique companies/distributors: {summary['unique_companies']}

## Categories
- Unique product categories: {summary['unique_categories']}

## USDA Mapping
- Unique USDA codes: {summary['unique_usda_codes']}
- USDA mapping coverage: {summary['products_with_usda_mapping']/summary['total_products']*100:.1f}%

## Next Steps
- The combined product data is available in both CSV and Excel formats
- You can use this data for further analysis and clustering
"""
    
    # Save report as markdown
    report_path = os.path.join(output_dir, "combined_product_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")
    
    return csv_path, excel_path, summary_path, report_path

if __name__ == "__main__":
    main()
