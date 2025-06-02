"""
Inventory data loading module.

This module provides functions for loading product category information 
from inventory reports with intelligent column detection.
"""

import os
import sys
import glob
import pandas as pd
from typing import Dict, Optional

# Add parent directories to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define proper config class for inventory loading
class Config:
    """
    Configuration class with inventory data paths and settings.
    """
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # Default inventory search directory
    INVENTORY_DIR = os.path.join(PROJECT_ROOT, "Source_data", "Actuals")

config = Config()

def load_inventory_data(inventory_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load product category information from inventory reports.
    
    Args:
        inventory_file: Path to inventory report file. If None, will try to locate it.
        
    Returns:
        Dictionary mapping product_code to category_description
    """
    try:
        # Dictionary to store product categories
        product_categories = {}
        
        # If specific inventory file is provided, use only that one
        if inventory_file is not None and os.path.exists(inventory_file):
            return _process_single_inventory_file(inventory_file)
            
        # Otherwise, try to find and process all inventory files
        inventory_files = []
        
        # Look in Source_data/Actuals for the specific inventory files
        actuals_dir = os.path.join(config.PROJECT_ROOT, "Source_data", "Actuals")
        if os.path.exists(actuals_dir):
            # Define patterns to match inventory file names
            inventory_patterns = [
                "*inventory*.xls*",
                "*Inventory*.xls*"
            ]
            
            # Find all files matching patterns
            for pattern in inventory_patterns:
                matching_files = glob.glob(os.path.join(actuals_dir, pattern))
                inventory_files.extend(matching_files)
                
            # Filter out any temporary Excel files
            inventory_files = [f for f in inventory_files if not os.path.basename(f).startswith("~$")]
        
        if not inventory_files:
            print("Warning: No inventory files found. Products will not have category information.")
            return {}
            
        print(f"Found {len(inventory_files)} inventory files to process")
        for i, file_path in enumerate(inventory_files):
            print(f"Processing inventory file {i+1}/{len(inventory_files)}: {os.path.basename(file_path)}")
            
            # Get distributor name from filename
            base_name = os.path.basename(file_path).lower()
            distributor = "Unknown"
            if "anmar" in base_name:
                distributor = "Anmar"
            elif "fulton" in base_name:
                distributor = "Fulton"
            elif "moesle" in base_name:
                distributor = "Moesle"
            elif "pritzlaff" in base_name:
                distributor = "Pritzlaff"
            elif "queen" in base_name:
                distributor = "Queen"
                
            # Process file and get categories
            file_categories = _process_single_inventory_file(file_path, distributor_prefix=distributor)
            print(f"  - Found {len(file_categories)} product-to-category mappings")
            
            # Add to overall mapping
            product_categories.update(file_categories)
        
        total_categories = len(product_categories)
        if total_categories > 0:
            print(f"\nSuccessfully loaded {total_categories} total product-to-category mappings")
        else:
            print("Warning: No category information found in inventory files.")
            
        return product_categories
        
    except Exception as e:
        print(f"Error loading inventory data: {e}")
        return {}


def _process_single_inventory_file(file_path: str, distributor_prefix: Optional[str] = None) -> Dict[str, str]:
    """
    Process a single inventory file to extract category information.
    
    Args:
        file_path: Path to the inventory file
        distributor_prefix: Optional prefix to add to categories
        
    Returns:
        Dictionary mapping product codes to categories
    """
    try:
        # Load the inventory data
        print(f"Loading inventory data from: {file_path}")
        df_inventory = None
        if file_path.lower().endswith(".csv"):
            df_inventory = pd.read_csv(file_path)
        else:
            # Try different Excel engines in case of issues
            try:
                df_inventory = pd.read_excel(file_path)
            except Exception as e1:
                try:
                    df_inventory = pd.read_excel(file_path, engine='openpyxl')
                except Exception as e2:
                    print(f"Error loading {file_path}: {e1}; {e2}")
                    return {}
        
        if df_inventory is None or df_inventory.empty:
            print(f"No data found in {file_path}")
            return {}
            
        print(f"Loaded {len(df_inventory)} rows from {file_path}")
        print(f"Columns: {list(df_inventory.columns)}")
        
        # Identify columns containing product codes and categories
        # Look for columns with common naming patterns
        code_patterns = ["code", "sku", "item", "product", "number"]
        category_patterns = ["category", "group", "department", "class", "type"]
        
        code_col = next((col for col in df_inventory.columns if any(pattern in str(col).lower() 
                                                              for pattern in code_patterns)), None)
        category_col = next((col for col in df_inventory.columns if any(pattern in str(col).lower() 
                                                                   for pattern in category_patterns)), None)
        
        # If we can't find category, try to use description as a fallback for category inference
        desc_col = None
        if category_col is None:
            desc_col = next((col for col in df_inventory.columns if any(pattern in str(col).lower() 
                                                                  for pattern in ["desc", "name", "title"])), None)
        
        if code_col is None or (category_col is None and desc_col is None):
            print(f"Warning: Could not identify necessary columns in {file_path}.")
            print(f"Available columns: {df_inventory.columns.tolist()}")
            return {}
        
        if category_col:
            print(f"Using columns: {code_col} (code) and {category_col} (category)")
        else:
            print(f"Using columns: {code_col} (code) and {desc_col} (description for category inference)")
        
        # Create mapping from product code to category
        product_categories = {}
        
        for _, row in df_inventory.iterrows():
            if pd.notna(row[code_col]):
                # Convert code to string and clean
                code = str(row[code_col]).strip()
                
                # Get category - either from category column or infer from description
                if category_col and pd.notna(row[category_col]):
                    category = str(row[category_col]).strip()
                elif desc_col and pd.notna(row[desc_col]):
                    # Infer category from first word of description or some other heuristic
                    desc = str(row[desc_col]).strip()
                    # Simple heuristic: use first word of description as category if it's a noun
                    words = desc.split()
                    if words:
                        category = words[0].strip().title()
                    else:
                        continue
                else:
                    continue
                
                # Add distributor prefix if provided
                if distributor_prefix:
                    # Only add prefix if it's not already part of the category
                    if distributor_prefix.lower() not in category.lower():
                        category = f"{distributor_prefix} - {category}"
                    
                if code and category:  # Ensure non-empty strings
                    product_categories[code] = category
        
        return product_categories
        
    except Exception as e:
        print(f"Error processing inventory file {file_path}: {e}")
        return {}


if __name__ == "__main__":
    # Test the inventory data loading
    product_categories = load_inventory_data()
    
    if product_categories:
        print("\nSample of product categories:")
        count = 0
        for code, category in product_categories.items():
            print(f"{code}: {category}")
            count += 1
            if count >= 10:
                break
        
        print(f"\nTotal categories loaded: {len(product_categories)}")
