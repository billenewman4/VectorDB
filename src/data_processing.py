import pandas as pd
import src.config as config
import re
import os
import json
from collections import defaultdict
from src.abbreviation_translator import expand_abbreviations
from typing import Optional

def clean_text(text):
    """Basic text cleaning: lowercase, strip whitespace."""
    if isinstance(text, str):
        text = text.lower().strip()
        # Optional: remove excessive whitespace inside the string
        text = re.sub(r'\s+', ' ', text)
    return text

def load_transaction_data(file_path=config.TRANSACTION_REPORT_FILE, 
                          sheet_name=config.TRANSACTION_SHEET_NAME):
    """Loads transaction data from the specified Excel file and sheet."""
    print(f"Loading transaction data from: {file_path}, Sheet: {sheet_name}")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: Transaction file not found at {file_path}")
        return None
    except Exception as e: # Catch other potential errors like sheet not found
        print(f"Error loading transaction data: {e}")
        return None

def load_inventory_data(inventory_file=None):
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
            import glob
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
        
def _process_single_inventory_file(file_path, distributor_prefix=None):
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
                
                # Formerly added distributor prefix, now removed as requested
                # Keep category as is without distributor prefix
                    
                if code and category:  # Ensure non-empty strings
                    product_categories[code] = category
        
        return product_categories
        
    except Exception as e:
        print(f"Error processing inventory file {file_path}: {e}")
        return {}

def process_transaction_data(df_raw: Optional[pd.DataFrame] = None, 
                           code_col: str = None, 
                           desc_col: str = None,
                           filter_no_category: bool = True) -> pd.DataFrame:
    """
    Process transaction data to extract unique product descriptions.
    
    Args:
        df: Transaction DataFrame. If None, data will be loaded.
        code_col: Column name for product codes. If None, uses config value.
        desc_col: Column name for product descriptions. If None, uses config value.
        filter_no_category: Whether to filter out products without category data.
            
    Returns:
        DataFrame with unique product descriptions and their codes.
    """
    if df_raw is None:
        print("No raw data to process.")
        return None

    if code_col is None:
        code_col = config.TRANSACTION_PRODUCT_CODE_COL
    if desc_col is None:
        desc_col = config.TRANSACTION_DESC_COL

    print(f"Processing data using columns: Code='{code_col}', Description='{desc_col}'")

    # Check if required columns exist
    if code_col not in df_raw.columns or desc_col not in df_raw.columns:
        print(f"Error: Required columns ('{code_col}', '{desc_col}') not found in the dataframe.")
        print(f"Available columns: {df_raw.columns.tolist()}")
        return None

    # Select necessary columns
    df = df_raw[[code_col, desc_col]].copy()

    # Handle missing values
    initial_rows = len(df)
    df.dropna(subset=[code_col, desc_col], inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows with missing ProductCode or ProductDescription.")

    # Ensure consistent types
    df[code_col] = df[code_col].astype(str) # Codes should be strings
    df[desc_col] = df[desc_col].astype(str)

    # Clean description and expand meat cut abbreviations
    df['cleaned_description'] = df[desc_col].apply(lambda text: expand_abbreviations(clean_text(text)))

    # Filter out empty descriptions after cleaning
    initial_rows = len(df)
    df = df[df['cleaned_description'] != '']
    if len(df) < initial_rows:
         print(f"Dropped {initial_rows - len(df)} rows with empty descriptions after cleaning.")

    # Get unique descriptions and their first associated code
    print("Extracting unique descriptions and associated codes...")
    # Group by cleaned description, take the first occurrence's code
    unique_products_df = df.groupby('cleaned_description').first().reset_index()
    
    # Rename columns for clarity
    unique_products_df = unique_products_df.rename(columns={
        'cleaned_description': 'product_description', # This is the column to embed
        code_col: 'product_code' # The associated code for linking
    })
    
    # Load product categories from inventory data
    product_categories = load_inventory_data()
    
    # Add category column to the dataframe
    unique_products_df['product_category'] = unique_products_df['product_code'].apply(
        lambda code: product_categories.get(code, None)
    )
    
    # Filter out products without a valid category only if requested
    initial_count = len(unique_products_df)
    if filter_no_category:
        unique_products_df = unique_products_df.dropna(subset=['product_category'])
        filtered_count = initial_count - len(unique_products_df)
        print(f"Filtered out {filtered_count} products without valid category descriptions")
    else:
        print(f"Keeping all {initial_count} products regardless of category information")
    
    # Count products per category
    category_counts = unique_products_df['product_category'].value_counts()
    print("\nProduct categories found:")
    for category, count in category_counts.items():
        print(f"  - {category}: {count} products")
    
    # Select final columns
    final_cols = ['product_description', 'product_code', 'product_category']
    unique_products_df = unique_products_df[final_cols]

    print(f"Found {len(unique_products_df)} unique product descriptions for embedding.")
    
    return unique_products_df

def group_products_by_category(df):
    """
    Group products by their category for category-based clustering.
    This ensures products from different categories are never clustered together.
    
    Args:
        df: DataFrame with product information including product_category column
        
    Returns:
        Dictionary mapping categories to DataFrames of products in that category
    """
    if df is None or 'product_category' not in df.columns:
        print("Error: DataFrame does not contain product_category column")
        return {}
    
    # Group the dataframe by category
    grouped_products = {}
    
    # Group products by category - all products should have a valid category at this point
    for category, group_df in df.groupby('product_category'):
        grouped_products[category] = group_df.reset_index(drop=True)
    
    print(f"Grouped products into {len(grouped_products)} categories")
    return grouped_products

def save_category_products(category_products, output_dir):
    """
    Save category-to-products mapping for use in clustering processes.
    
    Args:
        category_products: Dictionary mapping categories to DataFrames of products
        output_dir: Directory to save the mapping
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert DataFrames to lists of product codes
        output_mapping = {}
        for category, df in category_products.items():
            product_codes = df['product_code'].tolist()
            output_mapping[category] = product_codes
            
        output_path = os.path.join(output_dir, "category_products.json")
        with open(output_path, 'w') as f:
            json.dump(output_mapping, f, indent=2)
            
        print(f"Saved category-to-products mapping to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error saving category products: {e}")
        return None

# Example usage (optional, for testing)
if __name__ == "__main__":
    raw_data = load_transaction_data()
    if raw_data is not None:
        processed_data = process_transaction_data(raw_data)
        if processed_data is not None:
            print("\n--- Processed Data Sample ---")
            print(processed_data.head())
            print(f"\nTotal unique products: {len(processed_data)}")
            # Check for nulls in final df
            print("\nNull checks:")
            print(processed_data.isnull().sum())
            
            # Test category grouping
            category_groups = group_products_by_category(processed_data)
            print("\nCategory groups created:")
            for category, df in category_groups.items():
                print(f"  - {category}: {len(df)} products")
            
            # Test saving category products
            save_category_products(category_groups, "./data")
