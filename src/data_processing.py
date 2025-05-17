import pandas as pd
import src.config as config
import re
from src.abbreviation_translator import expand_abbreviations

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

def process_transaction_data(df_raw):
    """
    Processes the raw transaction data to extract unique product descriptions 
    and their associated product codes.
    """
    if df_raw is None:
        print("No raw data to process.")
        return None

    code_col = config.TRANSACTION_PRODUCT_CODE_COL
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
    
    # Select final columns
    final_cols = ['product_description', 'product_code'] 
    unique_products_df = unique_products_df[final_cols]

    print(f"Found {len(unique_products_df)} unique product descriptions for embedding.")
    
    return unique_products_df

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
