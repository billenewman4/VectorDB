import pandas as pd
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Define file paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TRANSACTION_FILE = BASE_DIR / 'Transactions' / 'product_transactions_semantic.xlsx'
MAPPING_FILE = BASE_DIR / 'CorrectMapping' / 'product_mapping_semantic.xlsx'


def load_transaction_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load transaction data from Excel file
    
    Args:
        file_path: Path to the Excel file, defaults to predefined TRANSACTION_FILE
        
    Returns:
        DataFrame containing transaction data
    """
    if file_path is None:
        file_path = TRANSACTION_FILE
        
    if not file_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {file_path}")
        
    print(f"Loading transaction data from {file_path}")
    return pd.read_excel(file_path)


def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess transaction data
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Expected columns (adapt based on actual file structure)
    expected_columns = {
        'product_description': str,
        'price': float,
        'quantity': int,
        # Add other columns as needed
    }
    
    # Check if essential columns exist
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        similar_cols = {}
        for missing in missing_cols:
            # Find similar column names
            pattern = re.compile(f".*{missing.replace('_', '.*')}.*", re.IGNORECASE)
            matches = [col for col in df.columns if pattern.match(col)]
            if matches:
                similar_cols[missing] = matches
        
        # Rename columns if we have matches
        for missing, matches in similar_cols.items():
            if len(matches) == 1:
                df.rename(columns={matches[0]: missing}, inplace=True)
                missing_cols.remove(missing)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle missing values
    if 'product_description' in df.columns:
        # Drop rows with missing product descriptions
        df = df.dropna(subset=['product_description'])
        # Standardize text: strip whitespace, convert to lowercase
        df['product_description'] = df['product_description'].str.strip().str.lower()
    
    # Convert data types
    for col, dtype in expected_columns.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to {dtype}: {e}")
    
    return df


def get_unique_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique product descriptions with aggregated metrics
    
    Args:
        df: Cleaned transaction DataFrame
        
    Returns:
        DataFrame with unique products and their aggregated metrics
    """
    # Group by product description and calculate aggregates
    unique_products = df.groupby('product_description').agg(
        transaction_count=('product_description', 'count'),
        avg_price=('price', 'mean'),
        avg_quantity=('quantity', 'mean'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        total_quantity=('quantity', 'sum')
    ).reset_index()
    
    return unique_products


def load_mapping_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load product mapping data from Excel file
    
    Args:
        file_path: Path to the Excel file, defaults to predefined MAPPING_FILE
        
    Returns:
        DataFrame containing mapping data
    """
    if file_path is None:
        file_path = MAPPING_FILE
        
    if not file_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {file_path}")
        
    print(f"Loading mapping data from {file_path}")
    return pd.read_excel(file_path)


def process_transaction_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline to load, clean, and process transaction data
    
    Returns:
        Tuple of (cleaned_transactions, unique_products)
    """
    # Load raw data
    raw_data = load_transaction_data()
    
    # Clean data
    cleaned_data = clean_transaction_data(raw_data)
    
    # Get unique products
    unique_products = get_unique_products(cleaned_data)
    
    print(f"Processed {len(raw_data)} transactions into {len(unique_products)} unique products")
    
    return cleaned_data, unique_products


if __name__ == "__main__":
    # Example usage
    try:
        cleaned_transactions, unique_products = process_transaction_data()
        print(f"Sample of unique products:\n{unique_products.head()}")
    except Exception as e:
        print(f"Error processing transaction data: {e}")
