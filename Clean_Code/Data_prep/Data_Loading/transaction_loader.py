"""
Transaction data loading module.

This module provides functions for loading transaction data from Excel files 
and performing basic validations.
"""

import os
import sys
import pandas as pd
from typing import Optional

# Add parent directories to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define a simple config with default values for this module
class Config:
    """
    Configuration class with default values for data loading.
    """
    # Default paths - point to the Source_data folder where actual transaction data is stored
    SOURCE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "Source_data")
    TRANSACTION_REPORT_FILE = os.path.join(SOURCE_DATA_DIR, "Actuals", "Transaction_Report_Actual.xlsx")
    TRANSACTION_SHEET_NAME = "Sheet1"  # Default sheet name, will be overridden if provided

config = Config()


def load_transaction_data(file_path: Optional[str] = None, 
                         sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Loads transaction data from the specified Excel file and sheet.
    
    Args:
        file_path: Path to the transaction Excel file. 
                  If None, uses the path from config.
        sheet_name: Name of the sheet containing transaction data.
                   If None, uses the name from config.
                   
    Returns:
        DataFrame containing transaction data, or None if loading fails.
    """
    # Use config values if parameters not provided
    if file_path is None:
        file_path = config.TRANSACTION_REPORT_FILE
    if sheet_name is None:
        sheet_name = config.TRANSACTION_SHEET_NAME
    
    print(f"Loading transaction data from: {file_path}, Sheet: {sheet_name}")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: Transaction file not found at {file_path}")
        return None
    except Exception as e:  # Catch other potential errors like sheet not found
        print(f"Error loading transaction data: {e}")
        return None


if __name__ == "__main__":
    # Test the transaction data loading
    transactions = load_transaction_data()
    
    if transactions is not None:
        print("\nSample of loaded transaction data:")
        print(transactions.head())
        
        print("\nTransaction data columns:")
        for i, col in enumerate(transactions.columns):
            print(f"{i+1}. {col}")
