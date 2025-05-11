#!/usr/bin/env python3
"""
Test script for excel.py functions
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

# Import functions from excel.py
from src.excel import (
    load_transaction_data,
    clean_transaction_data,
    get_unique_products,
    load_mapping_data,
    process_transaction_data,
)

def test_load_transaction_data():
    """Test loading transaction data"""
    print("\n=== Testing load_transaction_data() ===")
    try:
        df = load_transaction_data()
        print(f"✅ Successfully loaded transaction data with {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data (first 3 rows):\n{df.head(3)}")
        return df
    except Exception as e:
        print(f"❌ Error loading transaction data: {e}")
        return None

def test_clean_transaction_data(df):
    """Test cleaning transaction data"""
    print("\n=== Testing clean_transaction_data() ===")
    if df is None:
        print("❌ Cannot test data cleaning - no input data")
        return None
        
    try:
        clean_df = clean_transaction_data(df)
        print(f"✅ Successfully cleaned data with {len(clean_df)} rows")
        print(f"Cleaned columns: {clean_df.columns.tolist()}")
        print(f"Sample cleaned data (first 3 rows):\n{clean_df.head(3)}")
        
        # Check for standardized column names
        lowercased_cols = all(col == col.lower() for col in clean_df.columns)
        print(f"All column names are lowercase: {lowercased_cols}")
        
        # Check product description cleaning
        if 'product_description' in clean_df.columns:
            has_null_desc = clean_df['product_description'].isnull().any()
            print(f"Product descriptions have null values: {has_null_desc}")
        
        return clean_df
    except Exception as e:
        print(f"❌ Error cleaning data: {e}")
        return None

def test_get_unique_products(df):
    """Test extracting unique products"""
    print("\n=== Testing get_unique_products() ===")
    if df is None:
        print("❌ Cannot test unique products extraction - no input data")
        return None
        
    try:
        unique_df = get_unique_products(df)
        print(f"✅ Successfully extracted {len(unique_df)} unique products")
        print(f"Unique products columns: {unique_df.columns.tolist()}")
        print(f"Sample unique products (first 5 rows):\n{unique_df.head(5)}")
        
        # Check for expected metrics columns
        expected_cols = ['transaction_count', 'avg_price', 'avg_quantity', 'total_quantity']
        missing_cols = [col for col in expected_cols if col not in unique_df.columns]
        if missing_cols:
            print(f"⚠️ Missing expected columns: {missing_cols}")
        else:
            print("✅ All expected metric columns present")
            
        return unique_df
    except Exception as e:
        print(f"❌ Error extracting unique products: {e}")
        return None

def test_load_mapping_data():
    """Test loading mapping data"""
    print("\n=== Testing load_mapping_data() ===")
    try:
        mapping_df = load_mapping_data()
        print(f"✅ Successfully loaded mapping data with {len(mapping_df)} rows")
        print(f"Mapping columns: {mapping_df.columns.tolist()}")
        print(f"Sample mapping data (first 3 rows):\n{mapping_df.head(3)}")
        return mapping_df
    except Exception as e:
        print(f"❌ Error loading mapping data: {e}")
        return None

def test_process_transaction_data():
    """Test the full processing pipeline"""
    print("\n=== Testing process_transaction_data() ===")
    try:
        cleaned_data, unique_products = process_transaction_data()
        print(f"✅ Successfully processed data via pipeline")
        print(f"Cleaned data: {len(cleaned_data)} rows")
        print(f"Unique products: {len(unique_products)} rows")
        print(f"Sample unique products (first 3 rows):\n{unique_products.head(3)}")
        return cleaned_data, unique_products
    except Exception as e:
        print(f"❌ Error in processing pipeline: {e}")
        return None, None

def main():
    """Run all tests"""
    print("Starting tests for excel.py functions...")
    
    # Test individual functions
    raw_df = test_load_transaction_data()
    clean_df = test_clean_transaction_data(raw_df)
    unique_df = test_get_unique_products(clean_df)
    mapping_df = test_load_mapping_data()
    
    # Test full pipeline
    test_process_transaction_data()
    
    print("\n=== Test Summary ===")
    print(f"Transaction data loaded: {'✅' if raw_df is not None else '❌'}")
    print(f"Data cleaning: {'✅' if clean_df is not None else '❌'}")
    print(f"Unique products extraction: {'✅' if unique_df is not None else '❌'}")
    print(f"Mapping data loaded: {'✅' if mapping_df is not None else '❌'}")
    
if __name__ == "__main__":
    main()
