#!/usr/bin/env python3
"""
Process Transaction Data Script

This script processes the Whetstone Product Costs Report to calculate average
pricing metrics for each SKU (ProductCode). It handles multiple transactions
per SKU by calculating the average values for:
- Implied GP$ (actual)
- Implied GP% (actual)
- SalesPrice
- AccountingCost

The results are saved to a CSV file for further analysis.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

def load_transaction_data(file_path: str) -> pd.DataFrame:
    """
    Load transaction data from the Excel file.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame with transaction data
    """
    try:
        # The "Product Costs" sheet contains the data we need
        df = pd.read_excel(file_path, sheet_name="Product Costs")
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading transaction data: {e}")
        return pd.DataFrame()

def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the dataframe has all required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = [
        "ProductCode", 
        "Implied GP$ (actual)", 
        "Implied GP% (actual)", 
        "SalesPrice", 
        "AccountingCost"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    return True, ""

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and ensuring correct data types.
    
    Args:
        df: Raw transaction data
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Ensure ProductCode is string
    processed_df["ProductCode"] = processed_df["ProductCode"].astype(str)
    
    # Handle missing or invalid values in numeric columns
    numeric_columns = [
        "Implied GP$ (actual)", 
        "Implied GP% (actual)", 
        "SalesPrice", 
        "AccountingCost"
    ]
    
    for col in numeric_columns:
        # Convert to numeric, errors become NaN
        processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")
    
    # Add a calculated GP$ column to verify against the reported one
    processed_df["Calculated GP$"] = processed_df["SalesPrice"] - processed_df["AccountingCost"]
    
    # Add a calculated GP% column to verify against the reported one
    # Avoid division by zero
    mask = processed_df["SalesPrice"] > 0
    processed_df["Calculated GP%"] = np.nan
    processed_df.loc[mask, "Calculated GP%"] = (processed_df.loc[mask, "Calculated GP$"] / 
                                                processed_df.loc[mask, "SalesPrice"])
    
    # Remove rows with missing essential values
    initial_count = len(processed_df)
    processed_df = processed_df.dropna(subset=["ProductCode", "SalesPrice", "AccountingCost"])
    dropped_count = initial_count - len(processed_df)
    
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with missing essential values")
    
    return processed_df

def calculate_sku_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average values for each SKU (ProductCode).
    
    Args:
        df: Preprocessed transaction data
        
    Returns:
        DataFrame with average values per SKU
    """
    # Make a copy of the ProductCode column for counting transactions
    df = df.copy()
    df['TransactionCount'] = 1
    
    # Columns to average
    agg_columns = {
        "Implied GP$ (actual)": "mean",
        "Implied GP% (actual)": "mean",
        "SalesPrice": "mean",
        "AccountingCost": "mean",
        "Calculated GP$": "mean",
        "Calculated GP%": "mean",
        "TransactionCount": "sum"  # This will give us the transaction count
    }
    
    # Group by ProductCode and calculate averages
    avg_df = df.groupby("ProductCode").agg(agg_columns).reset_index()
    
    # Add description column if available
    if "ProductDescription" in df.columns:
        # Get the most common description for each ProductCode
        desc_df = df.groupby("ProductCode")["ProductDescription"].agg(
            lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else ""
        ).reset_index()
        
        # Merge with the averages
        avg_df = pd.merge(avg_df, desc_df, on="ProductCode", how="left")
    
    # Add a column indicating the variance between reported and calculated values
    avg_df["GP$_Variance"] = avg_df["Implied GP$ (actual)"] - avg_df["Calculated GP$"]
    avg_df["GP%_Variance"] = avg_df["Implied GP% (actual)"] - avg_df["Calculated GP%"]
    
    return avg_df

def main():
    """Main workflow for processing transaction data."""
    
    # Set file paths
    input_file = "Whetstone Product Costs Report.xlsx"
    output_file = "product_pricing_averaged.csv"
    
    # Load transaction data
    df = load_transaction_data(input_file)
    
    if df.empty:
        print("Error: Failed to load transaction data. Exiting.")
        return
    
    # Validate data
    is_valid, error_message = validate_data(df)
    if not is_valid:
        print(f"Error: {error_message}")
        return
    
    # Preprocess data
    print("Preprocessing transaction data...")
    processed_df = preprocess_data(df)
    
    # Calculate SKU averages
    print("Calculating average values per SKU...")
    avg_df = calculate_sku_averages(processed_df)
    
    # Print some statistics
    print(f"\nProcessed {len(processed_df)} transactions for {len(avg_df)} unique SKUs")
    print(f"Average transactions per SKU: {processed_df['ProductCode'].value_counts().mean():.2f}")
    
    # Save to CSV
    avg_df.to_csv(output_file, index=False)
    print(f"Saved averaged pricing data to {output_file}")
    
    # Print sample of the output
    print("\nSample of averaged pricing data:")
    print(avg_df.head())

if __name__ == "__main__":
    main()
