# Data Preparation Functions

This document lists all data preparation functions identified in the VectorDB codebase. Each function will be extracted into its own file in the Clean_Code/Data_prep folder to make the code more modular and maintainable.

## Text Processing Functions

1. **Text Normalization** (`text_normalization.py`)
   - `clean_text` - Basic text cleaning: lowercase, strip whitespace
     - Source: src/data_processing.py
   - `preprocess_text_for_clustering` - Enhanced preprocessing for product clustering
     - Source: product_clustering/data_prep.py

2. **Abbreviation Handling** (`abbreviation_handler.py`)
   - `get_abbreviation_map` - Returns a dictionary of abbreviation mappings
   - `expand_abbreviations` - Expands common food-related abbreviations in text
   - `expand_abbreviations_in_dataframe` - Expands abbreviations in specified DataFrame columns
     - Source: src/abbreviation_translator.py

## Data Loading Functions

6. `load_transaction_data` - Loads transaction data from Excel file
   - Source: src/data_processing.py
   - Description: Loads transaction data from the specified Excel file and sheet

7. `load_inventory_data` - Loads product category information from inventory reports
   - Source: src/data_processing.py
   - Description: Processes inventory files to extract category information

8. `_process_single_inventory_file` - Process a single inventory file to extract category information
   - Source: src/data_processing.py
   - Description: Helper function for load_inventory_data to process individual files

## Data Processing Functions

9. **Transaction Processing** (`transaction_processor.py`)
   - `process_transaction_data` - Processes transaction data to extract unique product descriptions
     - Source: src/data_processing.py
     - Description: Extracts and cleans unique product descriptions from transaction data

## Category Management Functions

10. **Category Operations** (`category_manager.py`)
    - `group_products_by_category` - Groups products by their category for category-based clustering
    - `save_category_products` - Saves category-to-products mapping for clustering processes
      - Source: src/data_processing.py

## Root Level Pipeline Functions

11. **Data Preparation Pipeline** (`prepare_data.py` - at root level)
    - `prepare_data_for_clustering` - Orchestrates the complete data preparation workflow
      - Source: product_clustering/data_prep.py
      - Description: Combines multiple data preparation steps for clustering applications

## Implementation Plan

Related functions will be grouped into single files with proper imports, docstrings, and type hints. The files will be organized in subdirectories based on their function category:

- Text_Processing/ - Functions for text normalization and manipulation
- Data_Loading/ - Functions for loading data from various sources
- Data_Processing/ - Functions for processing and transforming loaded data
- Category_Management/ - Functions for organizing products by category

Files will be named based on functionality, e.g., `text_normalization.py`, `abbreviation_handler.py`, `transaction_loader.py`, etc.

High-level orchestration functions will be placed at the root level of the Data_prep folder to serve as entry points and pipeline interfaces that combine the subfunctions.