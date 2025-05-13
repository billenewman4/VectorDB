# Configuration file for the VectorDB project

import os
from pathlib import Path

# --- Project Root ---
# Assuming this script is in VectorDB/src, the project root is its parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / "data"
ACTUALS_DATA_DIR = DATA_DIR / "Actuals"

# --- Transaction Data (Actuals) ---
TRANSACTION_REPORT_FILE = ACTUALS_DATA_DIR / "Transaction_Report_Actual.xlsx"
TRANSACTION_SHEET_NAME = "Sheet1"
TRANSACTION_PRODUCT_CODE_COL = "ProductCode" # Column linking to mapping file (indirectly)
TRANSACTION_DESC_COL = "ProductDescription" # Column to be embedded

# --- Ground Truth Mapping Data (Corrected Mapping) ---
GROUND_TRUTH_FILE = ACTUALS_DATA_DIR / "Corrected_mapping.xlsx"
GROUND_TRUTH_SHEET_NAME = "Sheet1"
# List of columns in the mapping file that might match the transaction ProductCode
# (Normalization needed: remove '-<number>')
GROUND_TRUTH_ID_COLS = [
    "Pritzlaff_code", 
    "Queen_code", 
    "Moesle_code", 
    "Anmar_code"
]
GROUND_TRUTH_USDA_COL = "USDA_Code" # The target value for evaluation

# --- Embedding Model ---
# Using a Sentence Transformer model suitable for product descriptions
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 
# Alternative: 'paraphrase-MiniLM-L6-v2'
# Alternative: 'multi-qa-MiniLM-L6-cos-v1' # Good for semantic search/QA

# --- Vector Database (ChromaDB) ---
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db_actuals" # New path for actuals data
COLLECTION_NAME = "actual_products"

# --- Similarity Search Parameters ---
# Default number of initial candidates to retrieve in forward search
N_RESULTS_INITIAL_SEARCH = 30
# Default number of final results after bidirectional check
N_RESULTS_FINAL = 5
# Default threshold for A->B similarity (initial search)
SIMILARITY_THRESHOLD_FORWARD = 0.25
# Default threshold for B->A similarity (direct calculation)
SIMILARITY_THRESHOLD_BACKWARD = 0.20

# --- Data Processing Parameters ---
# Minimum number of times a product must appear in transactions to be included
# Set to 1 if all unique products from the report should be included
MIN_TRANSACTION_COUNT = 1 

print(f"Config loaded. Project Root: {PROJECT_ROOT}")
