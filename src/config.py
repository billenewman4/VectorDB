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
    "Fulton_code",
    "Pritzlaff_code", 
    "Queen_code", 
    "Moesle_code", 
    "Anmar_code"
]
GROUND_TRUTH_USDA_COL = "USDA_Code" # The target value for evaluation

# --- Embedding Model Configuration ---
# Choose which embedding model to use: 'openai', 'sentence-transformer'
EMBEDDING_TYPE = 'sentence-transformer'  # Default to sentence-transformer for easier local testing

# SentenceTransformer model options
# Options: 'all-MiniLM-L6-v2' (baseline, faster for testing), 'all-mpnet-base-v2' (better quality for production)
SENTENCE_TRANSFORMER_MODEL = 'all-mpnet-base-v2'  # Default to our current best model
SENTENCE_TRANSFORMER_MODEL_TESTING = 'all-MiniLM-L6-v2'  # Faster model for testing
EMBEDDING_MODEL = SENTENCE_TRANSFORMER_MODEL  # For backward compatibility

# OpenAI embedding model options
# Options: 'text-embedding-3-small', 'text-embedding-3-large'
OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'

# OpenAI API configuration
OPENAI_API_KEY = ''  # Set your API key here or use environment variable

# --- Vector Database (ChromaDB) ---
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db_actuals" # New path for actuals data

# --- Category-Based Clustering Configuration ---
# Whether to use category-based clustering by default
USE_CATEGORY_CLUSTERING = True

# Clustering parameters
CLUSTERING_METRIC = "euclidean"  # Distance metric for HDBSCAN
MIN_CLUSTER_SIZE = 2  # Minimum size of clusters (reduced from 3 for better coverage)
MIN_SAMPLES = 2  # HDBSCAN min_samples parameter

# Re-ranking parameters
USE_RERANKING = True  # Whether to use CrossEncoder reranking by default
CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-base"  # Model for reranking
SIMILARITY_THRESHOLD = 0.6  # Threshold for reranking

# Testing parameters
TESTING_MODE = False  # Whether to run in testing mode by default
TEST_SAMPLE_SIZE = 20  # Number of products per category to use for testing

# Use different collection names for different embedding types to avoid conflicts
if EMBEDDING_TYPE == 'openai':
    COLLECTION_NAME = f"actual_products_{OPENAI_EMBEDDING_MODEL}"
else:
    COLLECTION_NAME = f"actual_products_{SENTENCE_TRANSFORMER_MODEL.replace('-', '_')}"

print(f"Using collection name: {COLLECTION_NAME}")

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
