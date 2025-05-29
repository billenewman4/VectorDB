# Product Clustering and Matching Solution

A comprehensive implementation for product semantic analysis, clustering, and USDA code mapping using advanced NLP techniques.

## Project Overview

This project solves two key challenges in retail product data management:

1. **Product Clustering**: Automatically groups similar products based on their semantic descriptions, ensuring that similar items (like different varieties of apples or cuts of beef) are grouped together.

2. **USDA Code Mapping**: Maps product descriptions to standardized USDA codes, enabling consistent categorization and analysis across different retailers and systems.

## High-Level Workflow

### Product Clustering Workflow

```
Transaction Data → Data Preparation → Category Grouping → Vector Embedding → HDBSCAN Clustering → CrossEncoder Refinement → Evaluation
```

1. **Data Preparation**: Clean and normalize product descriptions, including category and warehouse information
2. **Category Grouping**: Group products by their category description for hierarchical organization
3. **Embedding Generation**: Convert text descriptions to vector embeddings using a powerful neural model (all-mpnet-base-v2)
4. **Initial Clustering**: Group similar products within each category using HDBSCAN algorithm with optimized parameters
5. **CrossEncoder Refinement**: Apply pairwise similarity judgments to improve cluster coherence
6. **Evaluation**: Measure cluster quality using category-specific coherence metrics and manual inspection

### USDA Code Mapping Workflow

```
Transaction Data → Data Cleaning → Abbreviation Translation → Vector Embedding → Similarity Search → CrossEncoder Reranking
```

1. **Data Processing**: Clean and normalize product data, expanding abbreviations
2. **Vector Embedding**: Convert product descriptions to vector embeddings
3. **Vector Database**: Store embeddings in a ChromaDB vector database for efficient retrieval
4. **Similarity Search**: Find most similar USDA products based on semantic meaning
5. **Cross-Encoder Verification**: Re-rank matches using pairwise comparison for higher accuracy

## Key Improvements

### Embedding Model Upgrade
- Upgraded from 'all-MiniLM-L6-v2' to 'all-mpnet-base-v2'
- Provides higher quality vector representations for better semantic understanding
- Significantly improves clustering quality and USDA code matching accuracy
- Support for smaller 'all-MiniLM-L6-v2' model option for faster testing

### Hierarchical Category-Based Clustering
- **Category-Based Grouping**: Groups products by category before clustering, preventing mixing of unrelated products
- **Two-Level Organization**: Organizes products in a hierarchy (category → similarity cluster) for better organization
- **Category-Specific Parameters**: Allows fine-tuning clustering parameters for each product category
- **Enhanced Data Integration**: Incorporates product category and warehouse information from multiple sources

### Clustering Enhancements
- **Simplified Data Preparation**: Removed attribute extraction, using normalized product descriptions directly
- **Granular Clustering Parameters**: Optimized min_cluster_size=3 and min_samples=2 for more focused product groups
- **Test Mode**: Added capability to run on data subsets for faster parameter tuning
- **CrossEncoder Refinement**: Implemented pairwise similarity refinement to improve cluster coherence

### USDA Code Mapping Fixes
- **Improved Normalization Logic**: Disabled potentially problematic normalization patterns
  - Removed trailing number removal that caused incorrect mappings
  - Disabled first digit removal that created ambiguity
- **Conflict Detection**: Added warnings when multiple USDA mappings exist for the same product code

## Project Structure

The project has been organized into a modular structure:

```
VectorDB/
├── product_clustering/         # Product clustering implementation
│   ├── data_prep.py            # Data preparation for clustering
│   ├── embed_products.py       # Embedding generation
│   ├── clustering.py           # HDBSCAN clustering implementation
│   ├── improved_clustering.py  # Enhanced clustering with parameters
│   ├── reranking.py            # CrossEncoder refinement
│   ├── evaluation.py           # Cluster quality assessment
│   └── data/                   # Output directory for clustering results
├── src/                        # Core implementation
│   ├── VectorDB/               # Vector database modules
│   │   ├── vectordb.py         # Main vector DB implementation
│   │   ├── localEmbedder.py    # Local embedding generation
│   │   ├── OpenAIEmbedder.py   # OpenAI embedding integration
│   │   └── CrossEncoder.py     # CrossEncoder implementation
│   ├── data_processing.py      # Data processing pipeline
│   ├── excel.py                # Excel data processing utilities
│   └── abbreviation_translator.py # Meat cut abbreviation translation
├── data/                      # Data files
├── data_prep/                 # Data preparation modules
├── product_clustering/        # Product clustering module
├── analysis_results/          # Analysis outputs
├── chroma_db/                 # Vector database storage
├── requirements.txt           # Dependencies
└── instructions.txt           # Project requirements
```

## Detailed Implementation Steps

### 1. Product Clustering Implementation

The product clustering solution follows these detailed steps:

#### 1.1 Unified Data Preparation (`data_prep/processor.py`)
```python
python -m data_prep.processor
```
- Consolidated data preparation module that handles all preprocessing needs
- Loads transaction data and inventory valuation files from multiple warehouses
- Integrates product codes, descriptions, categories, and warehouse information
- Normalizes text and expands abbreviations using the built-in abbreviation translator
- Produces a unified product dataset with all relevant attributes
- Saves prepared data to CSV for subsequent stages

#### 1.2 Embedding Generation (`embed_products.py`)
```python
python -m product_clustering.embed_products
```
- Utilizes the high-quality 'all-mpnet-base-v2' embedding model (upgraded from 'all-MiniLM-L6-v2')
- Generates 768-dimensional vector embeddings for each product description
- Uses existing embedding infrastructure (LocalEmbedder) with batch processing
- Saves embeddings and product codes for clustering

#### 1.2.1 Category Filtering (`data_prep/category_filter.py`)
```python
python -m data_prep.category_filter
```
- Filters products to include only those with valid category descriptions
- Normalizes category names for consistent grouping
- Groups products by category to enable hierarchical clustering
- Saves category-product mappings for the clustering stage

#### 1.3 Clustering Options

##### 1.3.1 Standard Clustering (`product_clustering/improved_clustering.py`)
```python
python -m product_clustering.improved_clustering --min_cluster_size 3 --min_samples 2
```
- Implements HDBSCAN clustering algorithm optimized for product descriptions
- Configurable parameters to control cluster granularity:
  - min_cluster_size=3 (reduced from 5 for more granular clusters)
  - min_samples=2 (reduced from 3 for more focused product groupings)
- These optimized parameters work better with 'all-mpnet-base-v2' embeddings
- Test mode for rapid experimentation on data subsets:
  ```python
  python -m product_clustering.improved_clustering --test --sample_size 1000
  ```
- Produces cluster assignments, statistics, and visualization outputs

##### 1.3.2 Category-Based Clustering (`product_clustering/category_clustering.py`)
```python
python -m product_clustering.run_clustering --use_category_clustering
```
- Hierarchical clustering approach that first groups products by category
- Prevents irrelevant product types from being mixed in the same cluster
- Creates embeddings and clusters for each product category separately
- Configurable parameters through `src/config.py`:
  - `USE_CATEGORY_CLUSTERING`: Toggle for category-based clustering
  - `MIN_CLUSTER_SIZE`: Minimum number of products to form a cluster
  - `MIN_SAMPLES`: HDBSCAN density parameter
  - `CLUSTERING_METRIC`: Distance metric for clustering (cosine, euclidean, etc.)
- Test mode for faster development and validation:
  ```python
  python -m product_clustering.test_category_clustering --test_size 10
  ```
- Produces hierarchical cluster assignments organized by category

#### 1.4 CrossEncoder Refinement (`reranking.py`)
```python
python -m product_clustering.run_clustering --rerank
```
- Optional refinement step using CrossEncoder for pairwise similarity judgments
- Supports both standard clustering and category-based clustering approaches
- More precise than embedding similarity for distinguishing similar but distinct products
- Particularly effective at separating mixed product clusters (e.g., bananas and lettuce)
- Configurable similarity threshold through `src/config.py`:
  - `USE_RERANKING`: Toggle to enable/disable reranking
  - `CROSS_ENCODER_MODEL`: Model to use for pairwise similarity scoring
  - `SIMILARITY_THRESHOLD`: Minimum similarity score (default 0.6) to consider products in the same cluster
- Creates refined clusters with improved coherence, preserving category hierarchy when using category-based clustering

#### 1.5 Evaluation (`evaluation.py`)
```python
python -m product_clustering.evaluation --all
```
- Comprehensive assessment of cluster quality
- Calculates cluster coherence scores using cosine similarity
- Identifies "good" clusters based on size and coherence thresholds
- Generates visualizations (coherence histogram, cluster size distribution)
- Displays sample products from clusters for manual inspection

### 2. USDA Code Mapping Implementation

The USDA code mapping solution implements these detailed steps:

#### 2.1 Data Processing (`data_processing.py`)
```python
from src.data_processing import process_transaction_data
```
- Loads transaction data from Excel files
- Performs data cleaning and normalization
- Extracts unique product codes and descriptions
- Fixed code normalization logic to prevent incorrect USDA mappings:
  - Disabled trailing number removal
  - Disabled first digit removal patterns
  - Added conflict warnings for multiple mappings

#### 2.2 Vector Embedding (`VectorDB/localEmbedder.py` and `VectorDB/OpenAIEmbedder.py`)
```python
from src.VectorDB.localEmbedder import LocalEmbedder
```
- Uses 'all-mpnet-base-v2' model for high-quality embeddings
- Generates vector representations of product descriptions
- Supports both local sentence-transformers and OpenAI embeddings
- Includes batch processing for efficient embedding generation

#### 2.3 Vector Database (`VectorDB/vectordb.py`)
```python
from src.VectorDB.vectordb import create_product_vector_db, find_similar_products
```
- Builds vector database using ChromaDB
- Stores embeddings with associated product metadata
- Implements similarity search for efficient product matching
- Creates mapping from product codes to USDA codes

#### 2.4 Cross-Encoder Verification (`VectorDB/CrossEncoder.py`)
```python
from src.VectorDB.CrossEncoder import CrossEncoder
```
- Provides more accurate similarity scores for candidate matches
- Re-ranks initial matches using pairwise comparison
- Combines embedding similarity with cross-encoder scores
- Especially useful for boundary cases and ambiguous matches

## Quick Start Guide

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Product Clustering

```bash
# Step 1: Prepare data
python -m product_clustering.data_prep

# Step 2: Generate embeddings
python -m product_clustering.embed_products

# Step 3: Run basic clustering
python -m product_clustering.clustering

# OR run improved clustering with granular parameters
python -m product_clustering.improved_clustering --min_cluster_size 3 --min_samples 2

# Step 4: Add CrossEncoder refinement (optional)
python -m product_clustering.improved_clustering --rerank

# Step 5: Evaluate clusters
python -m product_clustering.evaluation --all
```

### USDA Code Mapping

```python
# Create and query the vector database
from src.VectorDB.vectordb import create_product_vector_db, find_similar_products

# Create/rebuild the database with the improved embedding model
db = create_product_vector_db(recreate=True)

# Find similar products using semantic search
results = find_similar_products("boneless chicken breast", n_results=5)
print(results)
```

## Implementation Details

### Embedding Model
- Uses 'all-mpnet-base-v2' model (upgraded from 'all-MiniLM-L6-v2')
- Generates 768-dimensional vector embeddings
- Provides higher quality semantic representations
- Enables more accurate clustering and matching

### HDBSCAN Clustering Configuration
- min_cluster_size=3 (default was 5)
- min_samples=2 (default was 3)
- metric='euclidean'
- These parameters create more granular, focused product groups
- Particularly effective at separating similar but distinct product types

### CrossEncoder Refinement
- Uses 'cross-encoder/stsb-roberta-base' model
- Performs pairwise similarity judgments within clusters
- Similarity threshold of 0.6 for determining membership
- Removes products that don't belong in a cluster

## Example Results

### Before Improvement (Mixed Produce Cluster)
```
50120136,30,cluster_30,0.9555,bananas 24ct green,bananas,24ct
50133,30,cluster_30,0.9688,lettuce iceburg cello 24 ct,lettuce,24 ct
50133109,30,cluster_30,0.9756,celery 30ct,celery,30ct
50120145,30,cluster_30,0.9555,bananas green plaintain 40#,bananas,
50134291,30,cluster_30,1.0000,lettuce iceburg liner premium 12ct,lettuce,12ct
```

### After CrossEncoder Refinement (Example Clusters)
```
# Beef Rib/Short Rib Cluster
1016230: bf short rib pla-bi me lft
1072106: bf short rib-chu-bi-z pr wre
10049998: bf ribeye bi hvy pr ibp

# Turkey Products Cluster
11071: smoked turkey necks frzn
341000: turkey thigh meat, b/s
84035810: smoked necks, turkey alex deli

# Bacon Products Cluster
13435D: bacon-z$ daily sl14-16 37435 lf 15#
13832015: bacon thm-pre-cooked-z #12500
13879C: bacon celeb slab derind 4/7#-z
```

## Real-World Examples and Results

### Clustering Results Examples

#### Before: Mixed Produce Cluster (Original Clustering)
```
50120136: BANANAS 24CT GREEN
50133: LETTUCE ICEBURG CELLO 24 CT
50133109: CELERY 30CT
50120145: BANANAS GREEN PLAINTAIN 40#
50134291: LETTUCE ICEBURG LINER PREMIUM 12CT
```

#### After: Focused Product Clusters (With CrossEncoder Refinement)

```
# Beef Rib/Short Rib Cluster
1016230: BF SHORT RIB PLA-BI ME LFT
1072106: BF SHORT RIB-CHU-BI-Z PR WRE
10049998: BF RIBEYE BI HVY PR IBP

# Frenched Pork Chops Cluster
240089: PORK CHOP FRENCHED, 8OZ
240102: PORK CHOP FRENCHED, 10OZ
240077: PORK CHOP FRENCHED, 7OZ
240094: PORK CHOP FRENCHED, 9OZ

# Veal Heads Cluster
1122231: VEAL HEAD-Z CAT 2CT
112223C: VEAL HEAD-Z CAT 3CT
1122242: VEAL HEAD-Z PRO 3CT
68897000: VEAL HEADS
30149980: VEAL, HEADS 3PC
```

### Performance Metrics

#### Clustering Quality
- **Cluster Size Distribution**:
  - 1-5 products: 35.7% of clusters
  - 6-10 products: 38.7% of clusters
  - 11-20 products: 19.6% of clusters
  - 21+ products: 6.0% of clusters
- **Coherence**: Average coherence score of 0.83 (scale 0-1)
- **Precision**: CrossEncoder refinement significantly improves cluster precision

#### USDA Code Mapping
- **Recall**: Near 100% - finds all relevant product matches
- **Precision**: ~40% - some false positives, but much better than baseline
- **F1 Score**: 0.58
- **Response Time**: ~0.011 seconds per query

## Challenges and Solutions

### Challenges Addressed
1. **Mixed Product Types**: Initial clusters contained unrelated products (e.g., bananas with lettuce)
   - *Solution*: CrossEncoder reranking to refine clusters based on pairwise similarity

2. **Incorrect USDA Mappings**: Product codes were being incorrectly normalized
   - *Solution*: Disabled problematic normalization patterns and added conflict warnings

3. **Performance Issues**: Initial embedding model wasn't providing sufficient quality
   - *Solution*: Upgraded to more powerful 'all-mpnet-base-v2' model

## Directory Structure

```
VectorDB/
├── CorrectMapping/            # Reference data for evaluation
│   └── product_mapping_semantic.xlsx
├── Transactions/              # Transaction data
│   └── product_transactions_semantic.xlsx
├── chroma_db/                 # Vector database storage
├── excel.py                   # Excel data processing
├── test_excel.py              # Tests for Excel functions
├── vectordb.py                # Vector embedding and database operations
├── test_vectordb.py           # Tests for vector database
├── evaluate_accuracy.py       # Evaluation against reference mappings
├── requirements.txt           # Python dependencies
└── instructions.txt           # Project tasks
```

## Technical Implementation

### Vector Embedding Process

```python
# The embedding process takes a product description and its metrics:
 def create_text_description(self, row: pd.Series) -> str:
    # Extract product description and metrics
    product_desc = row['product_description']
    
    # Format price and quantity info if available
    price_info = f", average price: ${row['avg_price']:.2f}" if 'avg_price' in row else ""
    qty_info = f", average quantity: {row['avg_quantity']:.1f}" if 'avg_quantity' in row else ""
    
    # Create enhanced description
    enhanced_desc = f"{product_desc}{price_info}{qty_info}"
    return enhanced_desc
```

### Similarity Search

When searching for similar products, the system:
1. Converts the query into a vector using the same model
2. Finds the closest vectors in the database using cosine similarity
3. Returns products ranked by similarity score

### Performance Analysis

Our evaluation showed:
- **High Recall (100%)**: Found all products that should match
- **Moderate Precision (40%)**: Returned some irrelevant matches
- **Response Time**: ~0.011 seconds per query

This pattern is common in search systems - the system is "generous" in what it considers similar, finding all relevant matches but including some false positives.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VectorDB.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Processing

```python
# Import the processing pipeline
from excel import process_transaction_data

# Process transaction data
cleaned_data, unique_products = process_transaction_data()

# View unique products
print(unique_products.head())
```

### Creating and Querying the Vector Database

```python
# Import vector database functions
from vectordb import create_product_vector_db, find_similar_products

# Create the vector database
vector_db = create_product_vector_db()

# Find similar products
results = find_similar_products("almond milk", n_results=5)
print(results[["similarity", "product_description", "avg_price"]])
```

### Running Tests

```bash
# Run Excel processing tests
python test_excel.py

# Run vector database tests
python test_vectordb.py

# Evaluate accuracy against reference mappings
python evaluate_accuracy.py
```

## Project Status

- ✅ GitHub repository created
- ✅ Transaction data loading and processing implemented
- ✅ Vector embeddings generation
- ✅ Vector database creation and similarity search
- ✅ Evaluation against reference mappings

## Potential Improvements

- **Similarity Threshold**: Implement a cutoff to reduce false positives
- **Expanded Abbreviation Dictionary**: Add more industry-specific abbreviations to the translation system
- **Weight Tuning**: Adjust the importance of price and quantity in embeddings
- **Domain Fine-tuning**: Train the model on product-specific data

## Recent Updates

- **Enhanced Embedding Model**: Upgraded from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` for higher quality embeddings
- **Meat Cut Abbreviation Translation**: Added a system to translate meat industry abbreviations to full descriptions
- **USDA Code Format Preservation**: Modified processing to maintain original USDA code formats
- **Summary Statistics Tool**: Added detailed analysis of USDA code mapping accuracy

## Detailed Workflow

This section explains the end-to-end workflow of the VectorDB system, from data processing to analysis.

### 1. Data Processing Pipeline

```
Transaction Data → Data Cleaning → Abbreviation Translation → USDA Code Mapping → Vector Embedding → Vector Database
```

#### Function Call Sequence:

1. `data_processing.load_transaction_data()` - Loads raw transaction data from Excel
2. `data_processing.process_transaction_data()` - Cleans data and expands abbreviations
3. `vectordb.build_usda_lookup()` - Creates mapping from product codes to USDA codes
4. `vectordb.create_product_vector_db()` - Main function that orchestrates the entire process
   - Initializes the sentence transformer model
   - Embeds product descriptions
   - Creates and populates the vector database

### 2. Running the System

To rebuild the vector database from scratch:

```python
# From Python
from src.vectordb import create_product_vector_db
create_product_vector_db(recreate=True)

# Or from the command line
python -c "from src.vectordb import create_product_vector_db; create_product_vector_db(recreate=True)"
```

To load an existing database (without recreating):

```python
from src.vectordb import create_product_vector_db
vector_db, products_df = create_product_vector_db(recreate=False)
```

### 3. Analysis Workflow

After the vector database is created, you can run various analyses:

#### Generate Best USDA Matches for Products

```bash
python analysis/generate_best_usda_matches.py
```

This script:
1. Loads the vector database
2. For each product, finds the best matching USDA code based on embedding similarity
3. Generates Excel and CSV reports in the `analysis_results` directory
4. Calculates accuracy for products with known USDA mappings

#### Analyze USDA Matching Statistics

```bash
python analysis/summarize_usda_matches.py
```

This script:
1. Loads the USDA matching results from `analysis_results/best_usda_matches.csv`
2. Calculates detailed statistics about matching accuracy
3. Analyzes performance by similarity threshold
4. Identifies common mismatches and patterns
5. Generates a detailed report at `analysis_results/usda_mapping_report.md`

#### Other Analysis Tools

- **Bidirectional Similarity Test**: `python analysis/run_bidirectional_test.py`
- **Debug Single Query**: `python analysis/debug_single_query_bidirectional.py [product_code]`
- **Cluster Analysis**: `python analysis/analyze_clusters.py`

### 4. Complete Processing Chain Example

To run the complete process from data loading to analysis:

```bash
# 1. Recreate the vector database with expanded abbreviations
python -c "from src.vectordb import create_product_vector_db; create_product_vector_db(recreate=True)"

# 2. Generate best USDA matches for all products
python analysis/generate_best_usda_matches.py

# 3. Analyze the matching results
python analysis/summarize_usda_matches.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
