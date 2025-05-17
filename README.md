# VectorDB

A vector database implementation for product matching using semantic similarity.

## Project Structure

The project has been organized into a clear directory structure:

```
VectorDB/
├── src/                     # Core implementation
│   ├── vectordb.py          # Main vector DB implementation
│   ├── data_processing.py   # Data processing pipeline
│   ├── excel.py             # Excel data processing utilities
│   └── abbreviation_translator.py # Meat cut abbreviation translation
├── tests/                   # Testing
│   ├── test_vectordb.py     # Vector DB functionality tests
│   └── test_excel.py        # Excel data processing tests
├── analysis/                # Analysis tools
│   ├── analyze_clusters.py           # Analyze problematic product clusters
│   ├── bidirectional_similarity.py   # Test similarity in both directions
│   └── evaluate_accuracy.py          # Evaluate overall matching accuracy
├── data/                    # Data files
│   ├── Transactions/                 # Transaction data
│   └── CorrectMapping/               # Ground truth mapping files
├── chroma_db/               # Vector database storage
├── README.md                # This documentation
├── instructions.txt         # Project requirements
└── requirements.txt         # Dependencies
```

## Tools and Analysis Scripts

- **analyze_clusters.py**: Tests specific product clusters with different similarity thresholds
- **bidirectional_similarity.py**: Investigates similarity asymmetry between product terms
- **evaluate_accuracy.py**: Detailed evaluation of matching accuracy for product descriptions
- **generate_best_usda_matches.py**: Generates best USDA code matches for products
- **summarize_usda_matches.py**: Produces detailed statistics on USDA matching accuracy

A project to match product descriptions using vector embeddings and similarity search.

## Project Overview

This project uses vector embeddings to match similar product descriptions based on semantic similarity. By converting product descriptions into vector embeddings and storing them in a vector database (Chroma DB), we can efficiently find similar products even when they use different wording or formatting.

## How It Works

### 1. Data Processing
- **Transaction Data**: Loads product transaction data from Excel files
- **Data Cleaning**: Standardizes text, handles missing values, and normalizes descriptions
- **Abbreviation Translation**: Converts industry-specific abbreviations to their full descriptions (e.g., "Bnls Rnd Stk" → "Boneless Round Steak")
- **USDA Code Preservation**: Maintains original USDA code formats throughout processing
- **Unique Products**: Extracts unique products and calculates metrics (avg price, quantity, etc.)

### 2. Vector Embedding
- **Model**: Uses Sentence Transformers (`all-mpnet-base-v2`) to convert text into high-dimensional vector embeddings
- **Enhanced Descriptions**: Combines product text with price and quantity data for richer embeddings
- **Example**: "almond milk, average price: $2.99, average quantity: 48.5" → [0.02, -0.04, ...]
- **Abbreviation Translation**: Automatically expands meat cut abbreviations (e.g., "Bnls" → "Boneless", "Rst" → "Roast") for improved semantic matching

### 3. Vector Database
- **Chroma DB**: Stores vectors in an efficient, searchable database
- **Similarity Search**: Finds products with similar vector representations
- **Matching**: Uses cosine similarity to rank product matches

### 4. Evaluation
- **Reference Mappings**: Compared against known product groupings
- **Metrics**: Precision, recall, and F1 score to measure accuracy
- **Performance**: Achieved 100% recall and 40% precision (F1 score of 0.58)

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
