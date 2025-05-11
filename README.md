# VectorDB

A vector database implementation for product matching using semantic similarity.

## Project Structure

The project has been organized into a clear directory structure:

```
VectorDB/
├── src/                     # Core implementation
│   ├── vectordb.py          # Main vector DB implementation
│   └── excel.py             # Excel data processing utilities
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
- **evaluate_accuracy.py**: Detailed evaluation of matching accuracy for product descriptions Product Matching

A project to match product descriptions using vector embeddings and similarity search.

## Project Overview

This project uses vector embeddings to match similar product descriptions based on semantic similarity. By converting product descriptions into vector embeddings and storing them in a vector database (Chroma DB), we can efficiently find similar products even when they use different wording or formatting.

## How It Works

### 1. Data Processing
- **Transaction Data**: Loads product transaction data from Excel files
- **Data Cleaning**: Standardizes text, handles missing values, and normalizes descriptions
- **Unique Products**: Extracts unique products and calculates metrics (avg price, quantity, etc.)

### 2. Vector Embedding
- **Model**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) to convert text into 384-dimensional vector embeddings
- **Enhanced Descriptions**: Combines product text with price and quantity data for richer embeddings
- **Example**: "almond milk, average price: $2.99, average quantity: 48.5" → [0.02, -0.04, ...]

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
- **Alternative Models**: Try different embedding models for better precision
- **Weight Tuning**: Adjust the importance of price and quantity in embeddings
- **Domain Fine-tuning**: Train the model on product-specific data

## License

This project is licensed under the MIT License - see the LICENSE file for details.
