# VectorDB Product Matching

A project to match product descriptions using vector embeddings and similarity search.

## Project Overview

This project uses vector embeddings to match product descriptions based on semantic similarity. By converting product descriptions into vector embeddings and storing them in a vector database (Chroma DB), we can efficiently find similar products.

## Features

- Data preprocessing and cleaning for product descriptions
- Vector embeddings generation using sentence-transformers
- Similarity matching using Chroma DB
- Evaluation metrics for matching accuracy

## Directory Structure

```
VectorDB/
├── CorrectMapping/            # Reference data for evaluation
│   └── product_mapping_semantic.xlsx
├── excel.py                   # Excel data processing
├── vectordb.py                # Vector database operations
├── evaluation.py              # Matching evaluation
└── instructions.txt           # Project tasks
```

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

[Usage instructions will be added as the project develops]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
