# Data Preparation Module

This module provides a clean, modular structure for data preparation functions used in the VectorDB project, with a focus on product clustering and analysis.

## Directory Structure

```
Data_prep/
├── Text_Processing/       # Text normalization and abbreviation handling
├── Data_Loading/          # Functions to load transaction and inventory data
├── Data_Processing/       # Data transformation and processing functions
├── Category_Management/   # Functions for organizing products by category
├── prepare_data.py        # Main pipeline entry point
├── Instructions.md        # List of available functions
└── README.md              # This file
```

## Main Usage

The main entry point for data preparation is the `prepare_data.py` file at the root level. This orchestrates the entire data preparation workflow:

```python
from Clean_Code.Data_prep.prepare_data import prepare_data_for_clustering

# Prepare data with default settings
prepared_data = prepare_data_for_clustering()

# Or customize options
prepared_data = prepare_data_for_clustering(
    normalize_text=True,
    expand_abbreviations=True,
    use_category_descriptions=True,
    test_mode=False
)
```

## Key Features

### Improved Text Normalization

The text normalization process has been refined based on previous improvements:

- Uses simplified product descriptions without attribute extraction
- Preserves full product codes (does not remove trailing numbers or first digits)
- Includes options to expand common industry abbreviations

### Category-Based Organization

Products are organized by category to ensure similar products are grouped together:

- Products are grouped by their category from inventory data
- Each category is processed independently to prevent cross-category clustering
- Categories with fewer than 2 products are excluded from clustering

### Optimized for Current Embedding Model

The data preparation process is optimized for use with the current embedding model:

- Compatible with the all-mpnet-base-v2 embedding model
- Produces normalized text that works well with the embedding process
- Granular clustering settings (min_cluster_size=3, min_samples=2)

## Command Line Usage

You can also run the data preparation process from the command line:

```bash
python -m Clean_Code.Data_prep.prepare_data --test --sample-size 200
```

Available options:

- `--no-categories`: Don't filter products without category information
- `--no-normalize`: Don't normalize text descriptions
- `--no-expand-abbr`: Don't expand abbreviations in text normalization
- `--test`: Run in test mode with sample data
- `--sample-size SIZE`: Number of rows to use in test mode (default: 100)

## Adding New Functions

To add new data preparation functions:

1. Identify the appropriate subdirectory based on function type
2. Create a new file or add to an existing module
3. Update `Instructions.md` with the new function details
4. If needed, import the new function in `prepare_data.py`

## Important Notes

- **Product Code Normalization**: Previous issues were found with removing trailing numbers and first digits from product codes, which caused incorrect USDA mappings. These normalizations have been disabled.
- **Embedding Model**: The current embedding model is 'all-mpnet-base-v2', which produces higher quality embeddings than the previous 'all-MiniLM-L6-v2' model.
- **Conflict Warnings**: The system will warn about conflicts when multiple USDA mappings exist for the same product code.