# Vector Embedding Module

This module provides a clean, modular structure for embedding product descriptions in the VectorDB project, supporting both local and OpenAI-based embedding models.

## Directory Structure

```
Vector_Embedding/
├── Models/                   # Embedding model implementations
│   ├── openai_embedder.py   # OpenAI API-based embeddings
│   ├── local_embedder.py    # Sentence-transformers local embeddings
│   └── base_embedder.py     # Base class and interfaces
├── Utilities/                # Helper functions
│   ├── batch_processor.py    # Batch processing for embeddings
│   └── normalization.py      # Vector normalization utilities
├── pipeline.py              # Main embedding pipeline functions
├── README.md                # This file
└── Instructions.md          # Implementation plan
```

## Main Usage

The main entry point for product embedding is the `pipeline.py` file:

```python
from Clean_Code.Vector_Embedding.pipeline import embed_products

# Generate embeddings for products
embeddings_array, product_codes = embed_products(
    df=processed_data,
    text_col='clustering_description',
    embedding_type='sentence-transformer',  # or 'openai'
    normalize_embeddings=True
)
```

## Embedding Models

### Local Embeddings (Default)

By default, the module uses sentence-transformers with the `all-mpnet-base-v2` model, which produces higher quality embeddings than the previously used `all-MiniLM-L6-v2` model.

```python
from Clean_Code.Vector_Embedding.Models.local_embedder import LocalEmbedder

# Create a local embedder
embedder = LocalEmbedder(model_name='all-mpnet-base-v2')

# Embed a single text
embedding = embedder.embed_query("beef ribeye steak")

# Embed multiple texts
embeddings = embedder(["beef ribeye steak", "chicken breast"])
```

### OpenAI Embeddings

For higher quality embeddings, you can use OpenAI's embedding models:

```python
from Clean_Code.Vector_Embedding.Models.openai_embedder import OpenAIEmbedder

# Create an OpenAI embedder (requires API key in .env file or passed directly)
embedder = OpenAIEmbedder(model_name='text-embedding-3-small')

# Embed multiple texts with batching and retries
embeddings = embedder(["beef ribeye steak", "chicken breast"])
```

## Key Features

### Vector Normalization

Embeddings are normalized to unit length by default, which improves clustering results:

```python
from Clean_Code.Vector_Embedding.Utilities.normalization import normalize_vectors

# Normalize a matrix of embeddings
normalized_embeddings = normalize_vectors(embeddings_array)
```

### Batch Processing

Large datasets are automatically processed in batches to manage memory usage and show progress:

```python
from Clean_Code.Vector_Embedding.Utilities.batch_processor import process_in_batches

# Process a large list in batches
results = process_in_batches(
    items=texts,
    processor_func=embedder,
    batch_size=100,
    show_progress=True
)
```

### DataFrame Integration

Embeddings can be added directly to a pandas DataFrame:

```python
from Clean_Code.Vector_Embedding.pipeline import generate_embeddings

# Add embeddings to a DataFrame
df_with_embeddings = generate_embeddings(
    embedder=embedder,
    df=product_data,
    text_col='product_description',
    batch_size=100
)
```

## Design Decisions and Improvements

### Model Selection

The default model has been upgraded from `all-MiniLM-L6-v2` to `all-mpnet-base-v2`, which produces higher quality embeddings. This change supports more accurate product similarity calculations and better clustering results.

### Clustering Optimization

When using these embeddings for clustering with HDBSCAN:
- Use more granular clustering parameters: `min_cluster_size=3` and `min_samples=2`
- This creates more focused product groups that prevent mixing different product types

### Product Code Normalization

Previous issues with product code normalization have been addressed:
- Disabled potentially problematic normalization patterns (trailing number removal and first digit removal)
- These were causing products to match to incorrect USDA codes
- The embedding module preserves the exact product codes without normalization

### OpenAI Model Support

Added support for the newer `text-embedding-3-small` model (default) while maintaining backward compatibility with `text-embedding-ada-002`.

## Examples

### Complete Embedding Pipeline

```python
# Import necessary modules
from Clean_Code.Data_prep.prepare_data import prepare_data_for_clustering
from Clean_Code.Vector_Embedding.pipeline import embed_products

# Prepare data
prepared_data = prepare_data_for_clustering(
    normalize_text=True,
    expand_abbreviations=True,
    use_category_descriptions=True
)

# Generate embeddings for products by category
embeddings_array, product_codes = embed_products(
    df=prepared_data,
    text_col='clustering_description',
    embedding_type='sentence-transformer',
    normalize_embeddings=True
)

# Now embeddings_array can be used for clustering or similarity search
```