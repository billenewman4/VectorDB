# Vector Embedding Module Implementation Plan

This document outlines the functions and classes that need to be implemented in the Clean_Code/Vector_Embedding module to handle all embedding-related functionality in the VectorDB project.

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
├── README.md                # Documentation
└── Instructions.md          # This file
```

## Functions and Classes to Implement

### 1. Base Embedder Interface (Models/base_embedder.py)

```python
class BaseEmbedder:
    """Base interface for all embedding models"""
    
    def name(self) -> str:
        """Return a unique identifier for this embedder"""
        pass
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        pass
    
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """Embed a list of text strings"""
        pass
```

### 2. OpenAI Embedder (Models/openai_embedder.py)

Implementation of the `OpenAIEmbedder` class from `src/VectorDB/OpenAIEmbedder.py` with these key methods:

- `__init__(api_key=None, model_name="text-embedding-3-small")`
- `name()` - Returns identifier for this embedder
- `embed_query(text)` - Embeds a single string
- `batch_embed(texts, batch_size=100, retry_count=3, retry_delay=5)` - Handles batch embedding with retries
- `__call__(input)` - Standard interface for embedding lists of texts

### 3. Local Embedder (Models/local_embedder.py)

Refactored version of `LocalEmbedder` from `src/VectorDB/localEmbedder.py` with these methods:

- `__init__(model_name="all-mpnet-base-v2")` - Default to the upgraded model
- `name()` - Returns identifier for this embedder
- `embed_query(text)` - Embeds a single string using sentence-transformers
- `__call__(input)` - Handles batch embedding with automatic batching

### 4. Batch Processing (Utilities/batch_processor.py)

```python
def process_in_batches(items: List[Any], processor_func: Callable, batch_size: int = 100, 
                    show_progress: bool = True) -> List[Any]:
    """Process a list of items in batches to manage memory and show progress."""
```

### 5. Vector Normalization (Utilities/normalization.py)

```python
def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length (L2 norm)"""
```

### 6. Main Embedding Pipeline (pipeline.py)

Implementation of high-level embedding functions:

#### 6.1 create_embedder

```python
def create_embedder(embedding_type: str = "sentence-transformer", 
                   model_name: Optional[str] = None, 
                   api_key: Optional[str] = None) -> BaseEmbedder:
    """Create and return an appropriate embedder based on the specified type."""
```

#### 6.2 embed_products

Refactored version of the `embed_products` function from `product_clustering/embed_products.py`:

```python
def embed_products(df: pd.DataFrame, 
                  text_col: str = 'clustering_description',
                  embedding_type: str = 'sentence-transformer',
                  model_name: Optional[str] = None,
                  batch_size: int = 100,
                  normalize_embeddings: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Generate embeddings for product descriptions."""
```

#### 6.3 generate_embeddings

Refactored version of the `generate_embeddings` function from `run_analysis.py`:

```python
def generate_embeddings(embedder: BaseEmbedder, 
                       df: pd.DataFrame, 
                       text_col: str = 'product_description',
                       batch_size: int = 100) -> pd.DataFrame:
    """Generate embeddings for all products and add them to the DataFrame."""
```

## Implementation Notes

- **Default Model**: Use 'all-mpnet-base-v2' as the default sentence-transformer model, which produces higher quality embeddings than the previously used 'all-MiniLM-L6-v2'.

- **Clustering Parameters**: When using these embeddings for clustering, optimal parameters are min_cluster_size=3 and min_samples=2 for more focused product groups.

- **OpenAI Models**: Support both 'text-embedding-ada-002' (legacy) and the newer 'text-embedding-3-small' models for OpenAI embeddings.

- **Error Handling**: Include robust error handling and retries, especially for API-based embedding generation.

- **Testing**: Include simple test functions at the bottom of each module file for easy validation.