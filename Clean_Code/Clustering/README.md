# Clustering Module

## Overview

The Clustering module is a comprehensive solution for identifying meaningful groups within product descriptions using embedding vectors. It offers flexible algorithms, evaluation metrics, and visualization tools to generate high-quality product clusters. The module can work independently or integrate with the Vector_Embedding and Cross_Encoder modules for enhanced performance.

## Key Features

- **Multiple clustering algorithms** including HDBSCAN (density-based) and hierarchical clustering
- **Robust evaluation metrics** to assess cluster quality (silhouette score, Davies-Bouldin index, etc.)
- **Visualization tools** for cluster analysis using t-SNE, UMAP, and PCA
- **Cross-encoder integration** to refine cluster boundaries and improve coherence
- **Comprehensive pipeline** with simple high-level functions for common workflows
- **Configurable parameters** with sensible defaults for different use cases
- **Memory-efficient processing** for handling large datasets

## Directory Structure

```
Clean_Code/Clustering/
├── Models/                    # Core clustering implementations
├── Utilities/                 # Support functions and tools
├── pipeline.py               # Main clustering pipeline functions
├── config.py                 # Configuration parameters
├── README.md                 # This documentation file
└── test_clustering.py        # Comprehensive tests
```

## Quick Start

### Basic Clustering with HDBSCAN

```python
from Clustering.pipeline import cluster_embeddings
from Vector_Embedding.pipeline import generate_embeddings

# Generate embeddings for product descriptions
texts = [
    "Organic Fuji Apple",
    "Red Delicious Apple",
    "Granny Smith Apple",
    "Yellow Banana",
    "Plantain Banana",
    "Green Plantain"
]

embeddings = generate_embeddings(texts)

# Cluster the embeddings using HDBSCAN
clusters = cluster_embeddings(
    vectors=embeddings,
    method='hdbscan',
    min_cluster_size=2,
    min_samples=1
)

# Access cluster assignments
for i, label in enumerate(clusters['labels']):
    print(f"{texts[i]}: Cluster {label}")

# Get cluster metrics
print(f"Silhouette score: {clusters['metrics']['silhouette_score']:.4f}")
```

### Enhanced Clustering with Cross-Encoder

```python
from Clustering.pipeline import cluster_embeddings, enhance_clusters_with_cross_encoder
from Cross_Encoder.pipeline import create_reranker

# Generate initial clusters
clusters = cluster_embeddings(embeddings)

# Create a cross-encoder reranker
reranker = create_reranker()

# Enhance clusters using cross-encoder scoring
enhanced_clusters = enhance_clusters_with_cross_encoder(
    clusters=clusters,
    texts=texts,
    reranker=reranker
)

# Compare results
print("Original clusters:")
print(clusters['labels'])
print("Enhanced clusters:")
print(enhanced_clusters['labels'])
```

### Visualization

```python
from Clustering.Utilities.visualization import plot_clusters

# Create a 2D visualization using t-SNE
plot_clusters(clusters, texts=texts, dim_reduction='tsne')

# Create a 2D visualization using UMAP
plot_clusters(clusters, texts=texts, dim_reduction='umap')
```

## Components

### Clustering Algorithms

#### HDBSCAN (Default)

HDBSCAN is particularly well-suited for embedding vectors because it:
- Discovers clusters of varying densities and shapes
- Identifies outliers as noise points
- Doesn't require specifying the number of clusters in advance
- Works well with high-dimensional data

```python
from Clustering.Models.hdbscan_clusterer import HdbscanClusterer

clusterer = HdbscanClusterer(
    min_cluster_size=3,  # Minimum points to form a cluster
    min_samples=2,       # Minimum points to be considered a core point
    metric='cosine',     # Distance metric (cosine recommended for embeddings)
    cluster_selection_method='eom'  # Excess of Mass for better small clusters
)

# Fit and predict
results = clusterer.fit_predict(embeddings)
```

#### Hierarchical Clustering

Useful for creating multi-level category structures:

```python
from Clustering.Models.hierarchical import HierarchicalClusterer

clusterer = HierarchicalClusterer(
    n_clusters=5,        # Number of top-level clusters
    affinity='cosine',   # Similarity metric
    linkage='average'    # Linkage criterion
)

# Fit and predict
results = clusterer.fit_predict(embeddings)

# Get hierarchical structure
hierarchy = clusterer.get_hierarchy()
```

### Evaluation Metrics

The module provides several metrics to evaluate cluster quality:

```python
from Clustering.Utilities.metrics import evaluate_clusters

metrics = evaluate_clusters(
    vectors=embeddings,
    labels=clusters['labels'],
    metric='cosine'
)

print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
print(f"Davies-Bouldin index: {metrics['davies_bouldin_index']:.4f}")
print(f"Calinski-Harabasz index: {metrics['calinski_harabasz_index']:.4f}")
```

When ground truth labels are available, you can use external validation:

```python
from Clustering.Utilities.metrics import evaluate_with_ground_truth

ground_truth = [0, 0, 0, 1, 1, 1]  # True labels
results = evaluate_with_ground_truth(
    predicted_labels=clusters['labels'],
    true_labels=ground_truth
)

print(f"Adjusted Rand Index: {results['adjusted_rand_index']:.4f}")
print(f"Normalized Mutual Information: {results['normalized_mutual_info']:.4f}")
```

### Cross-Encoder Integration

Cross-encoders can significantly improve clustering quality by providing more accurate pairwise similarity scores:

```python
from Clustering.pipeline import enhance_clusters_with_cross_encoder
from Cross_Encoder.pipeline import create_reranker

# Create a reranker
reranker = create_reranker()

# Enhance existing clusters
enhanced_clusters = enhance_clusters_with_cross_encoder(
    clusters=clusters,
    texts=texts,
    reranker=reranker,
    mode='refine_boundaries'  # Options: 'refine_boundaries', 'rescore_clusters', 'hybrid'
)
```

## Configuration

The module includes a configuration system with sensible defaults:

```python
from Clustering.config import get_default_config, save_config, load_config

# Get default configuration
config = get_default_config()

# Modify configuration
config['hdbscan']['min_cluster_size'] = 3
config['hdbscan']['min_samples'] = 2

# Save configuration
save_config(config, 'my_clustering_config.json')

# Load configuration
config = load_config('my_clustering_config.json')

# Use configuration
from Clustering.pipeline import cluster_embeddings
clusters = cluster_embeddings(embeddings, config=config)
```

## Advanced Usage

### Custom Distance Metrics

```python
from Clustering.Models.hdbscan_clusterer import HdbscanClusterer
import numpy as np

# Define a custom distance function
def weighted_cosine_distance(x, y, weights=None):
    if weights is None:
        weights = np.ones(x.shape)
    return 1.0 - np.sum(weights * x * y) / (
        np.sqrt(np.sum(weights * x * x)) * 
        np.sqrt(np.sum(weights * y * y))
    )

# Create clusterer with custom metric
clusterer = HdbscanClusterer(
    min_cluster_size=3,
    min_samples=2,
    metric=weighted_cosine_distance
)
```

### Processing Large Datasets

```python
from Clustering.pipeline import cluster_embeddings_in_batches

# Process a large dataset in memory-efficient batches
clusters = cluster_embeddings_in_batches(
    vectors=large_embeddings,
    batch_size=10000,
    method='hdbscan',
    min_cluster_size=5
)
```

### Saving and Loading Results

```python
from Clustering.pipeline import save_clusters, load_clusters

# Save clustering results
save_clusters(clusters, 'product_clusters.json')

# Load clustering results
clusters = load_clusters('product_clusters.json')
```

## Performance Considerations

- HDBSCAN scales approximately O(n log n) with the number of data points
- Pre-computing distance matrices can speed up repeated clustering runs
- For very large datasets (>100k points), consider:
  - Downsampling the dataset
  - Using approximate nearest neighbors
  - Processing in batches

## Integration with Other Modules

The Clustering module is designed to work seamlessly with other VectorDB modules:

- **Vector_Embedding module**: Provides input embeddings for clustering
- **Cross_Encoder module**: Enhances similarity scoring for better clusters
- **Database_Storage module**: Stores and indexes clustering results

## Future Improvements

- Incremental clustering for online learning
- Semi-supervised clustering with constraints
- Distributed clustering for very large datasets
- Automated hyperparameter tuning
- Interactive visualization dashboard

## Troubleshooting

### Common Issues

1. **No clusters found**: Try reducing `min_cluster_size` and `min_samples` parameters
2. **Too many small clusters**: Increase `min_cluster_size` or adjust `cluster_selection_method`
3. **Poor cluster quality**: Consider using cross-encoder enhancement or different distance metrics
4. **Memory errors**: Use batch processing or reduce dimensionality of embeddings

## References

- HDBSCAN: [https://hdbscan.readthedocs.io/](https://hdbscan.readthedocs.io/)
- Clustering evaluation metrics: [https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- t-SNE: [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- UMAP: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
