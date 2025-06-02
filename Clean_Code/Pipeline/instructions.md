# Hierarchical Clustering Pipeline Implementation Guide

## Overview

This document outlines the complete end-to-end process for implementing hierarchical product clustering with VectorDB. The pipeline integrates multiple components, providing flexibility to use either embedding-based clustering, cross-encoder-based clustering, or a combination of both at different hierarchical levels.

## Pipeline Steps

### 1. Data Preparation

- **Load Raw Product Data**: Use functions from `Data_Prep/Data_Loading` to load product data from source files.
- **Clean and Normalize Text**: Apply text normalization functions from `Data_Prep/Text_Processing` to standardize product descriptions.
- **Process Product Attributes**: Extract and normalize additional product attributes if needed.
- **Filter and Sample Data**: Optionally filter products by category or other attributes, and implement test mode for faster parameter tuning on subsets.

### 2. Product Embedding

- **Generate Embeddings**: Use the `Vector_Embedding` module with the improved `all-mpnet-base-v2` model to create high-quality embeddings.
- **Cache Embeddings**: Store generated embeddings to avoid redundant computation.
- **Preprocess Embeddings**: Normalize vectors, potentially reduce dimensions, and handle outliers.

### 3. Hierarchical Clustering Algorithm

- **Level 1 Clustering**: Perform initial clustering on all products, using either:
  - Embedding-based clustering (HDBSCAN with optimized parameters: min_cluster_size=3, min_samples=2)
  - Cross-encoder direct clustering (compute pairwise similarity matrix)
- **Cross-Encoder Refinement**: Optionally refine Level 1 clusters using cross-encoder to improve boundaries.
- **Level 2+ Clustering**: For each cluster from the previous level:
  - Extract member products
  - Apply clustering with potentially different parameters
  - Maintain parent-child relationships between clusters
  - Optionally apply cross-encoder refinement
- **Handle Edge Cases**: Address small clusters, identical vectors, and noise points appropriately.
- **Track Cluster Paths**: Maintain the hierarchical path for each product across all levels.

### 4. Configuration Management

- **Create Config File**: Use YAML configuration to control all aspects of the pipeline:
  - Data preparation options
  - Embedding model settings
  - Per-level clustering parameters (embedding vs. cross-encoder, refinement options)
  - Visualization settings
  - Output formats and locations
- **Parameter Overrides**: Allow command-line parameters to override config file settings for experimental runs.

### 5. Analysis and Visualization

- **Compute Metrics**: Calculate cluster quality metrics for each level.
- **Generate Visualizations**: Create 2D/3D visualizations of clusters at each level.
- **Export Results**: Convert clustering results to readable formats:
  - CSV files with cluster assignments and hierarchical paths and product descriptions

### 6. Testing and Validation

- **Test Different Configurations**: Validate pipeline with various combinations of:
  - Embedding-only clustering at all levels
  - Cross-encoder clustering at all levels
  - Mixed approaches (e.g., embedding at L1, cross-encoder at L2)
  - With and without refinement at each level
=- **Analyze Result Quality**: Review cluster coherence and separation for product groupings.

## Implementation Requirements

1. **Code Organization**: Maintain modularity with proper imports between components.
2. **Error Handling**: Implement robust error handling and logging throughout the pipeline. when in doubt throw an error rather than implement a work around. 
3. **Memory Efficiency**: Consider memory usage for large datasets, especially with cross-encoder processing.
4. **Documentation**: Include clear docstrings and comments explaining each component's purpose and function.
5. **Testing**: Create validation methods to verify output at each stage.

## Expected Outputs

1. **Cluster Assignments**: CSV file with columns for product ID, description, and cluster assignments at each level.
2. **Hierarchy Mapping**: JSON file showing the hierarchical relationships between clusters.
3. **Cluster Metadata**: Details about each cluster including size, centroid, and key characteristics.
4. **Visualizations**: Plots showing the clustering results at each hierarchical level.
5. **Performance Metrics**: Quality metrics for different clustering approaches.

## Usage

Run the pipeline using:

```bash
python main_hierarchical_clustering.py --config config.yaml
```

For testing specific configurations:

```bash
python main_hierarchical_clustering.py --config config.yaml --test_mode --levels 4 --l1_cross_encoder --l2_embedding --l3_cross_encoder --l4_embedding
```