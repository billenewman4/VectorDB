# Hierarchical Multi-Level Product Clustering

This module implements a progressive, hierarchical clustering approach that groups products at multiple levels of granularity, starting with broad categories (e.g., meat, fish, vegetables) and progressively refining to more specific groups (e.g., beef → ground beef → frozen ground beef).

## Files and Structure

- **hierarchical_config.json**: Configuration file defining the parameters for each clustering level
- **hierarchical_clustering.py**: Core implementation of the multi-level clustering algorithm
- **test_hierarchical_clustering.py**: Test script to verify the implementation on a small dataset

## How It Works

The hierarchical clustering implementation:

1. Starts with broad categories (Level 1) using maximum inclusion parameters
2. For each cluster at Level 1, performs more granular clustering to create Level 2 subclusters
3. Continues this process through Level 3 and Level 4, becoming more specific at each level
4. Maintains parent-child relationships between clusters at different levels

## Configuration

The `hierarchical_config.json` file defines:

- **Multiple clustering levels** with different parameters:
  - Level 1: Broad categories (min_cluster_size=50, epsilon=1.5)
  - Level 2: Product types (min_cluster_size=20, epsilon=1.0)
  - Level 3: Specific products (min_cluster_size=6, epsilon=0.5)
  - Level 4: Variants (min_cluster_size=3, epsilon=0.25)
- **Global settings** for embedding model and processing
- **Progression rules** that determine when to stop sub-clustering

## Running the Clustering

### Testing the Implementation

Run the test script to verify the implementation on a small subset of data:

```bash
cd /Users/billnewman/Desktop/GitHub/VectorDB/product_clustering
python Hierarchical_clustering/test_hierarchical_clustering.py
```

This will:
- Create a sample dataset of 500 products
- Run the hierarchical clustering with smaller cluster sizes for faster testing
- Output a summary of the hierarchy and sample paths through the clusters

### Running on Full Dataset

To run on the full dataset:

```bash
cd /Users/billnewman/Desktop/GitHub/VectorDB/product_clustering
python Hierarchical_clustering/hierarchical_clustering.py --config Hierarchical_clustering/hierarchical_config.json
```

## Output

The hierarchical clustering produces:

1. `hierarchical_clusters.json`: Complete hierarchical structure with clusters at all levels
2. `cluster_relationships.json`: Parent-child relationships between clusters

Results are stored in the `data/hierarchical_clustering` directory.

## Visualization

To visualize the hierarchical clusters, a separate visualization module will be developed in the next phase to create interactive tree-based visualizations of the hierarchy.

## Next Steps

- Create a visualization tool for exploring the hierarchy
- Implement LLM-based analysis to evaluate cluster coherence
- Add USDA code alignment analysis at each level
