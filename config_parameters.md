# Product Clustering Configuration Parameters

This document provides detailed explanations for all parameters in the `max_inclusion_config.json` configuration file.

## Pipeline Control Parameters

| Parameter | Description |
|-----------|-------------|
| `all` | Run the complete pipeline (all steps) |
| `prepare` | Run data preparation step |
| `embed` | Run embedding generation step |
| `cluster` | Run clustering step |
| `analyze` | Run analysis step |
| `force` | Force reprocessing even if files exist |

## Data Options

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Directory for data files (null = use default) |
| `use_category_descriptions` | If false, only use product descriptions for clustering |
| `normalize_text` | Apply text normalization to descriptions |
| `expand_abbreviations` | Expand abbreviations in descriptions (e.g., "oz" → "ounce") |

## Embedding Options

| Parameter | Description |
|-----------|-------------|
| `embedding_model` | Model to generate embeddings ("all-mpnet-base-v2" is more powerful than MiniLM) |
| `embedding_batch_size` | Batch size for embedding generation (higher = faster but more memory) |
| `normalize_embeddings` | Normalize embeddings to unit length (improves clustering) |

## Core Clustering Parameters

| Parameter | Description |
|-----------|-------------|
| `metric` | Distance metric for clustering (euclidean is more stable than cosine) |
| `min_cluster_size` | Minimum number of points to form a cluster (lower = more granular clusters) |
| `min_samples` | Minimum number of points in neighborhood for core point (lower = more clusters) |

## Advanced Clustering Parameters

| Parameter | Description |
|-----------|-------------|
| `cluster_selection_epsilon` | Distance threshold for merging clusters. Higher values (e.g., 0.2) merge nearby clusters, creating fewer larger clusters. Value 0.0 means no automatic merging beyond algorithm determination. |
| `alpha` | Controls point weighting in the algorithm. Lower values (< 1.0) allow clusters to include more peripheral points. Higher values (> 1.0) make clusters more strict about density. |
| `cluster_selection_method` | Algorithm for extracting clusters: "eom" (Excess of Mass) is better for varying density clusters, "leaf" is better for more uniform clusters. |
| `n_jobs` | Number of parallel jobs (-1 = use all processors) |

## Test Mode Options

| Parameter | Description |
|-----------|-------------|
| `test` | Enable test mode for faster development |
| `sample_size` | Number of samples to use in test mode |

## Cluster Refinement Options

| Parameter | Description |
|-----------|-------------|
| `rerank` | Use cross-encoder for result reranking |
| `cross_encoder` | Model for reranking |
| `cross_encoder_batch_size` | Batch size for reranking |
| `similarity_threshold` | Threshold for considering items similar in reranking |
| `rerank_weight` | Weight given to reranking score vs original score |
| `test_clusters` | Number of clusters to test for reranking (0 = all) |
| `min_cluster_size_for_reranking` | Minimum cluster size to consider for reranking |

## Category-Based Clustering Options

| Parameter | Description |
|-----------|-------------|
| `no_categories` | Disable category-based clustering |
| `category_exclusivity` | Threshold for category exclusivity in clusters |

## Analysis Options

| Parameter | Description |
|-----------|-------------|
| `refined` | Use refined clusters for analysis |
| `analyze_basic` | Perform basic cluster analysis |
| `analyze_margins` | Analyze price margins within clusters |
| `analyze_usda` | Analyze USDA code mapping within clusters |
| `analyze_llm` | Use LLM for advanced cluster analysis |
| `llm_model` | LLM model to use (null = use default) |
| `cluster_size_threshold` | Minimum cluster size for detailed analysis |
| `price_variation_threshold` | Threshold for flagging price variations |
| `detailed_output` | Generate detailed analysis reports |

## Recommendations for Common Use Cases

### More Precise Clusters

If you want more precise, granular clusters:
- Set `min_cluster_size` to a low value (2-3)
- Set `min_samples` to a low value (1-2)
- Set `cluster_selection_epsilon` to 0.0 (no automatic merging)
- Use `alpha` = 1.0 (standard density-based clustering)

### Broader Clusters

If you want broader clusters that group more similar items together:
- Increase `min_cluster_size` to 5+
- Increase `min_samples` to 3+
- Set `cluster_selection_epsilon` to 0.2-0.3 (allow merging nearby clusters)
- Use `alpha` < 1.0 (0.75 allows more peripheral points)

### Faster Experimental Runs

For faster experimentation:
- Set `test` to true
- Adjust `sample_size` to a reasonable subset (1000-2000)
- Disable expensive operations like `analyze_margins` or `analyze_usda`
