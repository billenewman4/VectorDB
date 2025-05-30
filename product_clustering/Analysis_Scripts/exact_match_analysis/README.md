# Exact Match Analysis

This directory contains scripts and data for identifying exact product matches using LLM technology.

## Files Overview

- **cluster_llm_matcher.py**: Main script for using LLMs to identify exact product matches within clusters
  - Runs the matching process across product clusters
  - Configurable sample size and overwrite options
  - Uses GPT-4o mini for high-quality matching results

- **exact_match_analyzer.py**: Analyzes match results and provides metrics and insights
  - Calculates metrics like total match groups, products matched, etc.
  - Identifies clusters with highest match density
  - Generates summary statistics for business decision-making

- **test_specific_match.py**: Utility for testing specific product matches
  - Allows focused testing of particular product combinations
  - Useful for debugging and validating match criteria

- **analysis.md**: Documentation of findings, methodology, and future improvements
  - Describes the exact match analysis process
  - Contains key insights and observed patterns
  - Outlines recommendations for further refinement

- **llm_exact_matches.csv**: Output file containing all identified exact matches
  - Each row represents a product in a match group
  - Includes SKU details, company information, and cluster origin

- **refined_clusters.json**: Input data file containing product clusters to analyze

## How to Run

### Finding Exact Matches

```bash
python cluster_llm_matcher.py --sample_size 1000 --overwrite
```

Options:
- `--sample_size`: Number of clusters to analyze (default: 100)
- `--overwrite`: Overwrite existing output file instead of appending (default: False)

### Analyzing Results

```bash
python exact_match_analyzer.py
```

This will generate a summary of the matches found, including metrics on match groups, 
products matched, and distribution of matches across clusters.

## Results Summary

The latest analysis across 1000 clusters found:

- 238 distinct match groups
- 683 products with at least one duplicate
- 114 clusters (11.4%) containing exact matches
- Average of 2.87 products per match group

These matches represent opportunities for inventory consolidation and purchasing optimization.
