# Pricing Analysis

This folder contains scripts for analyzing pricing data within product clusters to identify inconsistencies in pricing and margins.

## Overview

The analysis pipeline identifies clusters of similar products that have high variance in gross margin percentage, and determines whether the inconsistency is driven by pricing or cost factors. The final output is a management-ready report with detailed analysis and actionable recommendations.

## Data Sources

- **Transaction Data**: `Whetstone Product Costs Report.xlsx` containing product costs and pricing data
- **Cluster Definitions**: `refined_clusters.json` defining groups of similar products

## Complete Analysis Pipeline

### Step 1: Data Exploration

- **examine_excel.py** - Explores the Excel file structure to identify relevant sheets and columns
- **examine_excel_improved.py** - Enhanced version with better handling of multiple sheets

### Step 2: Data Processing

- **process_transaction_data.py** - Processes raw transaction data from the Excel file, validates and cleans the data, and calculates average pricing metrics per SKU

### Step 3: Match Products to Clusters

- **create_analysis_data.py** - Matches product SKUs between pricing data and cluster definitions, handling format differences

### Step 4: Variance Analysis

- **updated_cluster_analysis.py** - Analyzes variance of pricing metrics within product clusters, calculating statistics and identifying outlier products

### Step 5: Generate Final Report

- **generate_variance_report.py** - Analyzes the top clusters with highest variance, determines root causes (price or cost), and generates a readable management report

## How to Run the Complete Analysis

Run the following commands in order to execute the full analysis pipeline:

```bash
# Step 1: Process transaction data from the Excel file
python process_transaction_data.py

# Step 2: Match products between pricing data and clusters
python create_analysis_data.py

# Step 3: Perform cluster variance analysis
python updated_cluster_analysis.py

# Step 4: Generate the final analysis report
python generate_variance_report.py
```

## Output Files

The analysis produces the following output files:

- **product_pricing_averaged.csv** - Averaged pricing data per SKU
- **pricing_analysis_data.csv** - Matched products with cluster assignments
- **analysis_clusters.json** - Clusters with matched products
- **cluster_variance_stats.csv** - Statistics for each cluster
- **cluster_product_details.csv** - Details for each product in analyzed clusters
- **cluster_product_outliers.csv** - Products identified as outliers within their clusters
- **top_variance_clusters_analysis.md** - Final report with detailed analysis and recommendations

## Key Findings

The analysis successfully identified clusters with significant margin inconsistency and determined whether pricing or cost was the primary driver in each case:

1. **Cluster 575** (Beelers Bacon products) - Price-driven inconsistency
2. **Cluster 303** (Salsa products) - Cost-driven inconsistency
3. **Cluster 761** (Shrimp products) - Cost-driven inconsistency
4. **Cluster 687** (Beef bones) - Price-driven inconsistency
5. **Cluster 1061** (Chicken wings) - Price-driven inconsistency

For price-driven inconsistencies, we recommend standardizing pricing strategy across similar products. For cost-driven inconsistencies, we recommend reviewing vendor agreements and consolidating suppliers where possible.

## Debugging Utilities

The following scripts were used for debugging and can be helpful for troubleshooting:

- **debug_matches.py** - Checks overlap between pricing data SKUs and cluster product SKUs
- **debug_cluster_analysis.py** - Diagnoses issues with cluster variance analysis
- **debug_detailed.py** - Provides in-depth analysis of data structure and potential issues
- **fix_product_code_format.py** - Fixes format differences between product codes in different datasets
