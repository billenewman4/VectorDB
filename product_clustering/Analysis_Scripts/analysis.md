# Product Clustering Analysis

## Exact Match Analysis

### Overview

The product clustering algorithm groups similar products together based on their semantic meaning, but it doesn't specifically identify "exact matches" within these clusters. An exact match represents products that are essentially identical but may have been entered into the system differently or come from different vendors.

This analysis identifies true exact matches within each cluster by applying stricter matching criteria:

1. **Almost identical product names** (using text similarity)
2. **Same size** (if size information is provided)
3. **Same brand** (if brand information is provided)
4. **Same company** (products from the same vendor)

### The Problem We're Solving

In many retail environments, the same product might be listed multiple times in inventory systems for various reasons:

- Different vendors supply the identical product
- Data entry variations (e.g., "Chicken Breast" vs "Chicken Breasts")
- Inconsistencies in naming conventions
- Duplicate entries from system migrations

These inconsistencies make it difficult to:
- Maintain accurate inventory counts
- Optimize purchasing decisions
- Standardize product naming
- Analyze sales and margins effectively

### Methodology

We've implemented two different approaches for identifying exact matches:

#### 1. Rules-Based Approach (exact_match_analyzer.py)

This approach uses several text analysis techniques to identify exact matches:

1. **Name Cleaning and Normalization**
   - Convert all text to lowercase
   - Remove special characters and punctuation
   - Standardize whitespace
   - Remove common filler words ("Inc.", "LLC", "Company", etc.)

2. **Feature Extraction**
   - Extract size information (e.g., "12 oz", "1.5 lb", "750 ml")
   - Identify brand information when available
   - Capture company/vendor data

3. **Similarity Calculation**
   - Use sequence matching algorithms to calculate text similarity scores
   - Apply a threshold (default: 85% similarity) to identify close matches
   - Verify additional criteria (size, brand, company) as specified

#### 2. LLM-Based Approach (cluster_llm_matcher.py)

This refined approach leverages GPT-3.5-Turbo to analyze entire clusters at once with an appropriate level of matching strictness:

1. **Whole-Cluster Analysis**
   - Send all products in a cluster to the LLM in a single prompt
   - Include SKU information with explicit instructions to only match different SKUs
   - Process multiple products simultaneously rather than making pairwise comparisons

2. **Balanced Sampling Strategy**
   - Sample a mix of small (3-6 products), medium (7-15 products), and large (>15 products) clusters
   - Prioritize smaller clusters (50%) that are more likely to contain genuine exact matches
   - Include medium (30%) and large (20%) clusters for diversity in analysis
   - Avoid overly large clusters that might dilute matching quality

3. **Properly Calibrated Matching Criteria**
   - Require products to be the same fundamental item, not just similar products
   - Enforce matching on critical attributes: product type, brand, size/count/weight, key specifications
   - Allow for minor differences: word order, formatting, abbreviations, typographical variations
   - Explicitly prohibit matching products with the same SKU to avoid false positives
   - Emphasize that capitalization differences should NEVER prevent a match
   - Clearly specify that punctuation variations (e.g., "5-up" vs "5 up" vs "5up") are irrelevant

4. **Concrete Examples in Prompt**
   - Include carefully selected example matches to guide the LLM's decision-making
   - Provide counterexamples of similar but non-matching products (e.g., same product type but different weights)
   - Explain the reasoning behind each example to reinforce matching criteria
   - Specifically include examples from key product categories like meat cuts (beef tenderloin, chicken wings)

4. **Multi-Level Deduplication**
   - Deduplicate products with the same SKU before sending to the LLM
   - Apply post-LLM verification to ensure match groups don't contain duplicate SKUs
   - Deduplicate final output to ensure each unique product is only listed once

5. **Match Grouping & Naming**
   - Generate a unique ID for each group of exact matches
   - Create a descriptive name based on common words in product descriptions, filtering out stopwords
   - Group all matching SKUs together in a structured CSV output

### Results

#### Comparative Analysis

We tested both approaches on a sample of clusters and observed these key differences:

**Rules-Based Approach**:
- Relies heavily on text similarity thresholds
- More consistent but less nuanced matching
- Requires careful parameter tuning for each dataset
- Unable to recognize semantic similarities without exact text matches

**LLM-Based Approach**:
- Better at understanding semantic equivalence despite textual differences
- Can recognize specialized product terminology and abbreviations
- More discerning about true exact matches vs. similar but distinct products
- Produces higher quality matches with minimal false positives
- Found genuine exact matches in categories like meat cuts, prepared foods, and dairy products

#### Example Exact Matches Found by LLM

1. **Beef Tenderloin**: Identical prime beef tenderloin products with different formatting
   - [SKU: 40011] Beef Tenderloin PSMO 5up Prime
   - [SKU: 141050] BEEF TENDERLOIN, PSMO, 5UP, PRIME

2. **Cheek Meat**: Several matching Caviness cheek meat products with minor variations in description
   - [SKU: 10081000] Cheek Meat, Greater Omaha (60#)
   - [SKU: 10071070] Cheek Meat (STEER), Caviness (60#) #2012
   - [SKU: 10083660] Cheek Meat (COW), Caviness (60#) #20120

3. **Veal Products**: Several "Frenched Veal Rib Chop" products with different SKUs but identical specifications

4. **Pork Products**: "Pork Chop Frenched" products with the same weight specifications

5. **Bacon Products**: Same bacon products with minor differences in packaging specifications

#### Output Format

The analysis produces a CSV file with the following columns:

1. **Match_ID**: A unique identifier for each exact match group (e.g., MATCH_LLM_0001)
2. **Match_Group_Name**: A descriptive name generated for the match group
3. **SKU_ID**: The individual SKU/product code
4. **SKU_Name**: The original product description
5. **Company**: The company/vendor for this product
6. **Cluster_ID**: The original cluster ID containing these products

### Business Value

This analysis delivers several key benefits:

- **Inventory Consolidation**: Identify opportunities to consolidate inventory counts
- **Purchasing Optimization**: Enable better negotiation by identifying identical products from different vendors
- **Data Cleaning**: Provide a foundation for cleaning and standardizing product data
- **Margin Analysis**: Enable more accurate profitability analysis across product lines

### Customization Options

The exact match analyzer can be customized with these parameters:

- **Name similarity threshold**: How similar product names need to be (0.0-1.0)
- **Size matching**: Option to ignore size differences
- **Brand matching**: Option to ignore brand differences
- **Company matching**: Option to ignore company/vendor differences

### How to Run the Analysis

```bash
python product_clustering/Analysis_Scripts/exact_match_analyzer.py
```

With custom options:

```bash
python product_clustering/Analysis_Scripts/exact_match_analyzer.py \
  --name_similarity 0.9 \
  --ignore_size \
  --output custom_output.csv
```

### Recent Improvements

#### Model Upgrade
We've upgraded the LLM model from GPT-3.5 Turbo to GPT-4o Mini, which provides:
- Better reasoning capabilities while remaining cost-effective
- More nuanced understanding of product specifications
- Improved ability to distinguish between similar but distinct products

#### Enhanced Prompt Engineering
- Added more balanced examples of matches and non-matches
- Clarified criteria for matching, emphasizing that capitalization and punctuation differences should not prevent matches
- Ensured company names don't bias the matching process

#### Technical Improvements
- Fixed CSV output handling to properly append new results without overwriting
- Ensured match groups maintain consistent IDs throughout the process
- Implemented proper deduplication to avoid redundant entries

#### Results
The improved system now finds more precise match groups:

1. **Short Rib Cluster**: Found 11 matching products that are essentially the same product with different vendor codes

2. **Beef Tenderloin Cluster**: Found 7 distinct match groups based on:
   - Grade differences (Choice, Prime, Select)
   - Preparation method (PSMO vs. SS)
   - Size specifications (5up, 6/7up)

### Next Steps

Based on our analysis results, we recommend the following next steps:

1. **Full Dataset Analysis**: Run the improved LLM-based matcher on the complete dataset to identify all exact matches

2. **Prompt Refinement**: Continue refining the system prompt with more examples from other product categories

3. **Data Standardization**: Create a process to standardize product naming and specifications based on identified matches

4. **Cross-Vendor Analysis**: Expand the analysis to specifically look for exact matches across different vendors

5. **Inventory Integration**: Connect exact match results with inventory management systems to consolidate counts

6. **Decision Support Integration**: Incorporate exact match information into purchasing decision workflows

7. **Continuous Refinement**: Periodically re-run the analysis as new products are added to keep matches current

8. **Manual Validation**: Have domain experts review a sample of the matches to verify accuracy

9. **Cost Analysis**: Calculate potential cost savings from consolidating purchasing of equivalent products

10. **Vendor Analysis**: Compare pricing for identical products across vendors

11. **System Integration**: Implement findings in inventory management systems

### Questions and Considerations

When reviewing the exact match results, consider:

1. Are there legitimate reasons for some products to remain separate despite being similar?
2. Should products with slight variations (e.g., different packaging) be considered exact matches?
3. How frequently should this analysis be run to maintain data quality?
4. What threshold of name similarity is appropriate for your specific data?