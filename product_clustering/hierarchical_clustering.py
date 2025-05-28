#!/usr/bin/env python3
"""
Hierarchical Product Clustering

This script implements a tiered approach to product clustering:
1. Level 1: Broad categories (e.g., meat, produce, dairy)
2. Level 2: Subcategories (e.g., beef, poultry, pork)
3. Level 3: Specific products (e.g., steaks, ground beef, roasts)

This approach significantly increases clustering coverage while maintaining cluster quality.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import hdbscan
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HierarchicalProductClustering:
    """
    Implements hierarchical product clustering to maximize coverage.
    """
    
    def __init__(self, 
                data_dir: Optional[str] = None,
                embeddings_path: Optional[str] = None,
                product_codes_path: Optional[str] = None,
                transaction_data_path: Optional[str] = None):
        """
        Initialize the hierarchical clustering processor.
        
        Args:
            data_dir: Base directory for data files
            embeddings_path: Path to product embeddings
            product_codes_path: Path to product codes file
            transaction_data_path: Path to transaction data with product descriptions
        """
        # Set up data paths
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        else:
            self.data_dir = data_dir
            
        self.embeddings_path = embeddings_path or os.path.join(self.data_dir, "product_embeddings.npy")
        self.product_codes_path = product_codes_path or os.path.join(self.data_dir, "product_codes.txt")
        self.transaction_data_path = transaction_data_path
        
        # Output directories
        self.output_dir = os.path.join(self.data_dir, "hierarchical_clusters")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data containers
        self.embeddings = None
        self.product_codes = None
        self.product_descriptions = None
        self.embedding_dict = {}
        self.product_dict = {}
        
        # Category extraction patterns
        self.category_patterns = [
            # Meat categories
            (r'\b(beef|steak|ribeye|sirloin|chuck|roast|brisket)\b', 'beef'),
            (r'\b(pork|ham|bacon|loin|shoulder|rib)\b', 'pork'),
            (r'\b(chicken|poultry|turkey|breast|wing|thigh|drumstick)\b', 'poultry'),
            (r'\b(lamb|mutton|chop)\b', 'lamb'),
            (r'\b(fish|seafood|salmon|tuna|cod|shrimp|lobster|crab|clam|mussel|oyster)\b', 'seafood'),
            
            # Produce categories
            (r'\b(vegetable|produce|veg)\b', 'vegetables'),
            (r'\b(fruit|apple|orange|banana|grape|berry|melon)\b', 'fruit'),
            (r'\b(lettuce|salad|spinach|kale|greens)\b', 'leafy_greens'),
            (r'\b(potato|onion|carrot|garlic|root)\b', 'root_vegetables'),
            
            # Dairy categories
            (r'\b(milk|cream|dairy|yogurt|yoghurt)\b', 'dairy'),
            (r'\b(cheese|cheddar|swiss|mozzarella|parmesan|brie)\b', 'cheese'),
            (r'\b(butter|margarine)\b', 'butter'),
            
            # Bakery categories
            (r'\b(bread|loaf|roll|bun|bakery)\b', 'bread'),
            (r'\b(cake|pastry|dessert|cookie|muffin|donut)\b', 'pastries'),
            
            # Other food categories
            (r'\b(cereal|grain|rice|pasta|noodle)\b', 'grains'),
            (r'\b(oil|vinegar|dressing|sauce|condiment)\b', 'condiments'),
            (r'\b(spice|herb|seasoning)\b', 'spices'),
            (r'\b(soup|broth|stock)\b', 'soups'),
            (r'\b(frozen|ice cream|pizza)\b', 'frozen'),
            (r'\b(snack|chip|crisp|pretzel|popcorn)\b', 'snacks'),
            (r'\b(candy|chocolate|sweet|confection)\b', 'candy'),
            (r'\b(beverage|drink|water|juice|soda|pop)\b', 'beverages'),
            (r'\b(coffee|tea)\b', 'coffee_tea'),
            (r'\b(alcohol|wine|beer|spirit|liquor)\b', 'alcohol'),
            
            # Non-food categories
            (r'\b(cleaning|detergent|soap)\b', 'cleaning'),
            (r'\b(paper|napkin|tissue|toilet)\b', 'paper_goods'),
            (r'\b(health|medicine|vitamin|supplement)\b', 'health'),
            (r'\b(pet|dog|cat|animal)\b', 'pet'),
            
            # Fallback category
            (r'.*', 'uncategorized')
        ]
    
    def load_data(self):
        """
        Load embeddings, product codes, and product descriptions.
        """
        print("Loading data...")
        
        # Load embeddings and product codes
        if os.path.exists(self.embeddings_path) and os.path.exists(self.product_codes_path):
            self.embeddings = np.load(self.embeddings_path)
            with open(self.product_codes_path, 'r') as f:
                self.product_codes = [line.strip() for line in f.readlines()]
            
            if len(self.embeddings) != len(self.product_codes):
                print(f"Warning: Mismatch between embeddings ({len(self.embeddings)}) "
                      f"and product codes ({len(self.product_codes)})")
            
            # Create mapping from product code to embedding
            for i, code in enumerate(self.product_codes):
                if i < len(self.embeddings):
                    self.embedding_dict[code] = self.embeddings[i]
            
            print(f"Loaded {len(self.embedding_dict)} product embeddings")
        else:
            print(f"Error: Embeddings or product codes file not found")
            return False
        
        # Load product descriptions
        if self.transaction_data_path and os.path.exists(self.transaction_data_path):
            try:
                df = pd.read_excel(self.transaction_data_path)
                
                # Handle different column name formats
                code_col = next((col for col in df.columns if col.lower() in 
                                ['product_code', 'productcode', 'code']), None)
                desc_col = next((col for col in df.columns if col.lower() in 
                                ['description', 'product_description', 'productdescription', 'desc']), None)
                
                if code_col and desc_col:
                    print(f"Using columns: {code_col} (code) and {desc_col} (description)")
                    for _, row in df.iterrows():
                        if pd.notna(row[code_col]) and pd.notna(row[desc_col]):
                            self.product_dict[str(row[code_col])] = str(row[desc_col])
                    
                    print(f"Loaded {len(self.product_dict)} product descriptions")
                else:
                    print(f"Required columns not found. Available columns: {list(df.columns)}")
                    return False
            except Exception as e:
                print(f"Error loading transaction data: {str(e)}")
                return False
        else:
            print("Warning: No transaction data provided, cannot extract categories")
            return False
        
        return True
    
    def extract_categories(self):
        """
        Extract categories from product descriptions.
        
        Returns:
            Dictionary mapping product codes to their categories
        """
        product_categories = {}
        
        for code, description in self.product_dict.items():
            # Convert to uppercase for case-insensitive matching
            upper_desc = description.upper()
            
            # Find the first matching category
            category = None
            for pattern, cat in self.category_patterns:
                if re.search(pattern, upper_desc, re.IGNORECASE):
                    category = cat
                    break
            
            # Save the category
            product_categories[code] = category or 'uncategorized'
        
        # Count categories
        category_counts = defaultdict(int)
        for category in product_categories.values():
            category_counts[category] += 1
        
        print("Category distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} products")
        
        return product_categories
    
    def cluster_by_level(self, 
                        embeddings: np.ndarray, 
                        product_codes: List[str],
                        min_cluster_size: int = 5, 
                        min_samples: int = 2,
                        metric: str = 'euclidean'):
        """
        Perform clustering at a specific level.
        
        Args:
            embeddings: Product embeddings
            product_codes: Corresponding product codes
            min_cluster_size: Minimum size of clusters
            min_samples: HDBSCAN min_samples parameter
            metric: Distance metric to use
            
        Returns:
            Dictionary mapping cluster IDs to lists of product codes
        """
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            gen_min_span_tree=True,
            cluster_selection_method='eom'
        )
        
        # Fit the clusterer
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Organize into clusters
        clusters = defaultdict(list)
        unclustered = []
        
        for i, label in enumerate(cluster_labels):
            if label >= 0 and i < len(product_codes):
                clusters[f"cluster_{label}"].append(product_codes[i])
            elif i < len(product_codes):
                unclustered.append(product_codes[i])
        
        print(f"Created {len(clusters)} clusters with {sum(len(c) for c in clusters.values())} products")
        print(f"{len(unclustered)} products remained unclustered")
        
        return dict(clusters), unclustered
    
    def refine_clusters_with_crossencoder(self, 
                                         clusters: Dict[str, List[str]], 
                                         similarity_threshold: float = 0.6):
        """
        Refine clusters using CrossEncoder reranking.
        
        Args:
            clusters: Dictionary mapping cluster IDs to lists of product codes
            similarity_threshold: Minimum similarity threshold for products to remain in a cluster
            
        Returns:
            Dictionary of refined clusters
        """
        # Import here to avoid loading model if not needed
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            
            print("Loading CrossEncoder model...")
            model = CrossEncoder('cross-encoder/stsb-roberta-base')
            
            refined_clusters = {}
            
            for cluster_id, product_codes in clusters.items():
                if len(product_codes) <= 1:
                    refined_clusters[cluster_id] = product_codes
                    continue
                
                # Get descriptions for all products in this cluster
                descriptions = []
                valid_indices = []
                valid_codes = []
                
                for i, code in enumerate(product_codes):
                    if code in self.product_dict:
                        descriptions.append(self.product_dict[code])
                        valid_indices.append(i)
                        valid_codes.append(code)
                
                if len(descriptions) <= 1:
                    refined_clusters[cluster_id] = product_codes
                    continue
                
                # Generate all pairs for similarity comparison
                pairs = []
                for i in range(len(descriptions)):
                    for j in range(i+1, len(descriptions)):
                        pairs.append([descriptions[i], descriptions[j]])
                
                # Get similarity scores
                if pairs:
                    similarity_scores = model.predict(pairs)
                    
                    # Create graph of products that are similar enough
                    graph = defaultdict(set)
                    pair_idx = 0
                    
                    for i in range(len(descriptions)):
                        for j in range(i+1, len(descriptions)):
                            if similarity_scores[pair_idx] >= similarity_threshold:
                                graph[valid_codes[i]].add(valid_codes[j])
                                graph[valid_codes[j]].add(valid_codes[i])
                            pair_idx += 1
                    
                    # Find connected components (refined clusters)
                    refined_products = []
                    visited = set()
                    
                    for code in valid_codes:
                        if code not in visited:
                            component = set()
                            queue = [code]
                            
                            while queue:
                                current = queue.pop(0)
                                if current not in visited:
                                    visited.add(current)
                                    component.add(current)
                                    
                                    for neighbor in graph[current]:
                                        if neighbor not in visited:
                                            queue.append(neighbor)
                            
                            if len(component) >= 2:  # Only keep components with at least 2 products
                                refined_products.append(list(component))
                    
                    # Create refined clusters
                    for i, component in enumerate(refined_products):
                        refined_clusters[f"{cluster_id}_refined_{i}"] = component
                else:
                    refined_clusters[cluster_id] = product_codes
            
            total_products = sum(len(products) for products in refined_clusters.values())
            print(f"Created {len(refined_clusters)} refined clusters with {total_products} total products")
            
            return refined_clusters
        except ImportError:
            print("CrossEncoder model not available. Skipping refinement.")
            return clusters
    
    def run_hierarchical_clustering(self):
        """
        Run the complete hierarchical clustering process.
        
        Returns:
            Dictionary of hierarchical clusters
        """
        if not self.load_data():
            print("Failed to load required data")
            return {}
        
        # Extract product categories
        print("Extracting product categories...")
        product_categories = self.extract_categories()
        
        # Group products by category
        category_products = defaultdict(list)
        for code, category in product_categories.items():
            if code in self.embedding_dict:
                category_products[category].append(code)
        
        # Level 1: Cluster by broad categories
        level1_clusters = {}
        unclustered_products = []
        
        for category, products in category_products.items():
            print(f"\nProcessing category: {category} ({len(products)} products)")
            
            # Skip tiny categories
            if len(products) < 5:
                unclustered_products.extend(products)
                continue
            
            # Get embeddings for this category
            category_embeddings = []
            category_codes = []
            
            for code in products:
                if code in self.embedding_dict:
                    category_embeddings.append(self.embedding_dict[code])
                    category_codes.append(code)
            
            # Cluster within this category (Level 2)
            print(f"Running Level 2 clustering for {category}...")
            level2_clusters, level2_unclustered = self.cluster_by_level(
                np.array(category_embeddings),
                category_codes,
                min_cluster_size=5,  # Larger clusters at this level
                min_samples=3,
                metric='euclidean'
            )
            
            # Add Level 2 clusters to results
            for cluster_id, products in level2_clusters.items():
                level1_clusters[f"{category}_{cluster_id}"] = products
            
            # Further cluster the unclustered products with more lenient parameters (Level 3)
            if level2_unclustered:
                print(f"Running Level 3 clustering for {len(level2_unclustered)} unclustered products in {category}...")
                
                # Get embeddings for unclustered products
                unclustered_embeddings = []
                unclustered_codes = []
                
                for code in level2_unclustered:
                    if code in self.embedding_dict:
                        unclustered_embeddings.append(self.embedding_dict[code])
                        unclustered_codes.append(code)
                
                if len(unclustered_codes) >= 3:
                    level3_clusters, level3_unclustered = self.cluster_by_level(
                        np.array(unclustered_embeddings),
                        unclustered_codes,
                        min_cluster_size=3,  # Smaller clusters allowed
                        min_samples=2,
                        metric='euclidean'
                    )
                    
                    # Add Level 3 clusters to results
                    for cluster_id, products in level3_clusters.items():
                        level1_clusters[f"{category}_level3_{cluster_id}"] = products
                    
                    # Add remaining unclustered products to global unclustered list
                    unclustered_products.extend(level3_unclustered)
                else:
                    unclustered_products.extend(unclustered_codes)
            
        # Try one final clustering step for all unclustered products
        if unclustered_products:
            print(f"\nFinal clustering pass for {len(unclustered_products)} remaining unclustered products...")
            
            # Get embeddings for unclustered products
            final_embeddings = []
            final_codes = []
            
            for code in unclustered_products:
                if code in self.embedding_dict:
                    final_embeddings.append(self.embedding_dict[code])
                    final_codes.append(code)
            
            if len(final_codes) >= 3:
                final_clusters, final_unclustered = self.cluster_by_level(
                    np.array(final_embeddings),
                    final_codes,
                    min_cluster_size=2,  # Very small clusters allowed
                    min_samples=1,
                    metric='euclidean'
                )
                
                # Add final clusters to results
                for cluster_id, products in final_clusters.items():
                    level1_clusters[f"final_pass_{cluster_id}"] = products
            
        # Refine clusters with CrossEncoder (optional)
        refined_clusters = self.refine_clusters_with_crossencoder(level1_clusters)
        
        # Save results
        self.save_clusters(refined_clusters)
        
        # Generate report
        all_products = set(self.product_codes)
        clustered_products = set()
        
        for products in refined_clusters.values():
            clustered_products.update(products)
        
        coverage = len(clustered_products) / len(all_products) * 100 if all_products else 0
        
        print("\nHierarchical Clustering Summary:")
        print(f"Total clusters: {len(refined_clusters)}")
        print(f"Products clustered: {len(clustered_products)} out of {len(all_products)} ({coverage:.1f}%)")
        
        return refined_clusters
    
    def save_clusters(self, clusters: Dict[str, List[str]]):
        """
        Save clusters to JSON file.
        
        Args:
            clusters: Dictionary mapping cluster IDs to lists of product codes
        """
        output_path = os.path.join(self.output_dir, "hierarchical_clusters.json")
        
        with open(output_path, 'w') as f:
            json.dump(clusters, f, indent=2)
        
        print(f"Saved clusters to {output_path}")
        
        # Also save a human-readable version with descriptions
        readable_path = os.path.join(self.output_dir, "hierarchical_clusters_readable.md")
        
        with open(readable_path, 'w') as f:
            f.write("# Hierarchical Product Clusters\n\n")
            
            # Count total clustered products
            all_products = set()
            for products in clusters.values():
                all_products.update(products)
            
            f.write(f"## Summary\n\n")
            f.write(f"- Total clusters: {len(clusters)}\n")
            f.write(f"- Products clustered: {len(all_products)} out of {len(self.product_codes)} ")
            f.write(f"({len(all_products)/len(self.product_codes)*100:.1f}%)\n\n")
            
            # Group clusters by category
            categories = defaultdict(list)
            
            for cluster_id, products in clusters.items():
                if "_" in cluster_id:
                    category = cluster_id.split("_")[0]
                    categories[category].append((cluster_id, products))
                else:
                    categories["other"].append((cluster_id, products))
            
            # Write each category
            for category, category_clusters in sorted(categories.items()):
                f.write(f"## {category.title()} Category ({len(category_clusters)} clusters)\n\n")
                
                for cluster_id, products in sorted(category_clusters, key=lambda x: len(x[1]), reverse=True):
                    f.write(f"### {cluster_id} ({len(products)} products)\n\n")
                    
                    for code in products:
                        description = self.product_dict.get(code, "Unknown")
                        f.write(f"- {code}: {description}\n")
                    
                    f.write("\n")
        
        print(f"Saved human-readable clusters to {readable_path}")

def main():
    """Main function to run hierarchical clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hierarchical product clustering")
    parser.add_argument("--data_dir", help="Directory containing data files")
    parser.add_argument("--embeddings", help="Path to product embeddings")
    parser.add_argument("--product_codes", help="Path to product codes")
    parser.add_argument("--transaction_data", help="Path to transaction data with product descriptions")
    
    args = parser.parse_args()
    
    # Set default transaction data path if not provided
    transaction_data_path = args.transaction_data
    if transaction_data_path is None:
        potential_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "data", "Actuals", "Transaction_Report_Actual.xlsx")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                transaction_data_path = path
                break
    
    # Run hierarchical clustering
    clustering = HierarchicalProductClustering(
        data_dir=args.data_dir,
        embeddings_path=args.embeddings,
        product_codes_path=args.product_codes,
        transaction_data_path=transaction_data_path
    )
    
    clustering.run_hierarchical_clustering()

if __name__ == "__main__":
    main()
