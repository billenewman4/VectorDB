#!/usr/bin/env python3
"""
Hierarchical Multi-Level Product Clustering

This module implements a progressive, hierarchical clustering approach that groups products
at multiple levels of granularity, starting with broad categories and progressively
refining to more specific groups.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

# Add parent directories to path to import from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "hierarchical_clustering.log"))
    ]
)
logger = logging.getLogger("hierarchical_clustering")

# Import required modules from existing implementation
import sys
sys.path.insert(0, os.path.join(grandparent_dir, 'src'))
from improved_clustering import run_improved_clustering
from data_processing import load_transaction_data, process_transaction_data, clean_text
from abbreviation_translator import expand_abbreviations

# Import for cross-encoder support
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers package not found. Cross-encoder functionality will be disabled.")
    CROSS_ENCODER_AVAILABLE = False


class HierarchicalClusterer:
    """
    Implements hierarchical multi-level clustering on product data.
    """
    
    def __init__(self, config_path: str, data_dir: Optional[str] = None):
        """
        Initialize the hierarchical clusterer with a configuration file.
        
        Args:
            config_path: Path to the hierarchical configuration JSON file.
            data_dir: Directory containing data files. If None, uses default.
        """
        self.config_path = config_path
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data"
        )
        # Output directory specifically for hierarchical clustering
        self.output_dir = os.path.join(self.data_dir, "hierarchical_clustering")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        self.levels = self.config.get("levels", [])
        self.global_settings = self.config.get("global_settings", {})
        self.progression_rules = self.config.get("progression_rules", {})
        
        # Set default level jumps if not specified
        if "allow_level_jumps" not in self.progression_rules:
            self.progression_rules["allow_level_jumps"] = [3, 4]  # Allow level 1 clusters to jump to levels 3 and 4
        
        # Cross-encoder configuration
        self.use_cross_encoder = self.global_settings.get("use_cross_encoder", False) and CROSS_ENCODER_AVAILABLE
        self.cross_encoder_model = self.global_settings.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.cross_encoder_batch_size = self.global_settings.get("cross_encoder_batch_size", 32)
        self.cross_encoder_threshold = self.global_settings.get("cross_encoder_threshold", 0.5)
        self.cross_encoder = None
        
        # Load cross-encoder if enabled
        if self.use_cross_encoder:
            try:
                logger.info(f"Loading cross-encoder model: {self.cross_encoder_model}")
                self.cross_encoder = CrossEncoder(self.cross_encoder_model, max_length=512)
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading cross-encoder model: {e}")
                self.use_cross_encoder = False
        
        # Initialize storage for hierarchical clusters
        self.hierarchical_clusters = {}
        for level in self.levels:
            self.hierarchical_clusters[f"level_{level['level']}"] = {}
            
        # Track cluster relationships
        self.cluster_relationships = defaultdict(list)
        
        logger.info(f"Initialized HierarchicalClusterer with {len(self.levels)} levels")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Cross-encoder refinement: {'Enabled' if self.use_cross_encoder else 'Disabled'}")
        if self.use_cross_encoder:
            logger.info(f"Using cross-encoder model: {self.cross_encoder_model}")
            logger.info(f"Cross-encoder threshold: {self.cross_encoder_threshold}")
            logger.info(f"Cross-encoder batch size: {self.cross_encoder_batch_size}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the hierarchical configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_data(self) -> Tuple[np.ndarray, List[str]]:
        """Load product embeddings and product codes from files.
        
        Returns:
            A tuple of (embeddings, product_codes)
        """
        try:
            # Define paths to the embedding and product code files
            embeddings_path = os.path.join(self.data_dir, "product_embeddings.npy")
            product_codes_path = os.path.join(self.data_dir, "product_codes.txt")
            
            # Load embeddings from numpy file
            logger.info(f"Loading embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
            
            # Load product codes from text file
            logger.info(f"Loading product codes from {product_codes_path}")
            with open(product_codes_path, 'r') as f:
                product_codes = [line.strip() for line in f.readlines()]
            
            logger.info(f"Loaded {embeddings.shape[0]} product embeddings with {embeddings.shape[1]} dimensions")
            logger.info(f"Loaded {len(product_codes)} product codes")
            
            # Verify that we have the same number of embeddings and product codes
            if embeddings.shape[0] != len(product_codes):
                raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) does not match number of product codes ({len(product_codes)})")
                
            return embeddings, product_codes
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def run_hierarchical_clustering(self):
        """Run the complete hierarchical clustering process with the simplified approach.
        
        In this approach:
        1. All products start out assigned to their level 1 clusters
        2. For each level (2, 3, 4), we take all products and try to refine their clustering
        3. Products either move to a new, more refined cluster at the current level,
           or stay with their previous level assignment if clustering doesn't refine them
        4. We track the "latest" cluster assignment for each product across all levels
        """
        logger.info("Starting hierarchical clustering process")
        
        # Make sure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load product embeddings and codes
        embeddings, product_codes = self._load_data()
        
        # Process the first level (product categories)
        level_1_config = next((level for level in self.levels if level["level"] == 1), None)
        if not level_1_config:
            logger.error("Level 1 configuration not found")
            return
        
        logger.info(f"\n==== Processing Level 1: {level_1_config['name']} ====\n")
        print(f"\n==== Processing Level 1: {level_1_config['name']} ====\n")
        
        # Create the level 1 directory
        level_1_output_dir = os.path.join(self.output_dir, "level_1")
        os.makedirs(level_1_output_dir, exist_ok=True)
        
        # First level clusters - direct clustering on all products
        level_1_clusters = self._cluster_at_level(
            product_codes=product_codes,
            embeddings=embeddings,
            level_config=level_1_config,
            parent_id=None
        )
        
        # Initialize the hierarchical cluster structure
        self.hierarchical_clusters = {
            "level_1": level_1_clusters
        }
        
        # Initialize cluster relationships
        self.cluster_relationships = {}
        
        # Track the latest cluster assignment for each product
        # Initialize with level 1 assignments
        latest_product_clusters = {}
        for cluster_id, cluster_info in level_1_clusters.items():
            for product in cluster_info.get("products", []):
                latest_product_clusters[product] = {
                    "level": 1,
                    "cluster_id": cluster_id
                }
        
        # Create a mapping from product code to its index in the embeddings array
        product_to_idx = {code: idx for idx, code in enumerate(product_codes)}
        
        # Track failed refinement attempts - clusters that couldn't be refined at a specific level
        # Format: {level: {cluster_id: set(products)}}
        failed_refinements = {}
        
        # Process subsequent levels
        for level_idx in range(1, len(self.levels)):
            level_num = level_idx + 1
            print(f"\n\n===== STARTING LEVEL {level_num} PROCESSING =====\n")
            logger.info(f"Starting level {level_num} processing")
            
            if level_num > self.progression_rules.get("max_depth", 4):
                logger.info(f"Reached maximum depth {self.progression_rules['max_depth']}")
                break
            
            level_config = next((level for level in self.levels if level["level"] == level_num), None)
            if not level_config:
                logger.error(f"Level {level_num} configuration not found")
                continue
            
            logger.info(f"\n==== Processing Level {level_num}: {level_config['name']} ====\n")
            print(f"\n==== Processing Level {level_num}: {level_config['name']} ====\n")
            
            # Create the level directory
            level_output_dir = os.path.join(self.output_dir, f"level_{level_num}")
            os.makedirs(level_output_dir, exist_ok=True)
            
            # Initialize this level's clusters dictionary
            self.hierarchical_clusters[f"level_{level_num}"] = {}
            
            # Group products by their most recent cluster assignment
            products_by_latest_cluster = {}
            
            for product, assignment in latest_product_clusters.items():
                cluster_id = assignment["cluster_id"]
                if cluster_id not in products_by_latest_cluster:
                    products_by_latest_cluster[cluster_id] = []
                products_by_latest_cluster[cluster_id].append(product)
                
            # For levels 3 and 4, also include level 1 clusters that failed to refine at lower levels
            if level_num >= 3 and level_num in self.progression_rules.get("allow_level_jumps", [3, 4]):
                level_1_products_to_process = set()
                level_1_clusters_to_process = set()
                
                # Check if there are level 1 clusters that failed at level 2
                if 2 in failed_refinements:
                    logger.info(f"Found {len(failed_refinements[2])} clusters that failed refinement at level 2")
                    print(f"\nFound {len(failed_refinements[2])} clusters that failed refinement at level 2")
                    
                    # Count products from level 1 that failed at level 2
                    level_1_product_count = 0
                    level_1_cluster_count = 0
                    
                    for cluster_id, products in failed_refinements[2].items():
                        if not products:  # Skip if empty
                            continue
                            
                        # Get the original level for this cluster by checking a product in it
                        product_sample = next(iter(products), None)
                        if not product_sample:
                            continue
                            
                        product_assignment = latest_product_clusters.get(product_sample, {})
                        product_level = product_assignment.get("level")
                        original_cluster = product_assignment.get("cluster_id")
                        
                        # Only process if this is a level 1 cluster
                        if product_level == 1:
                            level_1_cluster_count += 1
                            level_1_product_count += len(products)
                            
                            # This is a level 1 cluster that failed at level 2, try it at this level
                            level_1_clusters_to_process.add(original_cluster)
                            
                            if original_cluster not in products_by_latest_cluster:
                                products_by_latest_cluster[original_cluster] = []
                                
                            # Add these products to be processed at this level
                            for product in products:
                                if product not in products_by_latest_cluster[original_cluster]:
                                    products_by_latest_cluster[original_cluster].append(product)
                                    level_1_products_to_process.add(product)
                    
                    logger.info(f"Found {level_1_cluster_count} level 1 clusters with {level_1_product_count} products that failed at level 2")
                    print(f"Found {level_1_cluster_count} level 1 clusters with {level_1_product_count} products that failed at level 2")
                
                if level_1_products_to_process:
                    logger.info(f"DIRECT LEVEL JUMP: Including {len(level_1_products_to_process)} products from {len(level_1_clusters_to_process)} level 1 clusters for direct refinement at level {level_num}")
                    print(f"\n===== DIRECT LEVEL JUMP =====\nIncluding {len(level_1_products_to_process)} products from {len(level_1_clusters_to_process)} level 1 clusters for direct refinement at level {level_num}\n============================")
            
            # Track stats for debugging
            progression_stats = {
                "clusters_processed": 0,
                "products_processed": 0,
                "new_clusters_created": 0,
                "products_refined": 0
            }
            
            total_clusters_created = 0
            
            # Process each group of products from the previous clusters
            for prev_cluster_id, cluster_products in products_by_latest_cluster.items():
                # Skip if this cluster has been marked as finished
                prev_level = int(latest_product_clusters[cluster_products[0]]["level"])
                prev_level_name = next((level["name"] for level in self.levels if level["level"] == prev_level), f"level_{prev_level}")
                prev_cluster_info = self.hierarchical_clusters[f"level_{prev_level}"].get(prev_cluster_id, {})
                
                # Track statistics
                progression_stats["clusters_processed"] += 1
                progression_stats["products_processed"] += len(cluster_products)
                
                # Apply cross-encoder refinement if enabled and appropriate
                # For levels 2 and above, we want to refine using cross-encoder if possible
                use_cross_encoder_refinement = self.use_cross_encoder and level_num >= 2 and len(cluster_products) > 1
                
                # Get embeddings for these products
                product_indices = [product_to_idx[p] for p in cluster_products if p in product_to_idx]
                if not product_indices:
                    logger.warning(f"No valid embeddings found for products in cluster {prev_cluster_id}")
                    continue
                    
                cluster_embeddings = embeddings[product_indices]
                
                # Get valid product codes that match the embeddings
                valid_products = [cluster_products[i] for i in range(len(cluster_products)) if i < len(product_indices)]
                
                # First try standard clustering with the embedding model
                subclusters = self._cluster_at_level(
                    product_codes=valid_products,
                    embeddings=cluster_embeddings,
                    level_config=level_config,
                    parent_id=prev_cluster_id
                )
                
                # Check if we need cross-encoder refinement
                if use_cross_encoder_refinement:
                    # If standard clustering didn't produce subclusters OR we have a large cluster
                    # that might benefit from further refinement, apply cross-encoder
                    no_subclusters = len(subclusters) <= 1
                    large_cluster = any(len(cluster_prods) > self.global_settings.get("min_products_for_refinement", 5) 
                                      for cluster_prods in subclusters.values())
                    
                    if no_subclusters or large_cluster:
                        logger.info(f"Applying cross-encoder refinement to cluster {prev_cluster_id} with {len(valid_products)} products")
                        print(f"\nApplying cross-encoder refinement to cluster {prev_cluster_id} with {len(valid_products)} products")
                        
                        # Load product descriptions for better cross-encoder performance
                        try:
                            # Try multiple possible locations for the transactions data
                            transactions_paths = [
                                os.path.join(self.data_dir, "processed_transactions.csv"),
                                os.path.join(os.path.dirname(self.data_dir), "processed_transactions.csv"),
                                os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), "data", "processed_transactions.csv")
                            ]
                            
                            found_file = False
                            for transactions_path in transactions_paths:
                                if os.path.exists(transactions_path):
                                    logger.info(f"Found transactions file at: {transactions_path}")
                                    df = pd.read_csv(transactions_path)
                                    product_df = df[df['product_code'].isin(valid_products)].drop_duplicates('product_code')
                                    product_lookup = dict(zip(product_df['product_code'], product_df['description']))
                                    descriptions = [product_lookup.get(p, f"Unknown product {p}") for p in valid_products]
                                    found_file = True
                                    break
                            
                            if not found_file:
                                logger.warning("Could not find processed_transactions.csv in any expected location")
                                descriptions = None
                        except Exception as e:
                            logger.error(f"Error loading descriptions for cross-encoder: {e}")
                            descriptions = None
                        
                        # Apply cross-encoder refinement
                        refined_subclusters = self._refine_with_cross_encoder(valid_products, descriptions)
                        
                        # Only use refined subclusters if they actually improved granularity
                        if len(refined_subclusters) > len(subclusters):
                            logger.info(f"Cross-encoder refinement improved clustering: {len(subclusters)} → {len(refined_subclusters)} clusters")
                            print(f"Cross-encoder refinement improved clustering: {len(subclusters)} → {len(refined_subclusters)} clusters")
                            
                            # Convert refined subclusters to the expected format
                            new_subclusters = {}
                            for refined_id, products in refined_subclusters.items():
                                # Create unique ID that includes parent and level info
                                unique_id = f"{prev_level_name}_{level_config['name']}_{refined_id}"
                                
                                new_subclusters[unique_id] = {
                                    "level": level_num,
                                    "parent": prev_cluster_id,
                                    "products": products,
                                    "children": [],
                                    "metadata": {
                                        "cluster_size": len(products),
                                        "parameters": {
                                            "refined_by_cross_encoder": True,
                                            "cross_encoder_model": self.cross_encoder_model,
                                            "cross_encoder_threshold": self.cross_encoder_threshold
                                        }
                                    }
                                }
                            subclusters = new_subclusters
                
                # Skip if no meaningful clusters were created
                if not subclusters:
                    logger.info(f"No meaningful subclusters found for cluster {prev_cluster_id} at level {level_num}")
                    
                    # Track this as a failed refinement attempt
                    if level_num not in failed_refinements:
                        failed_refinements[level_num] = {}
                    
                    if prev_cluster_id not in failed_refinements[level_num]:
                        failed_refinements[level_num][prev_cluster_id] = set()
                    
                    # Add all products from this cluster to the failed refinements
                    failed_refinements[level_num][prev_cluster_id].update(cluster_products)
                    
                    # Special handling for level 2 - we need to track which level 1 clusters failed here
                    # to enable direct jumps to levels 3 and 4
                    if level_num == 2 and prev_level == 1:
                        logger.info(f"Level 1 cluster {prev_cluster_id} failed refinement at level 2. Marking for direct jump to levels 3/4.")
                        print(f"Level 1 cluster {prev_cluster_id} with {len(cluster_products)} products failed refinement at level 2.")
                    
                    # Only log for the first occurrence of this level
                    if len(failed_refinements[level_num]) == 1:
                        logger.info(f"Started tracking failed refinements at level {level_num}")
                        
                    continue
                
                # Update relationship tracking
                if prev_cluster_id not in self.cluster_relationships:
                    self.cluster_relationships[prev_cluster_id] = []
                
                # Track which products were successfully refined at this level
                refined_products = set()
                
                # Add new clusters to this level
                for subcluster_id, subcluster_info in subclusters.items():
                    # Store the new cluster
                    self.hierarchical_clusters[f"level_{level_num}"][subcluster_id] = subcluster_info
                    
                    # Update parent-child relationship
                    self.cluster_relationships[prev_cluster_id].append(subcluster_id)
                    total_clusters_created += 1
                    
                    # Track which products were refined into this subcluster
                    for product in subcluster_info.get("products", []):
                        refined_products.add(product)
                        
                        # Update the latest cluster assignment for each product
                        latest_product_clusters[product] = {
                            "level": level_num,
                            "cluster_id": subcluster_id
                        }
                        progression_stats["products_refined"] += 1
                
                # For any products that weren't refined, they keep their previous assignment
                unrefined_products = set(cluster_products) - refined_products
                if unrefined_products:
                    logger.info(f"{len(unrefined_products)} products from cluster {prev_cluster_id} weren't refined at level {level_num}")
                    print(f"{len(unrefined_products)} products from cluster {prev_cluster_id} remain at their previous level")
                    
                    # If a level 1 cluster wasn't fully refined at level 2, track it for potential direct jumps
                    if level_num == 2 and prev_level == 1 and len(unrefined_products) > 0:
                        # Add the unrefined products to failed refinements
                        if level_num not in failed_refinements:
                            failed_refinements[level_num] = {}
                        
                        if prev_cluster_id not in failed_refinements[level_num]:
                            failed_refinements[level_num][prev_cluster_id] = set()
                            
                        failed_refinements[level_num][prev_cluster_id].update(unrefined_products)
                        logger.info(f"Tracking {len(unrefined_products)} unrefined products from level 1 cluster {prev_cluster_id} for direct jumps")
                
                # Update progression statistics
                progression_stats["new_clusters_created"] += len(subclusters)
            
            # Print progression statistics
            print(f"\n=== Level {level_num} Progression Statistics ===\n")
            print(f"Clusters processed: {progression_stats['clusters_processed']}")
            print(f"Products processed: {progression_stats['products_processed']}")
            print(f"New clusters created: {progression_stats['new_clusters_created']}")
            print(f"Products refined: {progression_stats['products_refined']}")
            print(f"Total clusters at level {level_num}: {total_clusters_created}")
            
            # Report on failed refinements
            if level_num in failed_refinements:
                total_failed_products = sum(len(products) for products in failed_refinements[level_num].values())
                print(f"Failed refinements at level {level_num}: {len(failed_refinements[level_num])} clusters with {total_failed_products} products")
                logger.info(f"Failed refinements at level {level_num}: {len(failed_refinements[level_num])} clusters with {total_failed_products} products")
            print("\n")
            
            # Save the hierarchical clusters after each level
            self._save_hierarchical_clusters()
            
        # Return the hierarchical clustering results
        return self.hierarchical_clusters
    
    def _cluster_at_level(self, product_codes: List[str], embeddings: np.ndarray, 
                         level_config: Dict[str, Any], parent_id: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """
        Run clustering at a specific level of the hierarchy.
        
        Args:
            product_codes: List of product codes to cluster
            embeddings: Embeddings for the products
            level_config: Configuration for this level
            parent_id: ID of the parent cluster (None for level 1)
            
        Returns:
            Dictionary of clusters at this level
        """
        level_name = level_config["name"]
        level_num = level_config["level"]
        
        print(f"\n=== Starting Clustering at Level {level_num} ({level_name}) ===")
        print(f"Products to cluster: {len(product_codes)}")
        print(f"HDBSCAN Parameters: min_cluster_size={level_config['min_cluster_size']}, min_samples={level_config['min_samples']}, epsilon={level_config.get('epsilon', 0.0)}, alpha={level_config.get('alpha', 1.0)}")
        logger.info(f"Clustering at level {level_num} ({level_name}) with {len(product_codes)} products")
        logger.info(f"Parameters: min_cluster_size={level_config['min_cluster_size']}, min_samples={level_config['min_samples']}")
        
        # Create a temporary directory for this level's clustering output
        level_output_dir = os.path.join(self.output_dir, f"level_{level_num}")
        os.makedirs(level_output_dir, exist_ok=True)
        
        # Save temporary embeddings and product codes for this subset
        # Note: Use 'product_embeddings.npy' and 'product_codes.txt' to match what improved_clustering.py expects
        temp_embeddings_path = os.path.join(level_output_dir, "product_embeddings.npy")
        temp_codes_path = os.path.join(level_output_dir, "product_codes.txt")
        
        np.save(temp_embeddings_path, embeddings)
        with open(temp_codes_path, 'w') as f:
            for code in product_codes:
                f.write(f"{code}\n")
                
        # Use fixed parameters from config file (no adaptive adjustment)
        print("\n--- Starting HDBSCAN clustering process ---")
        print(f"Embedding dimensionality: {embeddings.shape[1]}")
        print(f"Using metric: {level_config.get('metric', 'euclidean')}")
        
        # Are we using categories for clustering?
        use_cats = not level_config.get("no_categories", True)
        print(f"Using product categories for clustering: {'Yes' if use_cats else 'No'}")
        
        # Run improved clustering with this level's parameters
        print("Starting HDBSCAN algorithm - this might take a few minutes for large datasets...")
        run_improved_clustering(
            data_dir=level_output_dir,
            metric=level_config.get("metric", "euclidean"),
            min_cluster_size=level_config.get("min_cluster_size", 3),
            min_samples=level_config.get("min_samples", 2),
            cluster_selection_epsilon=level_config.get("epsilon", 0.0),
            alpha=level_config.get("alpha", 1.0),
            cluster_selection_method=level_config.get("cluster_selection_method", "eom"),
            use_categories=use_cats,
            category_exclusivity=level_config.get("category_exclusivity", 0.0),
            force=True,
            n_jobs=self.global_settings.get("n_jobs", -1)
        )
        
        # Load the resulting clusters
        clusters_path = os.path.join(level_output_dir, "improved_clustering", "clusters.json")
        
        try:
            with open(clusters_path, 'r') as f:
                clusters = json.load(f)
                print(f"\n=== HDBSCAN Results for Level {level_num} ===")
                print(f"Created {len(clusters)} clusters containing {sum(len(p) for p in clusters.values())} products")
                print(f"Inclusion rate: {sum(len(p) for p in clusters.values()) / len(product_codes) * 100:.1f}%")
                print(f"Noise points: {len(product_codes) - sum(len(p) for p in clusters.values())}")
                if len(clusters) > 0:
                    avg_size = sum(len(p) for p in clusters.values()) / len(clusters)
                    print(f"Average cluster size: {avg_size:.1f} products")
                    print(f"Largest cluster size: {max(len(p) for p in clusters.values()) if clusters else 0} products")
        except FileNotFoundError:
            logger.error(f"Clusters file not found at {clusters_path}")
            print("\n!!! ERROR: Clustering failed to produce output file !!!")
            return {}
        
        # Transform into our hierarchical format
        hierarchical_clusters = {}
        
        for cluster_id, products in clusters.items():
            # Create a unique ID for this cluster that includes level and position
            unique_id = f"{level_name}_{cluster_id.split('_')[-1]}"
            if parent_id:
                parent_prefix = parent_id.split('_')[0]  # Get the level name of parent
                unique_id = f"{parent_prefix}_{level_name}_{cluster_id.split('_')[-1]}"
            
            hierarchical_clusters[unique_id] = {
                "level": level_num,
                "parent": parent_id,
                "products": products,
                "children": [],
                "metadata": {
                    "cluster_size": len(products),
                    "parameters": {
                        "min_cluster_size": level_config.get("min_cluster_size"),
                        "min_samples": level_config.get("min_samples"),
                        "epsilon": level_config.get("epsilon"),
                        "alpha": level_config.get("alpha")
                    }
                }
            }
        
        logger.info(f"Created {len(hierarchical_clusters)} clusters at level {level_num}")
        print(f"\n=== Completed Level {level_num} Clustering ===\n")
        return hierarchical_clusters
    
    # We removed the adaptive_clustering method to simplify implementation
    # Instead, we rely on well-tuned parameters in the config file
    
    def _check_cluster_cohesion(self, cluster_info: Dict[str, Any]) -> bool:
        """
        Check if a cluster is cohesive enough to stop further subdivision using actual
        cohesion metrics based on embedding similarity.
        
        Args:
            cluster_info: Information about the cluster
            
        Returns:
            True if the cluster is cohesive enough, False otherwise
        """
        # If the cluster has metadata about its cohesion, use that
        if "metadata" in cluster_info and "cohesion_score" in cluster_info["metadata"]:
            cohesion_threshold = self.progression_rules.get("min_coherence_threshold", 0.6)
            cohesion_score = cluster_info["metadata"]["cohesion_score"]
            return cohesion_score >= cohesion_threshold
        
        # Don't automatically mark small clusters as cohesive - let them progress
        # Only consider very tiny clusters (singleton) as automatically cohesive
        products = cluster_info.get("products", [])
        
        # Only singleton clusters are automatically cohesive since they can't be subdivided
        if len(products) <= 1:
            logger.info(f"Singleton cluster with {len(products)} product automatically marked as cohesive")
            print(f"Auto-cohesive: Cluster has only {len(products)} product (cannot be subdivided)")
            return True
            
        # For clusters with 2+ products, allow progression regardless of size
        return False
            
    def _calculate_cohesion_metrics(self, embeddings: np.ndarray) -> bool:
        """
        Calculate actual cohesion metrics for a set of embeddings.
        
        Args:
            embeddings: Matrix of embeddings for products in a cluster
            
        Returns:
            True if the cluster is cohesive enough, False otherwise
        """
        # 1. Calculate pairwise cosine similarities
        # Normalize embeddings for cosine similarity calculation
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        pairwise_similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # 2. Calculate silhouette-like score using the mean intra-cluster similarity
        # Since we only have one cluster, we use the mean pairwise similarity as our cohesion measure
        # Remove the diagonal (self-similarities)
        n = pairwise_similarities.shape[0]
        mask = ~np.eye(n, dtype=bool)
        mean_similarity = pairwise_similarities[mask].mean()
        
        # 3. Calculate standard deviation of similarities to assess uniformity
        std_similarity = pairwise_similarities[mask].std()
        
        # 4. Calculate the minimum similarity to detect outliers
        min_similarity = pairwise_similarities[mask].min()
        
        # Log the cohesion metrics
        logger.info(f"Cluster cohesion metrics: mean_similarity={mean_similarity:.3f}, " 
                  f"std_similarity={std_similarity:.3f}, min_similarity={min_similarity:.3f}")
        
        # Thresholds for good cohesion
        # These can be tuned based on the specific data and requirements
        mean_similarity_threshold = self.progression_rules.get("mean_similarity_threshold", 0.7)
        std_similarity_threshold = self.progression_rules.get("std_similarity_threshold", 0.15)
        min_similarity_threshold = self.progression_rules.get("min_similarity_threshold", 0.4)
        
        # Check if the cluster meets all cohesion criteria
        is_cohesive = (mean_similarity >= mean_similarity_threshold and 
                      std_similarity <= std_similarity_threshold and 
                      min_similarity >= min_similarity_threshold)
        
        if is_cohesive:
            logger.info(f"Cluster is cohesive (mean={mean_similarity:.3f}, std={std_similarity:.3f}, min={min_similarity:.3f})")
        else:
            logger.info(f"Cluster is not cohesive enough for stopping criteria")
            
        return is_cohesive
    
    def _refine_with_cross_encoder(self, products: List[str], descriptions: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Refine clusters using cross-encoder for more accurate pairwise similarity assessment.
        This is particularly helpful for distinguishing between very similar products with minor variations.
        
        Args:
            products: List of product codes to refine
            descriptions: Optional list of product descriptions, if not provided will be loaded
            
        Returns:
            Dictionary mapping refined cluster IDs to lists of product codes
        """
        if not self.use_cross_encoder or not CROSS_ENCODER_AVAILABLE:
            logger.warning("Cross-encoder refinement requested but not available or enabled")
            # Return single cluster with all products if cross-encoder not available
            return {"0": products}
            
        # Load product descriptions if not provided
        if descriptions is None:
            try:
                # Try to load from transaction data
                transactions_path = os.path.join(self.data_dir, "processed_transactions.csv")
                df = pd.read_csv(transactions_path)
                product_df = df[df['product_code'].isin(products)].drop_duplicates('product_code')
                product_lookup = dict(zip(product_df['product_code'], product_df['description']))
                descriptions = [product_lookup.get(p, f"Unknown product {p}") for p in products]
            except Exception as e:
                logger.error(f"Error loading product descriptions for cross-encoder: {e}")
                # Return single cluster if descriptions can't be loaded
                return {"0": products}
        
        if len(products) <= 1:
            return {"0": products}
            
        # Create all pairs of products for comparison
        pairs = []
        pair_indices = []
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                pairs.append([descriptions[i], descriptions[j]])
                pair_indices.append((i, j))
                
        # Predict similarities with cross-encoder
        logger.info(f"Computing pairwise similarities for {len(pairs)} product pairs using cross-encoder")
        try:
            similarities = self.cross_encoder.predict(pairs, batch_size=self.cross_encoder_batch_size)
        except Exception as e:
            logger.error(f"Error computing cross-encoder similarities: {e}")
            return {"0": products}
            
        # Build similarity matrix
        n = len(products)
        similarity_matrix = np.zeros((n, n))
        for k, (i, j) in enumerate(pair_indices):
            similarity = similarities[k]
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric
            
        # Set diagonal to 1 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Create an adjacency matrix based on threshold
        adjacency = similarity_matrix < self.cross_encoder_threshold
        
        # Use connected components to identify clusters
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(1 - adjacency)
        
        # Group products by cluster label
        refined_clusters = {}
        for i, label in enumerate(labels):
            label_str = str(label)
            if label_str not in refined_clusters:
                refined_clusters[label_str] = []
            refined_clusters[label_str].append(products[i])
            
        logger.info(f"Cross-encoder refinement created {len(refined_clusters)} clusters from {len(products)} products")
        for cluster_id, cluster_products in refined_clusters.items():
            logger.info(f"  Refined cluster {cluster_id}: {len(cluster_products)} products")
            
        return refined_clusters
    
    def _save_hierarchical_clusters(self):
        """Save the hierarchical clustering results to a JSON file."""
        output_path = os.path.join(self.output_dir, "hierarchical_clusters.json")
        
        # Save the full hierarchical structure
        with open(output_path, 'w') as f:
            json.dump(self.hierarchical_clusters, f, indent=2)
        
        # Also save relationships separately for easier visualization
        relationships_path = os.path.join(self.output_dir, "cluster_relationships.json")
        with open(relationships_path, 'w') as f:
            json.dump(dict(self.cluster_relationships), f, indent=2)
        
        logger.info(f"Saved hierarchical clusters to {output_path}")
        logger.info(f"Saved cluster relationships to {relationships_path}")
        
        # Save a more user-friendly version that shows the product descriptions
        try:
            # Try to load product descriptions
            transactions_path = os.path.join(self.data_dir, "processed_transactions.csv")
            if os.path.exists(transactions_path):
                df = pd.read_csv(transactions_path)
                product_lookup = dict(zip(df['product_code'], df['description']))
                
                # Create readable version with descriptions
                readable_clusters = {}
                for level, clusters in self.hierarchical_clusters.items():
                    readable_clusters[level] = {}
                    for cluster_id, cluster_info in clusters.items():
                        readable_info = {**cluster_info}
                        products_with_desc = []
                        for product in cluster_info.get("products", []):
                            desc = product_lookup.get(product, "Unknown")
                            products_with_desc.append({"code": product, "description": desc})
                        readable_info["products_with_desc"] = products_with_desc
                        readable_clusters[level][cluster_id] = readable_info
                
                readable_path = os.path.join(self.output_dir, "hierarchical_clusters_readable.json")
                with open(readable_path, 'w') as f:
                    json.dump(readable_clusters, f, indent=2)
                logger.info(f"Saved readable hierarchical clusters to {readable_path}")
        except Exception as e:
            logger.warning(f"Could not create readable clusters with descriptions: {e}")
            pass


def run_hierarchical_clustering_pipeline(config_path: str, data_dir: Optional[str] = None):
    """
    Run the complete hierarchical clustering pipeline.
    
    Args:
        config_path: Path to the hierarchical configuration JSON file
        data_dir: Directory containing data files. If None, uses default.
    """
    clusterer = HierarchicalClusterer(config_path, data_dir)
    results = clusterer.run_hierarchical_clustering()
    
    # Output summary statistics
    level_stats = {}
    for level_name, clusters in results.items():
        level_num = int(level_name.split('_')[1])
        num_clusters = len(clusters)
        total_products = sum(len(info.get("products", [])) for info in clusters.values())
        avg_cluster_size = total_products / num_clusters if num_clusters > 0 else 0
        
        level_stats[level_name] = {
            "num_clusters": num_clusters,
            "total_products": total_products,
            "avg_cluster_size": avg_cluster_size
        }
    
    # Print summary
    logger.info("Hierarchical Clustering Summary:")
    for level_name, stats in sorted(level_stats.items()):
        logger.info(f"{level_name}: {stats['num_clusters']} clusters, "
                  f"avg size: {stats['avg_cluster_size']:.1f} products")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hierarchical multi-level clustering")
    parser.add_argument("--config", default="hierarchical_config.json",
                        help="Path to hierarchical configuration file")
    parser.add_argument("--data_dir", help="Directory containing data files")
    
    args = parser.parse_args()
    
    # Resolve config path if it's not absolute
    if not os.path.isabs(args.config):
        args.config = os.path.join(os.path.dirname(__file__), args.config)
    
    run_hierarchical_clustering_pipeline(args.config, args.data_dir)
