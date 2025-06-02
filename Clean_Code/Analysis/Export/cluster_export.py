"""
Cluster Export Utilities for VectorDB

This module provides functions for exporting clustering results to various formats
and generating summary statistics and analysis reports.
"""

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

def export_clusters_to_csv(
    results: Dict[str, Any], 
    df: pd.DataFrame, 
    output_dir: str
) -> str:
    """
    Export clustering results to CSV format.
    
    Args:
        results: Dictionary containing clustering results
        df: Original dataframe with product data
        output_dir: Directory to save the output file
        
    Returns:
        Path to the exported CSV file
    """
    logger.info("Exporting cluster assignments to CSV...")
    
    # Copy the dataframe to avoid modifying the original
    df_export = df.copy()
    
    # Debug info about results structure
    logger.debug(f"Results keys: {list(results.keys())}")
    
    # Get clustering paths for each item - check for different possible keys
    clustering_paths = results.get('clustering_paths', results.get('clustering_path', []))
    
    # If paths are still empty, initialize with proper length
    if not clustering_paths:
        logger.warning("No clustering paths found in results, initializing empty paths")
        clustering_paths = [None] * len(df_export)
    
    # Determine max level
    max_level = results.get('max_level', 3)
    
    # Add cluster assignments for each level
    for level in range(1, max_level + 1):
        level_key = f'level_{level}'
        
        if level_key in results and 'labels' in results[level_key]:
            level_labels = results[level_key]['labels']
            
            # Create the cluster column if it doesn't exist
            if f'cluster_level_{level}' not in df_export.columns:
                df_export[f'cluster_level_{level}'] = -1  # Default to noise
            
            # Update from level labels directly if available
            if len(level_labels) == len(df_export):
                df_export[f'cluster_level_{level}'] = level_labels
            else:
                # Otherwise, extract from clustering paths
                for i, path in enumerate(clustering_paths):
                    if i < len(df_export) and isinstance(path, dict) and level in path:
                        df_export.loc[i, f'cluster_level_{level}'] = path[level]
    
    # Ensure all cluster level columns have proper names and default values
    for level in range(1, max_level + 1):
        col_name = f'cluster_level_{level}'
        if col_name not in df_export.columns:
            df_export[col_name] = -1  # -1 indicates no cluster assignment (noise)
    
    # Remove any old cluster_path column if it exists
    if 'cluster_path' in df_export.columns:
        df_export.drop(columns=['cluster_path'], inplace=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hierarchical_clustering_results_{timestamp}.csv"
    csv_path = os.path.join(output_dir, filename)
    
    # Save to CSV
    df_export.to_csv(csv_path, index=False)
    logger.info(f"Exported cluster assignments to {csv_path}")
    
    return csv_path


def generate_cluster_summary(
    results: Dict[str, Any], 
    df: pd.DataFrame, 
    output_dir: str
) -> str:
    """
    Generate a summary of clustering results including statistics and metrics.
    
    Args:
        results: Dictionary containing clustering results
        df: Original dataframe with product data
        output_dir: Directory to save the output file
        
    Returns:
        Path to the summary file
    """
    logger.info("Generating cluster summary...")
    
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_size": len(df),
        "levels": {}
    }
    
    # Get max level
    max_level = results.get('max_level', 3)
    
    # Generate summary for each level
    for level in range(1, max_level + 1):
        level_key = f'level_{level}'
        
        if level_key in results:
            level_results = results[level_key]
            labels = level_results.get('labels', [])
            
            # Skip if no labels
            if not labels:
                continue
                
            # Count clusters and points assigned
            unique_clusters = set(labels)
            if -1 in unique_clusters:  # Remove noise cluster
                unique_clusters.remove(-1)
                
            cluster_count = len(unique_clusters)
            assigned_count = sum(1 for l in labels if l != -1)
            noise_count = sum(1 for l in labels if l == -1)
            
            # Calculate metrics
            noise_ratio = noise_count / len(labels) if labels else 0
            assigned_ratio = assigned_count / len(labels) if labels else 0
            
            # Cluster sizes
            cluster_sizes = {}
            for cluster_id in unique_clusters:
                cluster_sizes[str(cluster_id)] = sum(1 for l in labels if l == cluster_id)
            
            # Store in summary
            summary["levels"][level_key] = {
                "cluster_count": cluster_count,
                "assigned_count": assigned_count,
                "noise_count": noise_count,
                "noise_ratio": noise_ratio,
                "assigned_ratio": assigned_ratio,
                "cluster_sizes": cluster_sizes,
                "method": level_results.get("method", "unknown")
            }
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clustering_summary_{timestamp}.json"
    summary_path = os.path.join(output_dir, filename)
    
    # Save to JSON
    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles NumPy types."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Generated cluster summary at {summary_path}")
    
    return summary_path
