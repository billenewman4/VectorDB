"""
Evaluation metrics for clustering results.
Provides functions to evaluate clustering quality using various metrics.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score
)
from sklearn.preprocessing import normalize
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusterEvaluator:
    """
    Evaluation tools for assessing clustering quality.
    
    This class provides methods to evaluate clustering results using various
    internal and external metrics, as well as utilities for cluster analysis.
    """
    
    @staticmethod
    def internal_metrics(
        vectors: np.ndarray, 
        labels: np.ndarray, 
        metric: str = 'cosine'
    ) -> Dict[str, float]:
        """
        Calculate internal clustering quality metrics (no ground truth required).
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            labels: Array of cluster labels
            metric: Distance metric for silhouette score ('euclidean', 'cosine', etc.)
            
        Returns:
            Dictionary of metric names and values
            
        Raises:
            ValueError: If inputs are invalid or no valid clusters are found
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a numpy array")
        
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        if len(vectors) != len(labels):
            raise ValueError(f"Length mismatch: vectors has {len(vectors)} items, but labels has {len(labels)} items")
        
        # Initialize metrics dictionary
        metrics = {
            'num_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
            'num_samples': len(labels),
            'noise_points': np.sum(labels == -1),
            'noise_percentage': np.sum(labels == -1) / len(labels) * 100
        }
        
        # Check if we have enough data for computing metrics
        if metrics['num_clusters'] < 2:
            logger.warning("Need at least 2 clusters for internal metrics")
            metrics.update({
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None
            })
            return metrics
        
        # Filter out noise points for metrics calculation
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) <= 1:
            logger.warning("Not enough non-noise points for internal metrics")
            metrics.update({
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None
            })
            return metrics
            
        non_noise_vectors = vectors[non_noise_mask]
        non_noise_labels = labels[non_noise_mask]
        
        # Check if we still have enough unique labels after removing noise
        if len(np.unique(non_noise_labels)) < 2:
            logger.warning("Not enough unique clusters after removing noise")
            metrics.update({
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None
            })
            return metrics
        
        # Calculate metrics
        try:
            # Normalize vectors if using cosine metric
            if metric == 'cosine':
                non_noise_vectors = normalize(non_noise_vectors)
                
            # Silhouette score (higher is better, range: -1 to 1)
            s_score = silhouette_score(non_noise_vectors, non_noise_labels, metric=metric)
            metrics['silhouette_score'] = s_score
            
            # Davies-Bouldin score (lower is better, >= 0)
            db_score = davies_bouldin_score(non_noise_vectors, non_noise_labels)
            metrics['davies_bouldin_score'] = db_score
            
            # Calinski-Harabasz score (higher is better, >= 0)
            ch_score = calinski_harabasz_score(non_noise_vectors, non_noise_labels)
            metrics['calinski_harabasz_score'] = ch_score
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics.update({
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None,
                'error': str(e)
            })
        
        return metrics
    
    @staticmethod
    def external_metrics(
        labels_true: np.ndarray, 
        labels_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate external clustering metrics (requires ground truth).
        
        Args:
            labels_true: Ground truth cluster labels
            labels_pred: Predicted cluster labels
            
        Returns:
            Dictionary of metric names and values
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(labels_true, np.ndarray):
            labels_true = np.array(labels_true)
            
        if not isinstance(labels_pred, np.ndarray):
            labels_pred = np.array(labels_pred)
            
        if len(labels_true) != len(labels_pred):
            raise ValueError(f"Length mismatch: labels_true has {len(labels_true)} items, but labels_pred has {len(labels_pred)} items")
        
        # Initialize metrics dictionary
        metrics = {}
        
        try:
            # Adjusted Rand Index (higher is better, range: -1 to 1)
            ari = adjusted_rand_score(labels_true, labels_pred)
            metrics['adjusted_rand_score'] = ari
            
            # Adjusted Mutual Information (higher is better, range: 0 to 1)
            ami = adjusted_mutual_info_score(labels_true, labels_pred)
            metrics['adjusted_mutual_info_score'] = ami
            
            # Homogeneity (higher is better, range: 0 to 1)
            # Each cluster contains only members of a single class
            homogeneity = homogeneity_score(labels_true, labels_pred)
            metrics['homogeneity_score'] = homogeneity
            
            # Completeness (higher is better, range: 0 to 1)
            # All members of a given class are assigned to the same cluster
            completeness = completeness_score(labels_true, labels_pred)
            metrics['completeness_score'] = completeness
            
            # V-measure (higher is better, range: 0 to 1)
            # Harmonic mean of homogeneity and completeness
            v_measure = v_measure_score(labels_true, labels_pred)
            metrics['v_measure_score'] = v_measure
            
            # Fowlkes-Mallows score (higher is better, range: 0 to 1)
            # Geometric mean of precision and recall
            fm_score = fowlkes_mallows_score(labels_true, labels_pred)
            metrics['fowlkes_mallows_score'] = fm_score
            
        except Exception as e:
            logger.error(f"Error calculating external metrics: {str(e)}")
            metrics['error'] = str(e)
        
        return metrics
    
    @staticmethod
    def compare_clusterings(
        vectors: np.ndarray,
        labels_list: List[np.ndarray],
        labels_names: List[str],
        ground_truth: Optional[np.ndarray] = None,
        ground_truth_name: str = "Ground Truth"
    ) -> pd.DataFrame:
        """
        Compare multiple clustering results using various metrics.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            labels_list: List of cluster label arrays from different algorithms
            labels_names: Names of the clustering algorithms
            ground_truth: Optional ground truth labels for external metrics
            ground_truth_name: Name of the ground truth clustering
            
        Returns:
            DataFrame with comparison metrics for each clustering algorithm
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(labels_list) != len(labels_names):
            raise ValueError("labels_list and labels_names must have the same length")
        
        for i, labels in enumerate(labels_list):
            if len(vectors) != len(labels):
                raise ValueError(f"Length mismatch for {labels_names[i]}: vectors has {len(vectors)} items, but labels has {len(labels)} items")
        
        # Initialize results
        results = []
        
        # Evaluate each clustering
        for labels, name in zip(labels_list, labels_names):
            # Calculate internal metrics
            internal = ClusterEvaluator.internal_metrics(vectors, labels)
            
            # Create result dictionary
            result = {
                'Algorithm': name,
                'Clusters': internal['num_clusters'],
                'Noise Points': internal['noise_points'],
                'Noise %': f"{internal['noise_percentage']:.2f}%",
                'Silhouette': internal['silhouette_score'],
                'Davies-Bouldin': internal['davies_bouldin_score'],
                'Calinski-Harabasz': internal['calinski_harabasz_score']
            }
            
            # Add external metrics if ground truth is provided
            if ground_truth is not None:
                external = ClusterEvaluator.external_metrics(ground_truth, labels)
                result.update({
                    'ARI': external['adjusted_rand_score'],
                    'AMI': external['adjusted_mutual_info_score'],
                    'Homogeneity': external['homogeneity_score'],
                    'Completeness': external['completeness_score'],
                    'V-measure': external['v_measure_score'],
                    'FM-score': external['fowlkes_mallows_score']
                })
            
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Format floating point numbers
        for col in df.columns:
            if col not in ['Algorithm', 'Clusters', 'Noise Points', 'Noise %']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        
        return df
    
    @staticmethod
    def analyze_cluster_stability(
        vectors: np.ndarray,
        clusterer: Any,
        n_runs: int = 10,
        subsample_ratio: float = 0.8,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze the stability of clustering results by multiple runs with subsampling.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            clusterer: Clusterer object with fit_predict method
            n_runs: Number of clustering runs
            subsample_ratio: Ratio of data to use in each run
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with stability analysis results
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not hasattr(clusterer, 'fit_predict') or not callable(getattr(clusterer, 'fit_predict')):
            raise ValueError("Clusterer must have a fit_predict method")
        
        if subsample_ratio <= 0 or subsample_ratio > 1:
            raise ValueError(f"subsample_ratio must be in (0, 1], got {subsample_ratio}")
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize results
        all_num_clusters = []
        all_noise_percentages = []
        all_silhouette_scores = []
        all_rand_scores = []  # For comparing runs against each other
        
        # Run clustering multiple times
        for run in range(n_runs):
            # Subsample data
            n_samples = int(len(vectors) * subsample_ratio)
            indices = np.random.choice(len(vectors), size=n_samples, replace=False)
            subsample = vectors[indices]
            
            # Run clustering
            results = clusterer.fit_predict(subsample)
            labels = results['labels'] if isinstance(results, dict) and 'labels' in results else results
            
            # Calculate metrics
            metrics = ClusterEvaluator.internal_metrics(subsample, labels)
            
            # Store results
            all_num_clusters.append(metrics['num_clusters'])
            all_noise_percentages.append(metrics['noise_percentage'])
            if metrics['silhouette_score'] is not None:
                all_silhouette_scores.append(metrics['silhouette_score'])
            
            # Compare with previous runs
            if run > 0:
                for prev_run, prev_labels in enumerate(all_labels):
                    # Create a mapping between current and previous run
                    # For points that were selected in both runs
                    common_indices_curr = []
                    common_indices_prev = []
                    
                    for i, idx_curr in enumerate(indices):
                        if idx_curr in all_indices[prev_run]:
                            j = np.where(all_indices[prev_run] == idx_curr)[0][0]
                            common_indices_curr.append(i)
                            common_indices_prev.append(j)
                    
                    if len(common_indices_curr) > 1:
                        # Compare labels for common points
                        labels_curr = labels[common_indices_curr]
                        labels_prev = prev_labels[common_indices_prev]
                        
                        # Calculate ARI between runs
                        try:
                            ari = adjusted_rand_score(labels_prev, labels_curr)
                            all_rand_scores.append(ari)
                        except Exception as e:
                            logger.warning(f"Error calculating ARI between runs {prev_run} and {run}: {e}")
            
            # Store labels and indices for comparison with future runs
            if run == 0:
                all_labels = [labels]
                all_indices = [indices]
            else:
                all_labels.append(labels)
                all_indices.append(indices)
        
        # Calculate stability metrics
        stability_results = {
            'n_runs': n_runs,
            'subsample_ratio': subsample_ratio,
            'num_clusters': {
                'mean': np.mean(all_num_clusters),
                'std': np.std(all_num_clusters),
                'min': np.min(all_num_clusters),
                'max': np.max(all_num_clusters),
                'values': all_num_clusters
            },
            'noise_percentage': {
                'mean': np.mean(all_noise_percentages),
                'std': np.std(all_noise_percentages),
                'min': np.min(all_noise_percentages),
                'max': np.max(all_noise_percentages),
                'values': all_noise_percentages
            }
        }
        
        # Add silhouette score statistics if available
        if all_silhouette_scores:
            stability_results['silhouette_score'] = {
                'mean': np.mean(all_silhouette_scores),
                'std': np.std(all_silhouette_scores),
                'min': np.min(all_silhouette_scores),
                'max': np.max(all_silhouette_scores),
                'values': all_silhouette_scores
            }
        
        # Add ARI between runs if available
        if all_rand_scores:
            stability_results['adjusted_rand_score_between_runs'] = {
                'mean': np.mean(all_rand_scores),
                'std': np.std(all_rand_scores),
                'min': np.min(all_rand_scores),
                'max': np.max(all_rand_scores),
                'values': all_rand_scores
            }
        
        return stability_results
    
    @staticmethod
    def analyze_clusters(
        vectors: np.ndarray,
        labels: np.ndarray,
        data: Optional[List[Any]] = None,
        n_features_to_analyze: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze cluster characteristics, including centroid distances and feature importance.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            labels: Array of cluster labels
            data: Optional associated data for each vector
            n_features_to_analyze: Number of top features to analyze per cluster
            
        Returns:
            Dictionary with cluster analysis results
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(vectors) != len(labels):
            raise ValueError(f"Length mismatch: vectors has {len(vectors)} items, but labels has {len(labels)} items")
        
        if data is not None and len(data) != len(vectors):
            raise ValueError(f"Length mismatch: vectors has {len(vectors)} items, but data has {len(data)} items")
        
        # Get unique cluster labels
        unique_labels = np.unique(labels)
        unique_non_noise = [l for l in unique_labels if l != -1]
        
        # Initialize results
        cluster_analysis = {
            'num_clusters': len(unique_non_noise),
            'clusters': {},
            'inter_cluster_distances': {},
            'overall': {
                'total_points': len(vectors),
                'noise_points': np.sum(labels == -1),
                'noise_percentage': np.sum(labels == -1) / len(labels) * 100
            }
        }
        
        # Calculate cluster centroids
        centroids = {}
        for label in unique_non_noise:
            cluster_vectors = vectors[labels == label]
            centroids[label] = np.mean(cluster_vectors, axis=0)
        
        # Calculate inter-cluster distances
        distances = {}
        for i, label1 in enumerate(unique_non_noise):
            for label2 in unique_non_noise[i+1:]:
                # Use cosine similarity (1 - cosine distance)
                centroid1 = centroids[label1] / np.linalg.norm(centroids[label1])
                centroid2 = centroids[label2] / np.linalg.norm(centroids[label2])
                similarity = np.dot(centroid1, centroid2)
                distance = 1 - similarity
                
                distances[(label1, label2)] = distance
        
        # Sort distances
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        
        # Store in results
        cluster_analysis['inter_cluster_distances'] = {
            'closest_pair': {
                'clusters': sorted_distances[0][0],
                'distance': sorted_distances[0][1]
            },
            'furthest_pair': {
                'clusters': sorted_distances[-1][0],
                'distance': sorted_distances[-1][1]
            },
            'all_distances': {f"{k[0]}-{k[1]}": v for k, v in sorted_distances}
        }
        
        # Analyze each cluster
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_vectors = vectors[cluster_indices]
            
            # Skip empty clusters
            if len(cluster_vectors) == 0:
                continue
            
            # Calculate cluster statistics
            cluster_info = {
                'size': len(cluster_indices),
                'percentage': len(cluster_indices) / len(vectors) * 100
            }
            
            # Add centroid and intra-cluster distance for non-noise clusters
            if label != -1:
                centroid = centroids[label]
                cluster_info['centroid'] = centroid.tolist()
                
                # Calculate average distance to centroid
                normalized_vectors = normalize(cluster_vectors)
                normalized_centroid = centroid / np.linalg.norm(centroid)
                similarities = np.dot(normalized_vectors, normalized_centroid)
                distances_to_centroid = 1 - similarities
                
                cluster_info['avg_distance_to_centroid'] = np.mean(distances_to_centroid)
                cluster_info['max_distance_to_centroid'] = np.max(distances_to_centroid)
                cluster_info['min_distance_to_centroid'] = np.min(distances_to_centroid)
                
                # Find most central and most outlier points
                most_central_idx = np.argmin(distances_to_centroid)
                most_outlier_idx = np.argmax(distances_to_centroid)
                
                cluster_info['most_central_point_idx'] = int(cluster_indices[most_central_idx])
                cluster_info['most_outlier_point_idx'] = int(cluster_indices[most_outlier_idx])
                
                # Add data if available
                if data is not None:
                    cluster_data = [data[i] for i in cluster_indices]
                    cluster_info['most_central_point_data'] = data[cluster_indices[most_central_idx]]
                    cluster_info['most_outlier_point_data'] = data[cluster_indices[most_outlier_idx]]
                    
                    # Store a sample of data points
                    sample_size = min(5, len(cluster_indices))
                    central_indices = np.argsort(distances_to_centroid)[:sample_size]
                    cluster_info['sample_data'] = [data[cluster_indices[i]] for i in central_indices]
            
            # Store in results
            cluster_analysis['clusters'][int(label)] = cluster_info
        
        return cluster_analysis
    
    @staticmethod
    def optimal_clusters(
        vectors: np.ndarray,
        clusterer_factory: Callable[[int], Any],
        k_range: range,
        criterion: str = 'silhouette'
    ) -> Dict[str, Any]:
        """
        Determine the optimal number of clusters using various criteria.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            clusterer_factory: Function that takes k and returns a clusterer
            k_range: Range of k values to try
            criterion: Criterion to optimize ('silhouette', 'davies_bouldin', 'calinski_harabasz')
            
        Returns:
            Dictionary with optimization results
            
        Raises:
            ValueError: If inputs are invalid or criterion is unsupported
        """
        if not callable(clusterer_factory):
            raise ValueError("clusterer_factory must be callable")
        
        # Validate criterion
        valid_criteria = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        if criterion not in valid_criteria:
            raise ValueError(f"criterion must be one of {valid_criteria}, got {criterion}")
        
        # Initialize results
        results = {
            'k_values': list(k_range),
            'scores': [],
            'best_k': None,
            'best_score': None,
            'criterion': criterion
        }
        
        # Try each k value
        for k in k_range:
            # Get clusterer for this k
            clusterer = clusterer_factory(k)
            
            # Run clustering
            try:
                results_k = clusterer.fit_predict(vectors)
                labels = results_k['labels'] if isinstance(results_k, dict) and 'labels' in results_k else results_k
                
                # Calculate metrics
                metrics = ClusterEvaluator.internal_metrics(vectors, labels)
                
                # Get score based on criterion
                if criterion == 'silhouette':
                    score = metrics['silhouette_score']
                elif criterion == 'davies_bouldin':
                    score = metrics['davies_bouldin_score']
                    # Davies-Bouldin: lower is better, so negate for consistent comparison
                    score = -score if score is not None else None
                elif criterion == 'calinski_harabasz':
                    score = metrics['calinski_harabasz_score']
                
                # Store score
                results['scores'].append(score)
                
                # Update best if this is better
                if (results['best_score'] is None or 
                    (score is not None and score > results['best_score'])):
                    results['best_k'] = k
                    results['best_score'] = score
                    
            except Exception as e:
                logger.warning(f"Error evaluating k={k}: {str(e)}")
                results['scores'].append(None)
        
        return results


# Simple usage example
if __name__ == "__main__":
    # Create test data
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans, AgglomerativeClustering
    import hdbscan
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    # Run different clustering algorithms
    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
    y_kmeans = kmeans.labels_
    
    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=4).fit(X)
    y_agglo = agglo.labels_
    
    # HDBSCAN
    hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5).fit(X)
    y_hdbscan = hdb.labels_
    
    # Evaluate a single clustering
    kmeans_metrics = ClusterEvaluator.internal_metrics(X, y_kmeans)
    print(f"K-Means internal metrics: {kmeans_metrics}")
    
    # Compare with ground truth (external metrics)
    kmeans_external = ClusterEvaluator.external_metrics(y_true, y_kmeans)
    print(f"K-Means external metrics: {kmeans_external}")
    
    # Compare different clustering algorithms
    comparison = ClusterEvaluator.compare_clusterings(
        vectors=X,
        labels_list=[y_kmeans, y_agglo, y_hdbscan],
        labels_names=["K-Means", "Agglomerative", "HDBSCAN"],
        ground_truth=y_true,
        ground_truth_name="Ground Truth"
    )
    print("\nClustering Comparison:")
    print(comparison)
    
    # Find optimal number of clusters
    def kmeans_factory(k):
        return KMeans(n_clusters=k, random_state=42, n_init=10)
    
    optimization = ClusterEvaluator.optimal_clusters(
        vectors=X,
        clusterer_factory=kmeans_factory,
        k_range=range(2, 10),
        criterion='silhouette'
    )
    
    print("\nOptimal number of clusters:")
    print(f"Best k = {optimization['best_k']} with {optimization['criterion']} score = {optimization['best_score']}")
