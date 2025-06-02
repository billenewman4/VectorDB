"""
Cluster refinement using cross-encoder models.
Provides methods to improve clustering quality by refining cluster boundaries.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from sklearn.preprocessing import normalize
import networkx as nx
from collections import defaultdict
import logging

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
cross_encoder_dir = os.path.dirname(current_dir)
clustering_dir = os.path.dirname(cross_encoder_dir)
sys.path.append(os.path.dirname(clustering_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusterRefiner:
    """
    Refines clustering results using cross-encoder models for improved accuracy.
    
    This class provides methods to:
    1. Identify and reassign borderline points between clusters
    2. Validate cluster coherence using pairwise similarity
    3. Split or merge clusters based on cross-encoder similarity scores
    """
    
    def __init__(self, 
                 reranker: Any,
                 embedding_weight: float = 0.7,
                 cross_encoder_weight: float = 0.3,
                 batch_size: int = 32,
                 confidence_threshold: float = 0.6,
                 max_comparison_pairs: int = 10000):
        """
        Initialize the cluster refiner.
        
        Args:
            reranker: Cross-encoder reranker instance (from Cross_Encoder module)
            embedding_weight: Weight for embedding similarity (0.0 to 1.0)
            cross_encoder_weight: Weight for cross-encoder similarity (0.0 to 1.0)
            batch_size: Batch size for cross-encoder processing
            confidence_threshold: Minimum confidence score for reassignment
            max_comparison_pairs: Maximum number of pairs to compare (for performance)
            
        Raises:
            ValueError: If the weights don't sum to 1.0 or reranker is invalid
        """
        # Validate weights
        if not np.isclose(embedding_weight + cross_encoder_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {embedding_weight} + {cross_encoder_weight}")
        
        # Validate reranker
        if not hasattr(reranker, 'compute_similarity') or not callable(getattr(reranker, 'compute_similarity')):
            raise ValueError("Reranker must have a compute_similarity method")
        
        self.reranker = reranker
        self.embedding_weight = embedding_weight
        self.cross_encoder_weight = cross_encoder_weight
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.max_comparison_pairs = max_comparison_pairs
    
    def refine_clusters(self, 
                       clusters: List[Dict[str, Any]], 
                       labels: np.ndarray, 
                       vectors: np.ndarray,
                       texts: List[str],
                       refine_method: str = 'borderline',
                       strict_validation: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Refine cluster assignments using cross-encoder similarity.
        
        Args:
            clusters: List of cluster information dictionaries
            labels: Array of cluster labels
            vectors: Array of embedding vectors
            texts: List of text strings corresponding to vectors
            refine_method: Method for refinement ('borderline', 'coherence', 'reassign_all')
            
        Returns:
            Tuple containing:
                - Refined cluster labels
                - Updated cluster information
                - Refinement metrics
                
        Raises:
            ValueError: If inputs are invalid or method is unsupported
        """
        # Validate inputs
        if len(labels) != len(vectors) or len(vectors) != len(texts):
            raise ValueError(f"Length mismatch: labels={len(labels)}, vectors={len(vectors)}, texts={len(texts)}")
        
        # Initialize metrics
        metrics = {
            'points_evaluated': 0,
            'points_reassigned': 0,
            'reassignment_percentage': 0.0,
            'refinement_method': refine_method,
            'confidence_threshold': self.confidence_threshold
        }
        
        # Choose refinement method
        if refine_method == 'borderline':
            refined_labels, metrics = self._refine_borderline_points(clusters, labels, vectors, texts)
        elif refine_method == 'coherence':
            refined_labels, metrics = self._refine_by_coherence(clusters, labels, vectors, texts)
        elif refine_method == 'reassign_all':
            refined_labels, metrics = self._reassign_all_points(clusters, labels, vectors, texts)
        else:
            raise ValueError(f"Unsupported refinement method: {refine_method}")
        
        # Update cluster information
        updated_clusters = self._update_cluster_info(clusters, refined_labels, vectors, texts)
        
        return refined_labels, updated_clusters, metrics
    
    def _refine_borderline_points(self,
                                 clusters: List[Dict[str, Any]],
                                 labels: np.ndarray,
                                 vectors: np.ndarray,
                                 texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Refine cluster assignments for borderline points only.
        
        Borderline points are those close to the boundary between clusters in the
        embedding space. This method focuses only on these points to save computation.
        
        Args:
            clusters: List of cluster information dictionaries
            labels: Array of cluster labels
            vectors: Array of embedding vectors
            texts: List of text strings corresponding to vectors
            
        Returns:
            Tuple containing:
                - Refined cluster labels
                - Refinement metrics
        """
        # Copy labels to avoid modifying the original
        refined_labels = labels.copy()
        
        # Get centroids for each cluster
        centroids = {}
        for cluster in clusters:
            if cluster['id'] != -1 and 'centroid' in cluster:
                centroid = np.array(cluster['centroid']) if cluster['centroid'] is not None else None
                # Validate the centroid and handle invalid centroids gracefully
                if centroid is None or np.isnan(centroid).any() or np.isinf(centroid).any() or centroid.size == 0 or np.allclose(centroid, 0):
                    # Log the problematic cluster to Questions.md for analysis
                    questions_path = "/Users/billnewman/Desktop/GitHub/VectorDB/Clean_Code/Questions.md"
                    with open(questions_path, "a") as f:
                        f.write("\n---\n")
                        f.write(f"## Invalid Centroid Detected During Refinement\n")
                        f.write(f"**Cluster ID:** {cluster['id']}\n")
                        f.write(f"**Cluster Size:** {cluster.get('size', 'N/A')}\n")
                        f.write(f"**Centroid:** {centroid.tolist() if centroid is not None else None}\n")
                        f.write(f"**Members:** {cluster.get('members', 'N/A')}\n")
                        f.write(f"**Vectors Shape:** {vectors.shape if 'vectors' in locals() else 'unknown'}\n")
                        # Try to print the vectors for this cluster if possible
                        try:
                            member_indices = cluster.get('members', [])
                            if isinstance(member_indices, list) and len(member_indices) > 0:
                                cluster_vectors = vectors[member_indices]
                                f.write(f"**Vectors (first 3 shown):**\n{cluster_vectors[:3]}\n")
                            else:
                                f.write(f"**Vectors:** N/A\n")
                        except Exception as e:
                            f.write(f"**Vectors:** Could not retrieve due to error: {e}\n")
                        f.write("---\n")
                    raise ValueError(f"Invalid centroid for cluster {cluster['id']} detected in refinement. This should never happen. Check upstream clustering logic.")
                centroids[cluster['id']] = centroid

        
        # Find borderline points (points close to more than one centroid)
        borderline_indices = []
        borderline_candidates = []
        
        # Normalize vectors for cosine distance
        normalized_vectors = normalize(vectors)
        
        # Iterate through points, excluding noise points
        for i, (label, vector) in enumerate(zip(labels, normalized_vectors)):
            if label == -1:  # Skip noise points
                continue
            
            # Calculate distances to all centroids
            distances = {}
            for cluster_id, centroid in centroids.items():
                # Normalize centroid for cosine similarity (with safety check for zero norm)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm == 0:
                    error_msg = f"Zero-norm centroid detected for cluster {cluster_id}. Cannot compute similarity."
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                    
                norm_centroid = centroid / centroid_norm
                # Calculate cosine similarity (1 - cosine distance)
                similarity = np.dot(vector, norm_centroid)
                distances[cluster_id] = similarity
            
            # Sort distances to find closest centroids
            sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)
            
            # If point is assigned to its closest centroid, check if it's borderline
            if sorted_distances[0][0] == label:
                # If there's more than one centroid and the second closest is within 20% of the closest
                if len(sorted_distances) > 1:
                    closest_sim = sorted_distances[0][1]
                    second_closest_sim = sorted_distances[1][1]
                    
                    # Avoid division by zero
                    if closest_sim > 0:
                        # Check if borderline (second closest within 20% of closest)
                        if (closest_sim - second_closest_sim) / closest_sim < 0.2:
                            borderline_indices.append(i)
                            # Store the point and its candidate clusters (safely)
                            candidates = [c_id for c_id, sim in sorted_distances[:min(2, len(sorted_distances))]]
                            borderline_candidates.append((i, candidates))
            elif sorted_distances:  # Make sure we have at least one distance
                # If point is not assigned to its closest centroid, it's definitely borderline
                borderline_indices.append(i)
                # Store the point and its candidate clusters (safely)
                # First is the closest centroid by similarity, second is current label
                if len(sorted_distances) >= 1:
                    candidates = [sorted_distances[0][0], label]
                    borderline_candidates.append((i, candidates))
        
        # Limit the number of pairs to compare
        if len(borderline_candidates) > self.max_comparison_pairs:
            logger.info(f"Limiting borderline comparisons from {len(borderline_candidates)} to {self.max_comparison_pairs}")
            # Randomly select subset of borderline points
            np.random.shuffle(borderline_candidates)
            borderline_candidates = borderline_candidates[:self.max_comparison_pairs]
        
        # Process borderline points with cross-encoder
        points_reassigned = 0
        
        for point_idx, candidate_clusters in borderline_candidates:
            # Get the point's text
            query_text = texts[point_idx]
            
            # Collect representative texts from each candidate cluster
            candidate_texts = []
            cluster_ids = []
            
            for cluster_id in candidate_clusters:
                # Get indices of points in this cluster
                cluster_members = np.where(labels == cluster_id)[0]
                
                if len(cluster_members) > 0:
                    # Get center point of cluster
                    if 'core_samples' in clusters[cluster_id] and clusters[cluster_id]['core_samples']:
                        # Use a core sample if available
                        center_idx = clusters[cluster_id]['core_samples'][0]
                    else:
                        # Otherwise use the point closest to centroid
                        cluster_vectors = normalize(vectors[cluster_members])
                        centroid = normalize(np.array([centroids[cluster_id]]))
                        similarities = np.dot(cluster_vectors, centroid.T).flatten()
                        closest_idx = cluster_members[np.argmax(similarities)]
                        center_idx = closest_idx
                    
                    candidate_texts.append(texts[center_idx])
                    cluster_ids.append(cluster_id)
            
            # Skip if we don't have at least two candidates
            if len(candidate_texts) < 2:
                continue
            
            # Compute cross-encoder similarity between point and cluster representatives
            similarities = self.reranker.compute_similarity(
                queries=[query_text] * len(candidate_texts),
                passages=candidate_texts
            )
            
            # Get the best matching cluster
            best_cluster_idx = np.argmax(similarities)
            best_cluster_id = cluster_ids[best_cluster_idx]
            best_similarity = similarities[best_cluster_idx]
            
            # Reassign only if confidence is high enough and different from current
            if best_similarity >= self.confidence_threshold and best_cluster_id != labels[point_idx]:
                refined_labels[point_idx] = best_cluster_id
                points_reassigned += 1
        
        # Compile metrics
        metrics = {
            'points_evaluated': len(borderline_candidates),
            'points_reassigned': points_reassigned,
            'reassignment_percentage': (points_reassigned / len(labels)) * 100 if len(labels) > 0 else 0,
            'refinement_method': 'borderline',
            'confidence_threshold': self.confidence_threshold,
            'borderline_percentage': (len(borderline_candidates) / len(labels)) * 100 if len(labels) > 0 else 0
        }
        
        return refined_labels, metrics
    
    def _refine_by_coherence(self,
                           clusters: List[Dict[str, Any]],
                           labels: np.ndarray,
                           vectors: np.ndarray,
                           texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Refine clusters by measuring internal coherence with cross-encoder.
        
        This method evaluates whether clusters are coherent based on pairwise
        similarity of members, and may split or merge clusters.
        
        Args:
            clusters: List of cluster information dictionaries
            labels: Array of cluster labels
            vectors: Array of embedding vectors
            texts: List of text strings corresponding to vectors
            
        Returns:
            Tuple containing:
                - Refined cluster labels
                - Refinement metrics
        """
        # This method is more complex as it involves potential cluster merging/splitting
        # For now, we'll implement a simplified version that focuses on coherence
        
        # Copy labels to avoid modifying the original
        refined_labels = labels.copy()
        
        # Track metrics
        metrics = {
            'points_evaluated': 0,
            'points_reassigned': 0,
            'refinement_method': 'coherence',
            'confidence_threshold': self.confidence_threshold,
        }
        
        # For each cluster, measure internal coherence
        cluster_coherence = {}
        cluster_connections = defaultdict(list)
        
        # Skip processing if there are too many points (for performance)
        total_points = sum(cluster['size'] for cluster in clusters if cluster['id'] != -1)
        if total_points > self.max_comparison_pairs:
            logger.warning(f"Too many points ({total_points}) for coherence refinement. Skipping.")
            metrics['skipped_due_to_size'] = True
            return refined_labels, metrics
        
        # Process each cluster
        for cluster in clusters:
            cluster_id = cluster['id']
            if cluster_id == -1:  # Skip noise cluster
                continue
                
            members = cluster['members']
            
            # Skip tiny clusters
            if len(members) < 3:
                continue
                
            # Sample members for evaluation (limit to reasonable number)
            sample_size = min(10, len(members))
            sampled_indices = np.random.choice(members, size=sample_size, replace=False)
            
            # Get texts for sampled members
            sampled_texts = [texts[i] for i in sampled_indices]
            
            # Compute pairwise similarity using cross-encoder
            pairs = []
            for i in range(len(sampled_texts)):
                for j in range(i+1, len(sampled_texts)):
                    pairs.append((sampled_texts[i], sampled_texts[j]))
            
            # Process in batches
            all_similarities = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i+self.batch_size]
                queries = [pair[0] for pair in batch_pairs]
                passages = [pair[1] for pair in batch_pairs]
                
                batch_similarities = self.reranker.compute_similarity(queries, passages)
                all_similarities.extend(batch_similarities)
            
            # Calculate average similarity as coherence measure
            if all_similarities:
                coherence = np.mean(all_similarities)
                cluster_coherence[cluster_id] = coherence
            
            metrics['points_evaluated'] += sample_size
        
        # Log coherence measures
        logger.info(f"Cluster coherence: {cluster_coherence}")
        
        # Evaluate inter-cluster relationships (potential merges)
        cluster_ids = [c['id'] for c in clusters if c['id'] != -1]
        
        for i, cluster_id1 in enumerate(cluster_ids):
            for cluster_id2 in cluster_ids[i+1:]:
                # Get representative members from each cluster
                members1 = [m for m in clusters[cluster_id1]['members'][:5]]  # Limit to 5
                members2 = [m for m in clusters[cluster_id2]['members'][:5]]  # Limit to 5
                
                # Skip if either cluster is too small
                if not members1 or not members2:
                    continue
                
                # Sample pairs for comparison
                pairs = []
                for idx1 in members1:
                    for idx2 in members2:
                        pairs.append((texts[idx1], texts[idx2]))
                
                # Limit pairs to reasonable number
                if len(pairs) > 25:  # Max 25 comparisons between clusters
                    np.random.shuffle(pairs)
                    pairs = pairs[:25]
                
                # Process in batches
                all_similarities = []
                for i in range(0, len(pairs), self.batch_size):
                    batch_pairs = pairs[i:i+self.batch_size]
                    queries = [pair[0] for pair in batch_pairs]
                    passages = [pair[1] for pair in batch_pairs]
                    
                    batch_similarities = self.reranker.compute_similarity(queries, passages)
                    all_similarities.extend(batch_similarities)
                
                # Calculate average similarity between clusters
                if all_similarities:
                    inter_coherence = np.mean(all_similarities)
                    
                    # If inter-cluster coherence is high, they might be mergeable
                    if inter_coherence > self.confidence_threshold:
                        # Store connection for potential merging
                        cluster_connections[cluster_id1].append((cluster_id2, inter_coherence))
                        cluster_connections[cluster_id2].append((cluster_id1, inter_coherence))
        
        # Analyze connections to determine merges
        # Build a graph of strongly connected clusters
        G = nx.Graph()
        for cluster_id in cluster_ids:
            G.add_node(cluster_id)
            
        for cluster_id, connections in cluster_connections.items():
            for connected_id, strength in connections:
                if strength > self.confidence_threshold:
                    G.add_edge(cluster_id, connected_id, weight=strength)
        
        # Find connected components (clusters to merge)
        connected_components = list(nx.connected_components(G))
        
        # If we found potential merges, apply them
        merged_cluster_map = {}
        if len(connected_components) < len(cluster_ids):
            for i, component in enumerate(connected_components):
                # Create a new cluster ID (using max existing + i + 1)
                new_cluster_id = max(cluster_ids) + i + 1
                
                # Map all clusters in this component to the new ID
                for old_cluster_id in component:
                    merged_cluster_map[old_cluster_id] = new_cluster_id
            
            # Apply the merges to the labels
            for i, label in enumerate(refined_labels):
                if label in merged_cluster_map:
                    refined_labels[i] = merged_cluster_map[label]
        
        # Update metrics
        reassigned_count = sum(1 for old, new in zip(labels, refined_labels) if old != new)
        metrics['points_reassigned'] = reassigned_count
        metrics['reassignment_percentage'] = (reassigned_count / len(labels)) * 100 if len(labels) > 0 else 0
        metrics['clusters_before'] = len(cluster_ids)
        metrics['clusters_after'] = len(connected_components) if connected_components else len(cluster_ids)
        metrics['merged_clusters'] = [list(c) for c in connected_components if len(c) > 1]
        
        return refined_labels, metrics
    
    def _reassign_all_points(self,
                          clusters: List[Dict[str, Any]],
                          labels: np.ndarray,
                          vectors: np.ndarray,
                          texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reassign all points using cross-encoder similarity to cluster representatives.
        
        This is the most thorough but computationally expensive method as it
        evaluates every point against representatives of all clusters.
        
        Args:
            clusters: List of cluster information dictionaries
            labels: Array of cluster labels
            vectors: Array of embedding vectors
            texts: List of text strings corresponding to vectors
            
        Returns:
            Tuple containing:
                - Refined cluster labels
                - Refinement metrics
        """
        # Copy labels to avoid modifying the original
        refined_labels = labels.copy()
        
        # Skip processing if there are too many points (for performance)
        if len(labels) > self.max_comparison_pairs:
            logger.warning(f"Too many points ({len(labels)}) for complete reassignment. Using random sampling.")
            # Randomly sample points to reassign
            indices_to_reassign = np.random.choice(
                np.arange(len(labels)), 
                size=self.max_comparison_pairs, 
                replace=False
            )
        else:
            indices_to_reassign = np.arange(len(labels))
        
        # Get cluster representatives
        cluster_representatives = {}
        for cluster in clusters:
            cluster_id = cluster['id']
            if cluster_id == -1:  # Skip noise cluster
                continue
                
            members = cluster['members']
            if not members:
                continue
                
            # Use core samples as representatives if available
            if 'core_samples' in cluster and cluster['core_samples']:
                rep_indices = cluster['core_samples'][:3]  # Use up to 3 core samples
            else:
                # Otherwise use random members
                rep_indices = np.random.choice(members, size=min(3, len(members)), replace=False)
                
            cluster_representatives[cluster_id] = [texts[i] for i in rep_indices]
        
        # Process points in batches for reassignment
        points_reassigned = 0
        
        for i in range(0, len(indices_to_reassign), self.batch_size):
            batch_indices = indices_to_reassign[i:i+self.batch_size]
            
            # Create pairs for all points against all cluster representatives
            all_query_texts = []
            all_passage_texts = []
            all_cluster_ids = []
            point_to_comparisons = defaultdict(list)
            
            for point_idx in batch_indices:
                point_text = texts[point_idx]
                
                for cluster_id, representatives in cluster_representatives.items():
                    for rep_text in representatives:
                        all_query_texts.append(point_text)
                        all_passage_texts.append(rep_text)
                        all_cluster_ids.append(cluster_id)
                        point_to_comparisons[point_idx].append(len(all_query_texts) - 1)
            
            # Compute all similarities
            all_similarities = self.reranker.compute_similarity(all_query_texts, all_passage_texts)
            
            # Process results for each point
            for point_idx in batch_indices:
                comparison_indices = point_to_comparisons[point_idx]
                
                # Skip if no comparisons
                if not comparison_indices:
                    continue
                
                # Extract similarities and cluster IDs for this point
                point_similarities = [all_similarities[i] for i in comparison_indices]
                point_cluster_ids = [all_cluster_ids[i] for i in comparison_indices]
                
                # Average similarities by cluster
                cluster_avg_similarities = defaultdict(list)
                for sim, cid in zip(point_similarities, point_cluster_ids):
                    cluster_avg_similarities[cid].append(sim)
                
                avg_similarities = {
                    cid: np.mean(sims) for cid, sims in cluster_avg_similarities.items()
                }
                
                # Find best cluster
                if avg_similarities:
                    best_cluster_id = max(avg_similarities.items(), key=lambda x: x[1])[0]
                    best_similarity = avg_similarities[best_cluster_id]
                    
                    # Reassign if confidence is high enough and different from current
                    if best_similarity >= self.confidence_threshold and best_cluster_id != labels[point_idx]:
                        refined_labels[point_idx] = best_cluster_id
                        points_reassigned += 1
        
        # Compile metrics
        metrics = {
            'points_evaluated': len(indices_to_reassign),
            'points_reassigned': points_reassigned,
            'reassignment_percentage': (points_reassigned / len(labels)) * 100 if len(labels) > 0 else 0,
            'refinement_method': 'reassign_all',
            'confidence_threshold': self.confidence_threshold
        }
        
        return refined_labels, metrics
    
    def _update_cluster_info(self,
                           original_clusters: List[Dict[str, Any]],
                           refined_labels: np.ndarray,
                           vectors: np.ndarray,
                           texts: List[str]) -> List[Dict[str, Any]]:
        """
        Update cluster information based on refined labels.
        
        Args:
            original_clusters: Original cluster information
            refined_labels: Refined cluster labels
            vectors: Embedding vectors
            texts: Text data
            
        Returns:
            Updated cluster information
        """
        # Get unique cluster IDs in refined labels
        unique_labels = np.unique(refined_labels)
        
        # Create mapping from original to updated clusters for reuse of metadata
        original_id_to_cluster = {cluster['id']: cluster for cluster in original_clusters}
        
        # Create updated clusters
        updated_clusters = []
        
        for label in unique_labels:
            cluster_indices = np.where(refined_labels == label)[0]
            cluster_vectors = vectors[cluster_indices]
            
            # Create new cluster or update existing
            if label in original_id_to_cluster:
                # Update existing cluster
                cluster_info = original_id_to_cluster[label].copy()
                cluster_info['members'] = cluster_indices.tolist()
                cluster_info['size'] = len(cluster_indices)
                
                # Recalculate centroid
                if len(cluster_indices) > 0:
                    cluster_info['centroid'] = np.mean(cluster_vectors, axis=0).tolist()
                
                # If noise cluster, keep minimal info
                if label == -1:
                    cluster_info = {
                        'id': -1,
                        'size': len(cluster_indices),
                        'centroid': None,
                        'members': cluster_indices.tolist(),
                        'core_samples': []
                    }
            else:
                # Create new cluster
                centroid = np.mean(cluster_vectors, axis=0) if len(cluster_vectors) > 0 else None
                cluster_info = {
                    'id': int(label),
                    'size': len(cluster_indices),
                    'centroid': centroid.tolist() if centroid is not None else None,
                    'members': cluster_indices.tolist(),
                    'core_samples': []  # No core samples for new clusters
                }
            
            # Add data if available
            if len(cluster_indices) > 0:
                cluster_info['data'] = [texts[i] for i in cluster_indices]
            
            updated_clusters.append(cluster_info)
        
        return updated_clusters


# Simple test
if __name__ == "__main__":
    # Import necessary modules
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from sentence_transformers import CrossEncoder
    
    # Mock reranker for testing
    class MockReranker:
        def compute_similarity(self, queries, passages):
            # Just return random similarities for testing
            return np.random.rand(len(queries))
    
    # Create test data
    X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.6, random_state=42)
    
    # Create mock texts
    texts = [f"Text {i}" for i in range(len(X))]
    
    # Create mock cluster info
    clusters = []
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        vectors = X[indices]
        centroid = np.mean(vectors, axis=0)
        
        clusters.append({
            'id': int(label),
            'size': len(indices),
            'centroid': centroid.tolist(),
            'members': indices.tolist(),
            'core_samples': indices[:3].tolist()  # First 3 as core samples
        })
    
    # Create refiner
    refiner = ClusterRefiner(reranker=MockReranker(), confidence_threshold=0.6)
    
    # Refine clusters
    refined_labels, updated_clusters, metrics = refiner.refine_clusters(
        clusters=clusters,
        labels=y,
        vectors=X,
        texts=texts,
        refine_method='borderline'
    )
    
    # Print results
    print(f"Refinement metrics: {metrics}")
    print(f"Original clusters: {len(clusters)}")
    print(f"Updated clusters: {len(updated_clusters)}")
    
    # Visualize before and after (for 2D data)
    if X.shape[1] == 2:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("Before Refinement")
        
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=refined_labels)
        plt.title("After Refinement")
        
        plt.tight_layout()
        plt.show()
