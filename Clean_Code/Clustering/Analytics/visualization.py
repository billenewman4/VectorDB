"""
Visualization tools for clustering results.
Provides functions to visualize clusters in 2D and 3D using dimensionality reduction.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusterVisualizer:
    """
    Visualization tools for clustering results using dimensionality reduction.
    
    This class provides methods to visualize clusters in 2D and 3D using various
    dimensionality reduction techniques like t-SNE, UMAP, and PCA.
    """
    
    @staticmethod
    def reduce_dimensions(
        vectors: np.ndarray, 
        method: str = 'tsne', 
        n_components: int = 2,
        random_state: int = 42,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce dimensionality of vectors for visualization.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
            n_components: Number of components (2 or 3 for visualization)
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the specific method
            
        Returns:
            Array of reduced vectors with shape (n_samples, n_components)
            
        Raises:
            ValueError: If inputs are invalid or method is unsupported
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a numpy array")
        
        if len(vectors.shape) != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        
        if len(vectors) == 0:
            raise ValueError("vectors cannot be empty")
        
        if n_components not in (2, 3):
            raise ValueError(f"n_components must be 2 or 3 for visualization, got {n_components}")
        
        # Use appropriate method for dimensionality reduction
        if method.lower() == 'tsne':
            # Default parameters for t-SNE
            tsne_params = {
                'perplexity': min(30, len(vectors) - 1),
                'n_iter': 1000,
                'random_state': random_state
            }
            tsne_params.update(kwargs)
            
            # Check and adjust perplexity based on dataset size
            if tsne_params['perplexity'] >= len(vectors):
                tsne_params['perplexity'] = max(1, len(vectors) // 2 - 1)
                logger.warning(f"Adjusted perplexity to {tsne_params['perplexity']} based on dataset size")
            
            reducer = TSNE(n_components=n_components, **tsne_params)
            return reducer.fit_transform(vectors)
            
        elif method.lower() == 'umap':
            # Default parameters for UMAP
            umap_params = {
                'n_neighbors': min(15, len(vectors) - 1),
                'min_dist': 0.1,
                'random_state': random_state
            }
            umap_params.update(kwargs)
            
            # Check and adjust n_neighbors based on dataset size
            if umap_params['n_neighbors'] >= len(vectors):
                umap_params['n_neighbors'] = max(2, len(vectors) // 2)
                logger.warning(f"Adjusted n_neighbors to {umap_params['n_neighbors']} based on dataset size")
            
            reducer = umap.UMAP(n_components=n_components, **umap_params)
            return reducer.fit_transform(vectors)
            
        elif method.lower() == 'pca':
            # PCA parameters
            pca_params = {
                'random_state': random_state
            }
            pca_params.update(kwargs)
            
            reducer = PCA(n_components=n_components, **pca_params)
            return reducer.fit_transform(vectors)
            
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    @staticmethod
    def visualize_clusters_2d(
        vectors: np.ndarray,
        labels: np.ndarray,
        method: str = 'tsne',
        title: str = 'Cluster Visualization',
        figsize: Tuple[int, int] = (12, 8),
        marker_size: int = 50,
        random_state: int = 42,
        alpha: float = 0.7,
        highlight_centroids: bool = False,
        centroids: Optional[np.ndarray] = None,
        data_labels: Optional[List[str]] = None,
        colormap: str = 'tab10',
        output_file: Optional[str] = None,
        show_legend: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Visualize clusters in 2D using matplotlib.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            labels: Array of cluster labels
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
            title: Plot title
            figsize: Figure size as (width, height)
            marker_size: Size of markers
            random_state: Random state for reproducibility
            alpha: Transparency of markers
            highlight_centroids: Whether to highlight cluster centroids
            centroids: Array of cluster centroids (optional)
            data_labels: Optional labels for each data point
            colormap: Matplotlib colormap name
            output_file: Path to save the figure (if None, figure is not saved)
            show_legend: Whether to show the legend
            **kwargs: Additional parameters for dimensionality reduction
            
        Returns:
            Matplotlib figure object
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if len(vectors) != len(labels):
            raise ValueError(f"Length mismatch: vectors has {len(vectors)} items, but labels has {len(labels)} items")
        
        # Reduce dimensions to 2D
        reduced_vectors = ClusterVisualizer.reduce_dimensions(
            vectors=vectors,
            method=method,
            n_components=2,
            random_state=random_state,
            **kwargs
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique labels and assign colors
        unique_labels = np.unique(labels)
        
        # Create scatter plot for each cluster
        for i, label in enumerate(unique_labels):
            # Get indices for this cluster
            mask = labels == label
            
            # Extract 2D coordinates
            x = reduced_vectors[mask, 0]
            y = reduced_vectors[mask, 1]
            
            # Use different style for noise points (label -1)
            if label == -1:
                scatter = ax.scatter(
                    x, y, 
                    c='gray', 
                    marker='x', 
                    s=marker_size*0.7, 
                    alpha=alpha*0.7, 
                    label='Noise'
                )
            else:
                # For small clusters, using c=[i] causes issues with scatter plot dimensions
                # Use a consistent color based on the cluster label instead
                scatter = ax.scatter(
                    x, y, 
                    c=np.full(x.shape, i), # Ensure color array matches point count
                    cmap=colormap, 
                    marker='o', 
                    s=marker_size, 
                    alpha=alpha, 
                    label=f'Cluster {label}'
                )
                
                # Add text labels if provided
                if data_labels is not None:
                    for xi, yi, label_i in zip(x, y, np.array(data_labels)[mask]):
                        ax.annotate(
                            label_i, 
                            (xi, yi),
                            fontsize=8,
                            alpha=0.7,
                            ha='right',
                            va='bottom'
                        )
        
        # Highlight centroids if requested
        if highlight_centroids and centroids is not None:
            # Reduce dimensions of centroids
            reduced_centroids = ClusterVisualizer.reduce_dimensions(
                vectors=centroids,
                method=method,
                n_components=2,
                random_state=random_state,
                **kwargs
            )
            
            # Plot centroids
            ax.scatter(
                reduced_centroids[:, 0], 
                reduced_centroids[:, 1],
                c='black',
                marker='*',
                s=marker_size*2,
                alpha=1.0,
                label='Centroids'
            )
        
        # Set title and labels
        reduction_method_name = {'tsne': 't-SNE', 'umap': 'UMAP', 'pca': 'PCA'}
        ax.set_title(f"{title} ({reduction_method_name.get(method.lower(), method)})")
        ax.set_xlabel(f"Dimension 1")
        ax.set_ylabel(f"Dimension 2")
        
        # Add legend if requested
        if show_legend:
            ax.legend()
        
        # Remove grid and ticks for cleaner visualization
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add statistics text
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        stats_text = f"Clusters: {n_clusters}, Points: {len(vectors)}, Noise: {n_noise} ({n_noise/len(vectors)*100:.1f}%)"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10, alpha=0.7)
        
        # Save figure if output path is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_clusters_3d(
        vectors: np.ndarray,
        labels: np.ndarray,
        method: str = 'tsne',
        title: str = 'Cluster Visualization (3D)',
        marker_size: int = 5,
        random_state: int = 42,
        opacity: float = 0.7,
        highlight_centroids: bool = False,
        centroids: Optional[np.ndarray] = None,
        data_labels: Optional[List[str]] = None,
        colormap: str = 'plasma',
        output_file: Optional[str] = None,
        show_legend: bool = True,
        **kwargs
    ) -> go.Figure:
        """
        Visualize clusters in 3D using Plotly.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            labels: Array of cluster labels
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
            title: Plot title
            marker_size: Size of markers
            random_state: Random state for reproducibility
            opacity: Transparency of markers
            highlight_centroids: Whether to highlight cluster centroids
            centroids: Array of cluster centroids (optional)
            data_labels: Optional labels for each data point
            colormap: Plotly colormap name
            output_file: Path to save the figure as HTML (if None, figure is not saved)
            show_legend: Whether to show the legend
            **kwargs: Additional parameters for dimensionality reduction
            
        Returns:
            Plotly figure object
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if len(vectors) != len(labels):
            raise ValueError(f"Length mismatch: vectors has {len(vectors)} items, but labels has {len(labels)} items")
        
        # Reduce dimensions to 3D
        reduced_vectors = ClusterVisualizer.reduce_dimensions(
            vectors=vectors,
            method=method,
            n_components=3,
            random_state=random_state,
            **kwargs
        )
        
        # Create dataframe for Plotly
        df = pd.DataFrame({
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1],
            'z': reduced_vectors[:, 2],
            'label': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
        })
        
        # Add text labels if provided
        if data_labels:
            df['text'] = data_labels
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='label',
            symbol='label',
            opacity=opacity,
            size=[marker_size] * len(df),  # Uniform size
            hover_data=['text'] if data_labels else None,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Customize symbol for noise points
        if -1 in labels:
            noise_trace_index = list(df['label'].unique()).index('Noise')
            fig.data[noise_trace_index].marker.symbol = 'x'
            fig.data[noise_trace_index].marker.color = 'gray'
        
        # Highlight centroids if requested
        if highlight_centroids and centroids is not None:
            # Reduce dimensions of centroids
            reduced_centroids = ClusterVisualizer.reduce_dimensions(
                vectors=centroids,
                method=method,
                n_components=3,
                random_state=random_state,
                **kwargs
            )
            
            # Add centroids to plot
            centroid_trace = go.Scatter3d(
                x=reduced_centroids[:, 0],
                y=reduced_centroids[:, 1],
                z=reduced_centroids[:, 2],
                mode='markers',
                marker=dict(
                    size=marker_size*2,
                    color='black',
                    symbol='diamond'
                ),
                name='Centroids'
            )
            
            fig.add_trace(centroid_trace)
        
        # Add statistics text
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        stats_text = f"Clusters: {n_clusters}, Points: {len(vectors)}, Noise: {n_noise} ({n_noise/len(vectors)*100:.1f}%)"
        
        fig.add_annotation(
            x=0,
            y=0,
            text=stats_text,
            showarrow=False,
            xanchor='left',
            xref='paper',
            yref='paper'
        )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(orientation="h") if show_legend else dict(visible=False)
        )
        
        # Save figure if output path is provided
        if output_file:
            fig.write_html(output_file)
        
        return fig
    
    @staticmethod
    def cluster_comparison_plot(
        vectors: np.ndarray,
        labels_list: List[np.ndarray],
        labels_names: List[str],
        method: str = 'tsne',
        figsize: Tuple[int, int] = (16, 10),
        marker_size: int = 30,
        random_state: int = 42,
        alpha: float = 0.7,
        title: str = 'Clustering Method Comparison',
        colormap: str = 'tab10',
        output_file: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Compare multiple clustering results in a grid of plots.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            labels_list: List of cluster label arrays from different algorithms
            labels_names: Names of the clustering algorithms for each label array
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
            figsize: Figure size as (width, height)
            marker_size: Size of markers
            random_state: Random state for reproducibility
            alpha: Transparency of markers
            title: Overall plot title
            colormap: Matplotlib colormap name
            output_file: Path to save the figure (if None, figure is not saved)
            **kwargs: Additional parameters for dimensionality reduction
            
        Returns:
            Matplotlib figure object
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if len(labels_list) != len(labels_names):
            raise ValueError("labels_list and labels_names must have the same length")
        
        for i, labels in enumerate(labels_list):
            if len(vectors) != len(labels):
                raise ValueError(f"Length mismatch for {labels_names[i]}: vectors has {len(vectors)} items, but labels has {len(labels)} items")
        
        # Reduce dimensions to 2D
        reduced_vectors = ClusterVisualizer.reduce_dimensions(
            vectors=vectors,
            method=method,
            n_components=2,
            random_state=random_state,
            **kwargs
        )
        
        # Create figure with subplots
        n_plots = len(labels_list)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])  # Convert to array for consistent indexing
        axes = axes.flatten()
        
        # Plot each clustering result
        for i, (labels, name) in enumerate(zip(labels_list, labels_names)):
            ax = axes[i]
            
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Plot each cluster
            for j, label in enumerate(unique_labels):
                mask = labels == label
                
                # Use different style for noise points (label -1)
                if label == -1:
                    ax.scatter(
                        reduced_vectors[mask, 0],
                        reduced_vectors[mask, 1],
                        c='gray',
                        marker='x',
                        s=marker_size*0.7,
                        alpha=alpha*0.7,
                        label='Noise'
                    )
                else:
                    ax.scatter(
                        reduced_vectors[mask, 0],
                        reduced_vectors[mask, 1],
                        c=[j],
                        cmap=colormap,
                        marker='o',
                        s=marker_size,
                        alpha=alpha,
                        label=f'C{label}'
                    )
            
            # Set subplot title and remove ticks
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)
            noise_pct = n_noise / len(labels) * 100 if len(labels) > 0 else 0
            
            ax.set_title(f"{name}\n{n_clusters} clusters, {noise_pct:.1f}% noise")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        # Set overall title
        reduction_method_name = {'tsne': 't-SNE', 'umap': 'UMAP', 'pca': 'PCA'}
        fig.suptitle(f"{title} ({reduction_method_name.get(method.lower(), method)})", fontsize=16)
        
        # Save figure if output path is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
        return fig


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
    
    # Visualize individual clustering results
    fig_kmeans = ClusterVisualizer.visualize_clusters_2d(
        vectors=X,
        labels=y_kmeans,
        title="K-Means Clustering",
        method='pca'  # Using PCA since data is already 2D
    )
    
    # Visualize in 3D using Plotly
    fig_3d = ClusterVisualizer.visualize_clusters_3d(
        vectors=X,
        labels=y_hdbscan,
        title="HDBSCAN Clustering (3D)",
        method='pca'
    )
    
    # Compare different clustering methods
    fig_comparison = ClusterVisualizer.cluster_comparison_plot(
        vectors=X,
        labels_list=[y_true, y_kmeans, y_agglo, y_hdbscan],
        labels_names=["Ground Truth", "K-Means", "Agglomerative", "HDBSCAN"],
        method='pca'
    )
    
    plt.show()
