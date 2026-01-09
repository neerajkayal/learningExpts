"""
Unsupervised Graph Clustering Parameter Optimization

Problem Statement:
Let x be a vector of m parameters that we want to optimize for.
Suppose we have a collection of n objects and associated to each of these objects is a vector valued 
embedding function E_i(x) that maps the parameters x to a d-dimensional embedding space.

The embeddings give rise to a similarity matrix S(x) defined as S_ij(x) = similarity(E_i(x), E_j(x)) 
for some similarity function (e.g., cosine similarity, dot product, etc.). S(x) is thus an n x n matrix 
whose entries are functions of the parameters x. 

We interpret S(x) as the adjacency matrix of a weighted undirected graph G(x) with n nodes, where the 
weight of the edge between node i and node j is given by S_ij(x). Our goal is to optimize the parameters x 
such that the graph G(x) has a few large and well-connected components, while the rest of the nodes are 
weakly connected or isolated.

This module provides an abstract framework for exploring different unsupervised learning objectives:
- Clustering-Consistency Loss: Robustness under perturbations
- Graph Reconstruction: Recovering graph from corrupted versions
- Contrastive Reconstruction: Consistency across augmentations

Possible Learners:
1. ModularityLearner - maximizes graph modularity
2. SpectralLearner - optimizes spectral properties
3. ContrastiveLearner - uses augmentation consistency
4. HybridLearner - combines multiple objectives
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Callable
import torch
import torch.nn as nn
import numpy as np


class ClusterabilityBasedLearner(ABC):
    """
    Abstract base class for learning parameters x that optimize for graph clusterability.
    
    This class defines the interface for different unsupervised optimization strategies
    that improve the community structure of graphs defined by similarity matrices.
    
    Subclasses must implement:
    - embed(x): Compute embeddings E(x) from parameters
    - compute_similarity(E): Compute similarity matrix S from embeddings
    - compute_loss(S, x): Compute unsupervised loss given similarity matrix
    - compute_clustering_quality(S): Evaluate quality of clustering
    """
    
    def __init__(
        self,
        n_objects: int,
        embedding_dim: int,
        param_dim: int,
        device: str = "cpu"
    ):
        """
        Initialize the learner.
        
        Args:
            n_objects: Number of objects n
            embedding_dim: Dimension of embedding space d
            param_dim: Dimension of parameter vector m
            device: torch device ('cpu' or 'cuda')
        """
        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.param_dim = param_dim
        self.device = device
        
    @abstractmethod
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings E(x) from parameters.
        
        Args:
            x: Parameter vector of shape (param_dim,)
            
        Returns:
            Embeddings E of shape (n_objects, embedding_dim)
        """
        pass
    
    @abstractmethod
    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix S from embeddings.
        
        Args:
            embeddings: Embeddings E of shape (n_objects, embedding_dim)
            
        Returns:
            Similarity matrix S of shape (n_objects, n_objects)
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        similarity_matrix: torch.Tensor,
        x: torch.Tensor,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute unsupervised loss for optimization.
        
        Args:
            similarity_matrix: S(x) of shape (n_objects, n_objects)
            x: Parameter vector of shape (param_dim,)
            embeddings: E(x) of shape (n_objects, embedding_dim)
            
        Returns:
            Scalar loss value (to be minimized)
        """
        pass
    
    def compute_clustering_quality(
        self,
        similarity_matrix: torch.Tensor,
        n_clusters: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate quality of clustering in the graph.
        
        Args:
            similarity_matrix: S of shape (n_objects, n_objects)
            n_clusters: Optional number of clusters (if None, infer from spectral gap)
            
        Returns:
            Dictionary with quality metrics (modularity, conductance, etc.)
        """
        S = similarity_matrix.detach().cpu().numpy()
        metrics = {}
        
        # Compute modularity (approximation)
        A = S / (S.max() + 1e-8)  # Normalize to [0, 1]
        degree = A.sum(axis=1)
        m = A.sum() / 2
        
        if m > 0:
            modularity = 0
            for i in range(A.shape[0]):
                for j in range(A.shape[0]):
                    modularity += (A[i, j] - degree[i] * degree[j] / (2 * m))
            metrics['modularity'] = modularity / (2 * m)
        else:
            metrics['modularity'] = 0.0
        
        # Compute average edge weight (density of strong connections)
        metrics['avg_edge_weight'] = float(A.mean())
        metrics['max_edge_weight'] = float(A.max())
        
        return metrics
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: parameters -> embeddings -> similarity -> loss.
        
        Args:
            x: Parameter vector of shape (param_dim,)
            
        Returns:
            Tuple of (loss, similarity_matrix, embeddings)
        """
        embeddings = self.embed(x)
        similarity = self.compute_similarity(embeddings)
        loss = self.compute_loss(similarity, x, embeddings)
        
        return loss, similarity, embeddings
    
    def optimize(
        self,
        x_init: torch.Tensor,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize parameters x to minimize loss.
        
        Args:
            x_init: Initial parameter vector of shape (param_dim,)
            learning_rate: Learning rate for optimizer
            num_iterations: Number of optimization steps
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results:
            - 'x_final': Optimized parameters
            - 'losses': Loss history
            - 'qualities': Clustering quality metrics over time
            - 'similarities': Similarity matrices (final)
        """
        x = x_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=learning_rate)
        
        losses = []
        qualities = []
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            loss, similarity, embeddings = self.forward(x)
            
            loss.backward()
            optimizer.step()
            
            # Record metrics
            losses.append(loss.item())
            quality = self.compute_clustering_quality(similarity)
            qualities.append(quality)
            
            if verbose and (iteration + 1) % max(1, num_iterations // 10) == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}: "
                      f"Loss = {loss.item():.4f}, "
                      f"Modularity = {quality.get('modularity', 0):.4f}")
        
        return {
            'x_final': x.detach(),
            'losses': losses,
            'qualities': qualities,
            'similarities': similarity.detach()
        }