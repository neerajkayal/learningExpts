"""
Spectral Learning for Graph Clustering Parameter Optimization

This module implements SpectralLearner, which optimizes parameters x to maximize
the spectral gap of the graph Laplacian. A large spectral gap indicates well-separated
clusters, as characterized by spectral graph theory and the Cheeger inequality.

Key Concepts:
- Laplacian: L(x) = D(x) - S(x) where D is the degree matrix
- Spectral gap: λ₂(L) - λ₁(L) measures cluster separation
- Fiedler vector: The eigenvector corresponding to λ₂(L) gives a 2-way partition
- Optimizing for spectral gap encourages natural community structure
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse

from ClusterabilityBasedLearner import ClusterabilityBasedLearner


class SpectralLearner(ClusterabilityBasedLearner):
    """
    Optimizes parameters to maximize the spectral gap of the graph Laplacian.
    
    The spectral gap λ₂ - λ₁ is maximized, which encourages:
    - Well-separated clusters
    - Natural community structure (via Cheeger inequality)
    - Robustness of the clustering structure
    
    Uses distance-based similarity: S_ij = exp(-beta * ||p_i - p_j||^2)
    where beta is a learnable parameter controlling the decay rate.
    """
    
    def __init__(
        self,
        n_objects: int,
        embedding_dim: int,
        param_dim: int,
        device: str = "cpu",
        temperature: float = 1.0,
        spectral_regularization: float = 0.1,
        use_distance_similarity: bool = False,
        normalize_laplacian: bool = True
    ):
        """
        Initialize SpectralLearner.
        
        Args:
            n_objects: Number of objects n
            embedding_dim: Dimension of embedding space d
            param_dim: Dimension of parameter vector m (unused if use_distance_similarity=True)
            device: torch device ('cpu' or 'cuda')
            temperature: Temperature parameter for similarity scaling
            spectral_regularization: Weight for spectral regularization term
            use_distance_similarity: If True, use distance-based similarity with learnable beta
            normalize_laplacian: If True, use normalized (random walk) Laplacian: L = I - D^(-1)W
        """
        super().__init__(n_objects, embedding_dim, param_dim, device)
        self.temperature = temperature
        self.spectral_regularization = spectral_regularization
        self.use_distance_similarity = use_distance_similarity
        self.normalize_laplacian = normalize_laplacian
        
        if use_distance_similarity:
            # Learnable beta parameter for distance-based similarity
            # S_ij = exp(-beta * ||p_i - p_j||^2)
            self.beta = nn.Parameter(torch.tensor(1.0, device=device))
        else:
            # Learnable embedding matrix: (n_objects, embedding_dim, param_dim)
            # E_i(x) = embedding_matrix[i] @ x
            self.embedding_matrix = nn.Parameter(
                torch.randn(n_objects, embedding_dim, param_dim, device=device)
            )
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings E(x) = embedding_matrix @ x.
        
        Each object's embedding is a linear function of the parameters x:
        E_i(x) = M_i @ x, where M_i is the i-th slice of embedding_matrix
        
        Args:
            x: Parameter vector of shape (param_dim,)
            
        Returns:
            Embeddings E of shape (n_objects, embedding_dim)
        """
        # E of shape (n_objects, embedding_dim)
        embeddings = torch.einsum('ijk,k->ij', self.embedding_matrix, x)
        return embeddings
    
    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix S from embeddings or distances.
        
        If use_distance_similarity=True:
            Uses distance-based similarity: S_ij = exp(-beta * ||p_i - p_j||^2)
            embeddings should be coordinates (n_objects, d)
        
        Otherwise:
            Uses cosine similarity: S_ij = exp(cosine_similarity(E_i, E_j) / temperature)
        
        Args:
            embeddings: Either coordinates or embeddings of shape (n_objects, embedding_dim)
            
        Returns:
            Similarity matrix S of shape (n_objects, n_objects)
        """
        if self.use_distance_similarity:
            # Compute pairwise distances: ||p_i - p_j||^2
            # embeddings shape: (n_objects, embedding_dim)
            distances_sq = torch.cdist(embeddings, embeddings, p=2) ** 2
            
            # Apply exponential decay with learnable beta
            # S_ij = exp(-beta * ||p_i - p_j||^2)
            similarity = torch.exp(-self.beta * distances_sq)
            
            return similarity
        else:
            # Normalize embeddings
            embeddings_normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Cosine similarity: S = E_norm @ E_norm^T
            cosine_sim = torch.mm(embeddings_normalized, embeddings_normalized.t())
            
            # Apply temperature scaling and exponential to get soft similarities
            similarity = torch.exp(cosine_sim / self.temperature)
            
            return similarity
    
    def compute_laplacian(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the graph Laplacian L from similarity matrix W.
        
        If normalize_laplacian=True (default):
            Uses random walk normalization: L = I - D^(-1)W
            where D is the diagonal degree matrix
            This is better for spectral clustering on unnormalized similarities.
        
        If normalize_laplacian=False:
            Uses unnormalized Laplacian: L = D - W
        
        Args:
            similarity_matrix: W of shape (n_objects, n_objects)
            
        Returns:
            Laplacian L of shape (n_objects, n_objects)
        """
        # Degree matrix: D_ii = sum_j W_ij
        degree = similarity_matrix.sum(dim=1)
        
        if self.normalize_laplacian:
            # Random walk normalization: L = I - D^(-1)W
            # This is equivalent to row-normalizing the similarity matrix
            # Avoid division by zero
            degree_inv = torch.where(
                degree > 1e-8,
                1.0 / degree,
                torch.zeros_like(degree)
            )
            # D^(-1)W
            D_inv_W = degree_inv.unsqueeze(1) * similarity_matrix
            # L = I - D^(-1)W
            laplacian = torch.eye(self.n_objects, device=similarity_matrix.device) - D_inv_W
        else:
            # Unnormalized Laplacian: L = D - W
            laplacian = torch.diag(degree) - similarity_matrix
        
        return laplacian
    
    def compute_spectral_gap(
        self,
        laplacian: torch.Tensor,
        k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the smallest k eigenvalues of the Laplacian (DIFFERENTIABLE).
        
        Uses PyTorch's torch.linalg.eigh for full eigendecomposition, which
        supports automatic differentiation via implicit differentiation.
        
        Args:
            laplacian: Laplacian matrix L of shape (n_objects, n_objects)
            k: Number of smallest eigenvalues to compute
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) for the k smallest eigenvalues
        """
        # Use PyTorch's differentiable eigendecomposition
        # This performs full eigendecomposition but maintains gradient flow
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        
        # Return k smallest eigenvalues and corresponding eigenvectors
        # eigenvalues are already sorted in ascending order
        return eigenvalues[:k], eigenvectors[:, :k]
    
    def compute_spectral_gap_numpy(
        self,
        laplacian: torch.Tensor,
        k: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues using scipy (non-differentiable, for evaluation only).
        
        This method is useful for computing metrics during evaluation when
        gradients are not needed and sparse methods might be faster.
        
        Args:
            laplacian: Laplacian matrix L of shape (n_objects, n_objects)
            k: Number of smallest eigenvalues to compute
            
        Returns:
            Tuple of (eigenvalues_np, eigenvectors_np) as numpy arrays
        """
        L_np = laplacian.detach().cpu().numpy()
        
        try:
            # Try sparse eigenvalue decomposition
            eigenvalues_np, eigenvectors_np = eigsh(
                sparse.csr_matrix(L_np),
                k=min(k, L_np.shape[0] - 1),
                which='SM'  # Smallest magnitude
            )
        except Exception:
            # Fallback to dense eigenvalue decomposition
            eigenvalues_np, eigenvectors_np = np.linalg.eigh(L_np)
            eigenvalues_np = eigenvalues_np[:k]
            eigenvectors_np = eigenvectors_np[:, :k]
        
        return eigenvalues_np, eigenvectors_np
    
    def compute_loss(
        self,
        similarity_matrix: torch.Tensor,
        x: torch.Tensor,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss as negative spectral gap + regularization.
        
        Loss = -(λ₂ - λ₁) + λ_reg * (||S||_F² / ||S||_max)
        
        We maximize the spectral gap (minimize negative spectral gap) and add
        a regularization term to prevent degenerate similarity matrices.
        
        This is fully differentiable with respect to learnable parameters (beta or embeddings).
        
        Args:
            similarity_matrix: S(x) of shape (n_objects, n_objects)
            x: Parameter vector of shape (param_dim,) or None for distance-based
            embeddings: E(x) or coordinates of shape (n_objects, embedding_dim)
            
        Returns:
            Scalar loss value (differentiable)
        """
        # Compute Laplacian
        laplacian = self.compute_laplacian(similarity_matrix)
        
        # Compute spectral properties
        eigenvalues, _ = self.compute_spectral_gap(laplacian, k=2)
        
        # Spectral gap: λ₂ - λ₁
        # We want to maximize this, so we minimize its negative
        if len(eigenvalues) >= 2:
            spectral_gap = eigenvalues[1] - eigenvalues[0]
        else:
            spectral_gap = eigenvalues[0]
        
        # Main loss: negative spectral gap (to be minimized)
        loss_spectral = -spectral_gap
        
        # Regularization: encourage moderate similarity values (not all 0 or all 1)
        # This prevents degenerate solutions where all similarities collapse
        s_norm = torch.norm(similarity_matrix, p='fro')
        s_max = similarity_matrix.max()
        
        if s_max > 1e-8:
            regularization_loss = self.spectral_regularization * (s_norm / s_max)
        else:
            regularization_loss = torch.tensor(0.0, device=similarity_matrix.device)
        
        total_loss = loss_spectral + regularization_loss
        
        return total_loss
    
    def compute_clustering_quality(
        self,
        similarity_matrix: torch.Tensor,
        n_clusters: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate clustering quality using spectral properties.
        
        Args:
            similarity_matrix: S of shape (n_objects, n_objects)
            n_clusters: Optional number of clusters
            
        Returns:
            Dictionary with spectral and graph quality metrics
        """
        # Get parent class metrics
        metrics = super().compute_clustering_quality(similarity_matrix, n_clusters)
        
        # Add spectral metrics
        laplacian = self.compute_laplacian(similarity_matrix)
        eigenvalues, eigenvectors = self.compute_spectral_gap(laplacian, k=min(5, self.n_objects - 1))
        
        eigenvalues_np = eigenvalues.cpu().numpy()
        
        # Spectral gap
        if len(eigenvalues_np) >= 2:
            metrics['spectral_gap'] = float(eigenvalues_np[1] - eigenvalues_np[0])
            metrics['fiedler_eigenvalue'] = float(eigenvalues_np[1])
        else:
            metrics['spectral_gap'] = 0.0
            metrics['fiedler_eigenvalue'] = 0.0
        
        # Trace of Laplacian (sum of eigenvalues)
        metrics['laplacian_trace'] = float(np.sum(eigenvalues_np))
        
        # Normalized spectral gap (for stability)
        if eigenvalues_np[0] < -1e-8:  # Has negative eigenvalue
            norm_gap = eigenvalues_np[1] / abs(eigenvalues_np[0])
        else:
            norm_gap = eigenvalues_np[1]
        metrics['normalized_spectral_gap'] = float(norm_gap)
        
        return metrics
    
    def iterative_refinement(
        self,
        embeddings: torch.Tensor,
        optimizer,
        n_iterations: int = 5,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Iteratively refine clustering by alternating between:
        1. Learning beta with cluster-aware loss
        2. Re-clustering based on updated Fiedler vector
        
        Args:
            embeddings: Coordinates of shape (n_objects, embedding_dim)
            optimizer: Optimizer for beta
            n_iterations: Number of refinement iterations
            verbose: Print progress
            
        Returns:
            Tuple of (final_similarity, final_clusters)
        """
        if verbose:
            print("\nIterative Refinement:")
            print("Iter | Spectral Gap | Intra-cluster sim | Inter-cluster sim | Cluster Sizes")
            print("-" * 80)
        
        # Initial clustering
        similarity = self.compute_similarity(embeddings)
        laplacian = self.compute_laplacian(similarity)
        eigenvalues, eigenvectors = self.compute_spectral_gap(laplacian, k=2)
        
        # Find initial clusters using gap-based partitioning
        fiedler = eigenvectors[:, 1].detach().cpu().numpy()
        sorted_indices = np.argsort(fiedler)
        sorted_fiedler = fiedler[sorted_indices]
        gaps = np.diff(sorted_fiedler)
        max_gap_idx = np.argmax(gaps)
        threshold = (sorted_fiedler[max_gap_idx] + sorted_fiedler[max_gap_idx + 1]) / 2
        clusters = (fiedler > threshold).astype(int)
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Compute similarity with current embeddings and beta
            similarity = self.compute_similarity(embeddings)
            
            # Cluster-aware loss: encourage intra-cluster similarity and inter-cluster dissimilarity
            cluster_loss = self._compute_cluster_aware_loss(similarity, clusters)
            
            # Also include spectral gap maximization
            laplacian = self.compute_laplacian(similarity)
            eigenvalues_iter, eigenvectors_iter = self.compute_spectral_gap(laplacian, k=2)
            spectral_gap = eigenvalues_iter[1] - eigenvalues_iter[0] if len(eigenvalues_iter) >= 2 else eigenvalues_iter[0]
            spectral_loss = -spectral_gap
            
            # Combined loss
            total_loss = 0.7 * spectral_loss + 0.3 * cluster_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Re-cluster based on new Fiedler vector
            with torch.no_grad():
                similarity_new = self.compute_similarity(embeddings)
                laplacian_new = self.compute_laplacian(similarity_new)
                _, eigenvectors_new = self.compute_spectral_gap(laplacian_new, k=2)
                
                fiedler_new = eigenvectors_new[:, 1].detach().cpu().numpy()
                sorted_indices_new = np.argsort(fiedler_new)
                sorted_fiedler_new = fiedler_new[sorted_indices_new]
                gaps_new = np.diff(sorted_fiedler_new)
                max_gap_idx_new = np.argmax(gaps_new)
                threshold_new = (sorted_fiedler_new[max_gap_idx_new] + sorted_fiedler_new[max_gap_idx_new + 1]) / 2
                clusters_new = (fiedler_new > threshold_new).astype(int)
                
                # Compute metrics
                intra_sim = self._compute_intra_cluster_similarity(similarity_new, clusters_new)
                inter_sim = self._compute_inter_cluster_similarity(similarity_new, clusters_new)
            
            clusters = clusters_new
            
            if verbose:
                c0_count = (clusters == 0).sum()
                c1_count = (clusters == 1).sum()
                print(f"{iteration+1:4d} | {float(spectral_gap):12.6f} | {intra_sim:17.6f} | {inter_sim:17.6f} | [{c0_count}, {c1_count}]")
        
        return similarity, clusters
    
    def _compute_cluster_aware_loss(self, similarity: torch.Tensor, clusters: np.ndarray) -> torch.Tensor:
        """
        Compute loss that encourages cluster separation.
        
        Maximize intra-cluster similarity and minimize inter-cluster similarity.
        """
        clusters_t = torch.from_numpy(clusters).to(similarity.device)
        
        # Intra-cluster: encourage high similarity for same-cluster pairs
        same_cluster = (clusters_t.unsqueeze(0) == clusters_t.unsqueeze(1)).float()
        intra_loss = -(similarity * same_cluster).sum() / (same_cluster.sum() + 1e-8)
        
        # Inter-cluster: encourage low similarity for different-cluster pairs
        diff_cluster = 1.0 - same_cluster
        inter_loss = (similarity * diff_cluster).sum() / (diff_cluster.sum() + 1e-8)
        
        return intra_loss + inter_loss
    
    def _compute_intra_cluster_similarity(self, similarity: torch.Tensor, clusters: np.ndarray) -> float:
        """Average similarity within clusters."""
        clusters_t = torch.from_numpy(clusters).to(similarity.device)
        same_cluster = (clusters_t.unsqueeze(0) == clusters_t.unsqueeze(1)).float()
        same_cluster.fill_diagonal_(0)  # Exclude self-similarity
        
        if same_cluster.sum() > 0:
            return float((similarity * same_cluster).sum() / same_cluster.sum())
        return 0.0
    
    def _compute_inter_cluster_similarity(self, similarity: torch.Tensor, clusters: np.ndarray) -> float:
        """Average similarity between clusters."""
        clusters_t = torch.from_numpy(clusters).to(similarity.device)
        diff_cluster = (clusters_t.unsqueeze(0) != clusters_t.unsqueeze(1)).float()
        
        if diff_cluster.sum() > 0:
            return float((similarity * diff_cluster).sum() / diff_cluster.sum())
        return 0.0