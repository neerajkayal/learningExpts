#!/usr/bin/env python
"""Analyze why spectral clustering differs from k-NN ground truth."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from segmentImage import getForeground, downsamplePixels, get_knn_connected_components
from SpectralLearner import SpectralLearner

def main():
    img_path = "C:\\code\\learningExpts\\infodata\\clusterableImages\\interlockingCurves.png"
    device = "cpu"
    
    # Get pixels
    foreground_coords = getForeground(img_path, method="adaptive_threshold", device=device)
    n_objects = min(foreground_coords.shape[0], 1600)
    pixel_coords = downsamplePixels(foreground_coords, n_samples=n_objects)
    pixel_coords = pixel_coords.float()
    
    # Normalize
    min_coords = pixel_coords.min(dim=0)[0]
    max_coords = pixel_coords.max(dim=0)[0]
    pixel_coords_normalized = 2 * (pixel_coords - min_coords) / (max_coords - min_coords + 1e-8) - 1
    
    print(f"Analyzing {n_objects} pixels...")
    
    # 1. Get ground truth
    knn_labels, n_knn = get_knn_connected_components(pixel_coords, k=6)
    print(f"\nGround Truth (k-NN): {n_knn} components")
    print(f"  Sizes: {np.bincount(knn_labels)}")
    
    # 2. Build k-NN adjacency matrix for reference
    pixels_np = pixel_coords.numpy()
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(pixels_np)
    distances, indices = nbrs.kneighbors(pixels_np)
    
    # 3. Create spectral learner with learned beta
    learner = SpectralLearner(
        n_objects=n_objects,
        embedding_dim=2,
        param_dim=1,
        device=device,
        use_distance_similarity=True,
        normalize_laplacian=True,
        spectral_regularization=0.01
    )
    
    # Train to get optimal beta
    optimizer = torch.optim.Adam([learner.beta], lr=0.1)
    for epoch in range(100):
        optimizer.zero_grad()
        similarity = learner.compute_similarity(pixel_coords_normalized)
        loss = learner.compute_loss(similarity, None, pixel_coords_normalized)
        loss.backward()
        optimizer.step()
    
    print(f"\nOptimal beta from training: {learner.beta.item():.4f}")
    
    # 4. Analyze the similarity matrix vs k-NN
    with torch.no_grad():
        similarity = learner.compute_similarity(pixel_coords_normalized)
        similarity_np = similarity.cpu().numpy()
    
    # For each pixel, find how many of its k=6 neighbors have highest similarity
    k_check = 6
    n_matches = 0
    
    for i in range(n_objects):
        # Get k-NN neighbors (indices 1: to skip self)
        knn_neighbors = set(indices[i, 1:k_check+1])
        
        # Get k highest similarity neighbors
        sim_neighbors = set(np.argsort(similarity_np[i, :])[-k_check:])
        
        # Count overlap
        overlap = len(knn_neighbors & sim_neighbors)
        n_matches += overlap
    
    avg_overlap = n_matches / (n_objects * k_check)
    print(f"\nSimilarity-kNN Overlap Analysis:")
    print(f"  Average overlap: {avg_overlap:.3f} (out of {k_check} neighbors)")
    print(f"  This means {100*avg_overlap:.1f}% of top-k similarity neighbors match k-NN neighbors")
    
    # 5. Try spectral clustering with k-NN adjacency instead of distance decay
    print(f"\n" + "="*60)
    print("Alternative: Using k-NN adjacency for spectral clustering...")
    
    # Build k-NN adjacency matrix
    row_indices = []
    col_indices = []
    for i in range(n_objects):
        for j in indices[i, 1:]:
            row_indices.append(i)
            col_indices.append(j)
            row_indices.append(j)
            col_indices.append(i)
    
    data = np.ones(len(row_indices))
    knn_adj = csr_matrix((data, (row_indices, col_indices)), shape=(n_objects, n_objects))
    
    # Convert to dense and use for spectral clustering
    knn_adj_dense = torch.from_numpy(knn_adj.toarray()).float()
    
    # Compute Laplacian from k-NN adjacency
    degree = knn_adj_dense.sum(dim=1)
    degree_inv = torch.where(degree > 1e-8, 1.0 / degree, torch.zeros_like(degree))
    D_inv_W = degree_inv.unsqueeze(1) * knn_adj_dense
    laplacian_knn = torch.eye(n_objects) - D_inv_W
    
    # Get Fiedler vector
    eigenvalues_knn, eigenvectors_knn = torch.linalg.eigh(laplacian_knn)
    fiedler_knn = eigenvectors_knn[:, 1].numpy()
    
    # Partition at gap
    sorted_fiedler = np.sort(fiedler_knn)
    gaps = np.diff(sorted_fiedler)
    max_gap_idx = np.argmax(gaps)
    threshold = (sorted_fiedler[max_gap_idx] + sorted_fiedler[max_gap_idx + 1]) / 2
    clusters_knn_spec = (fiedler_knn > threshold).astype(int)
    
    print(f"\nSpectral clustering using k-NN adjacency (k=6):")
    print(f"  Cluster 0: {(clusters_knn_spec == 0).sum()} pixels")
    print(f"  Cluster 1: {(clusters_knn_spec == 1).sum()} pixels")
    print(f"  Spectral gap: {float(eigenvalues_knn[1] - eigenvalues_knn[0]):.6f}")
    print(f"  ⚠️  Problem: k-NN graph is too dense, curves are already connected!")
    
    # 5b. Try with distance threshold instead
    print(f"\n" + "-"*60)
    print("Better approach: Using distance threshold (0.15) instead of k-NN...")
    
    # Compute pairwise distances
    distances_full = np.linalg.norm(pixels_np[:, np.newaxis, :] - pixels_np[np.newaxis, :, :], axis=2)
    
    # Create adjacency with distance threshold
    threshold_dist = 0.15
    dist_adj = (distances_full < threshold_dist).astype(float)
    np.fill_diagonal(dist_adj, 0)  # Remove self-loops
    
    dist_adj_torch = torch.from_numpy(dist_adj).float()
    degree_dist = dist_adj_torch.sum(dim=1)
    degree_inv_dist = torch.where(degree_dist > 1e-8, 1.0 / degree_dist, torch.zeros_like(degree_dist))
    D_inv_W_dist = degree_inv_dist.unsqueeze(1) * dist_adj_torch
    laplacian_dist = torch.eye(n_objects) - D_inv_W_dist
    
    eigenvalues_dist, eigenvectors_dist = torch.linalg.eigh(laplacian_dist)
    fiedler_dist = eigenvectors_dist[:, 1].numpy()
    
    sorted_fiedler_dist = np.sort(fiedler_dist)
    gaps_dist = np.diff(sorted_fiedler_dist)
    max_gap_idx_dist = np.argmax(gaps_dist)
    threshold_dist_spec = (sorted_fiedler_dist[max_gap_idx_dist] + sorted_fiedler_dist[max_gap_idx_dist + 1]) / 2
    clusters_dist = (fiedler_dist > threshold_dist_spec).astype(int)
    
    print(f"\nSpectral clustering using distance threshold (r=0.15):")
    print(f"  Cluster 0: {(clusters_dist == 0).sum()} pixels")
    print(f"  Cluster 1: {(clusters_dist == 1).sum()} pixels")
    print(f"  Spectral gap: {float(eigenvalues_dist[1] - eigenvalues_dist[0]):.6f}")
    
    # Compare both with ground truth
    def compute_accuracy(clusters, knn_labels):
        agreement = 0
        for c in range(2):
            cluster_c = (clusters == c)
            knn_c = (knn_labels == c)
            agreement += max(np.sum(cluster_c == knn_c), np.sum(cluster_c != knn_c))
        return agreement / (2 * n_objects)
    
    acc_knn = compute_accuracy(clusters_knn_spec, knn_labels)
    acc_dist = compute_accuracy(clusters_dist, knn_labels)
    
    print(f"\n  Accuracy (k-NN spec) vs ground truth: {100*acc_knn:.1f}%")
    print(f"  Accuracy (dist threshold) vs ground truth: {100*acc_dist:.1f}%")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # k-NN ground truth
    for label in np.unique(knn_labels):
        mask = knn_labels == label
        color = 'red' if label == 0 else 'blue'
        axes[0, 0].scatter(pixels_np[mask, 1], pixels_np[mask, 0], c=color, s=5, alpha=0.7)
    axes[0, 0].set_title(f'k-NN Ground Truth ({n_knn} components)')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spectral with k-NN adjacency (failed)
    for label in [0, 1]:
        mask = clusters_knn_spec == label
        color = 'red' if label == 0 else 'blue'
        axes[0, 1].scatter(pixels_np[mask, 1], pixels_np[mask, 0], c=color, s=5, alpha=0.7)
    axes[0, 1].set_title(f'Spectral (k-NN, k=6)\nGap={eigenvalues_knn[1]-eigenvalues_knn[0]:.6f}')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectral with distance threshold (better)
    for label in [0, 1]:
        mask = clusters_dist == label
        color = 'red' if label == 0 else 'blue'
        axes[1, 0].scatter(pixels_np[mask, 1], pixels_np[mask, 0], c=color, s=5, alpha=0.7)
    axes[1, 0].set_title(f'Spectral (distance r=0.15)\nGap={eigenvalues_dist[1]-eigenvalues_dist[0]:.6f}')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Similarity comparison
    sample_size = min(50, n_objects)
    sample_idx = np.random.choice(n_objects, sample_size, replace=False)
    similarity_sample = similarity_np[np.ix_(sample_idx, sample_idx)]
    
    im = axes[1, 1].imshow(similarity_sample, cmap='hot')
    axes[1, 1].set_title(f'Distance-Decay Similarity\n(beta={learner.beta.item():.3f})')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('spectral_analysis.png', dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to 'spectral_analysis.png'")
    plt.close()

if __name__ == "__main__":
    main()
