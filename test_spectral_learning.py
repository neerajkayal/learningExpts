#!/usr/bin/env python
"""Test spectral learning on image segmentation."""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segmentImage import getForeground, downsamplePixels
from SpectralLearner import SpectralLearner

def main():
    img_path = "C:\\code\\learningExpts\\infodata\\clusterableImages\\interlockingCurves.png"
    device = "cpu"
    
    # Extract and prepare pixel coordinates
    foreground_coords = getForeground(img_path, method="adaptive_threshold", device=device)
    print(f"Extracted {foreground_coords.shape[0]} foreground pixels.")
    
    n_objects = 1600
    pixel_coords = downsamplePixels(foreground_coords, n_samples=n_objects)
    pixel_coords = pixel_coords.float()
    
    # Normalize to [-1, 1]
    min_coords = pixel_coords.min(dim=0)[0]
    max_coords = pixel_coords.max(dim=0)[0]
    pixel_coords_normalized = 2 * (pixel_coords - min_coords) / (max_coords - min_coords + 1e-8) - 1
    
    print(f"\nInitializing SpectralLearner with {n_objects} pixels...")
    learner = SpectralLearner(
        n_objects=n_objects,
        embedding_dim=2,
        param_dim=1,
        device=device,
        use_distance_similarity=True,
        spectral_regularization=0.01
    )
    
    optimizer = torch.optim.Adam([learner.beta], lr=0.1)
    
    print(f"Initial beta: {learner.beta.item():.4f}\n")
    print("Training to maximize spectral gap...")
    print("Epoch | Loss       | Spectral Gap | Beta")
    print("-" * 45)
    
    n_epochs = 30
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        similarity = learner.compute_similarity(pixel_coords_normalized)
        loss = learner.compute_loss(similarity, None, pixel_coords_normalized)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                laplacian = learner.compute_laplacian(similarity)
                eigenvalues, _ = learner.compute_spectral_gap(laplacian, k=2)
                spectral_gap = float(eigenvalues[1] - eigenvalues[0])
            
            print(f"{epoch+1:5d} | {loss.item():10.6f} | {spectral_gap:12.6f} | {learner.beta.item():.4f}")
    
    # Final results
    print("\n" + "=" * 45)
    print("Training completed!")
    print(f"Final beta: {learner.beta.item():.4f}\n")
    
    with torch.no_grad():
        similarity_final = learner.compute_similarity(pixel_coords_normalized)
        laplacian_final = learner.compute_laplacian(similarity_final)
        eigenvalues_final, eigenvectors_final = learner.compute_spectral_gap(laplacian_final, k=10)
        
        eigenvalues_np = eigenvalues_final.cpu().numpy()
        eigenvectors_np = eigenvectors_final.cpu().numpy()
        
        spectral_gap = eigenvalues_np[1] - eigenvalues_np[0]
        print(f"Final Spectral Gap (λ₂ - λ₁): {spectral_gap:.6f}")
        print(f"First 5 eigenvalues: {eigenvalues_np[:5]}")
        
        # Cluster using Fiedler vector
        fiedler = eigenvectors_np[:, 1]
        clusters = (fiedler > np.median(fiedler)).astype(int)
        
        print(f"\n2-way clustering from Fiedler vector:")
        print(f"  Cluster 0: {(clusters == 0).sum()} pixels")
        print(f"  Cluster 1: {(clusters == 1).sum()} pixels")
    
    # Visualize the clusters on the original image
    print("\nVisualizing clusters...")
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    
    # Create a color image for visualization
    cluster_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Denormalize pixel coordinates back to original image space
    pixel_coords_original = pixel_coords.numpy()
    min_coords_np = min_coords.numpy()
    max_coords_np = max_coords.numpy()
    pixel_coords_original[:, 0] = pixel_coords_original[:, 0] * (max_coords_np[0] - min_coords_np[0]) + min_coords_np[0]
    pixel_coords_original[:, 1] = pixel_coords_original[:, 1] * (max_coords_np[1] - min_coords_np[1]) + min_coords_np[1]
    
    # Color clusters: Cluster 0 = Red, Cluster 1 = Blue
    for i, (row, col) in enumerate(pixel_coords_original):
        row, col = int(row), int(col)
        if 0 <= row < height and 0 <= col < width:
            if clusters[i] == 0:
                cluster_image[row, col] = [0, 0, 255]  # Red in BGR
            else:
                cluster_image[row, col] = [255, 0, 0]  # Blue in BGR
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Foreground mask
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    for row, col in pixel_coords_original:
        row, col = int(row), int(col)
        if 0 <= row < height and 0 <= col < width:
            binary_mask[row, col] = 255
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title('Foreground Pixels')
    axes[1].axis('off')
    
    # Clustered image
    axes[2].imshow(cv2.cvtColor(cluster_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Spectral Clustering\n(Red={sum(clusters==0)}, Blue={sum(clusters==1)})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('spectral_clustering_result.png', dpi=100, bbox_inches='tight')
    print("Saved visualization to 'spectral_clustering_result.png'")
    plt.close()  # Don't show, just save

if __name__ == "__main__":
    main()
