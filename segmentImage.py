import cv2
import numpy as np
import torch
from typing import Optional, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

def downsamplePixels(pixels: torch.Tensor, n_samples: int = 1600) -> torch.Tensor:
    """
    Randomly downsample a collection of pixel coordinates to a target number.
    
    Args:
        pixels: Tensor of shape (N, 2) containing (row, col) coordinates
        n_samples: Target number of pixels to sample (default: 1600)
        
    Returns:
        Downsampled tensor of shape (n_samples, 2) or original if N < n_samples
    """
    n_pixels = pixels.shape[0]
    
    if n_pixels <= n_samples:
        return pixels
    
    # Randomly select indices
    indices = torch.randperm(n_pixels)[:n_samples]
    return pixels[indices]


def getForeground(image_path: str, method: str = "adaptive_threshold", device: str = "cpu") -> torch.Tensor:
    """
    Extract foreground pixel locations as an (N, 2) torch.LongTensor of (row, col).
    
    Returns the SMALLER subset of (masked vs unmasked) pixels as foreground,
    since the object of interest is typically smaller than the background.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == "adaptive_threshold":
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "otsu":
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "background_subtraction":
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get both foreground (white, 255) and background (black, 0) pixels
    mask_white = binary > 0
    mask_black = binary == 0
    
    white_count = np.count_nonzero(mask_white)
    black_count = np.count_nonzero(mask_black)
    
    # Choose the smaller subset as foreground
    if white_count <= black_count:
        mask_bool = mask_white
    else:
        mask_bool = mask_black
    
    if not mask_bool.any():
        return torch.empty((0, 2), dtype=torch.long, device=device)

    coords = np.column_stack(np.where(mask_bool))  # (row, col)
    coords_t = torch.from_numpy(coords).to(device=device, dtype=torch.long)
    return coords_t


def get_knn_connected_components(
    pixels: torch.Tensor,
    k: int = 6
) -> np.ndarray:
    """
    Find connected components using k-nearest neighbors graph.
    
    Args:
        pixels: Pixel coordinates of shape (n_pixels, 2)
        k: Number of nearest neighbors
        
    Returns:
        Component labels of shape (n_pixels,) where each unique value is a component
    """
    # Convert to numpy
    pixels_np = pixels.cpu().numpy() if isinstance(pixels, torch.Tensor) else pixels
    
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(pixels_np)
    distances, indices = nbrs.kneighbors(pixels_np)
    
    # Build adjacency matrix (sparse)
    n_pixels = pixels_np.shape[0]
    row_indices = []
    col_indices = []
    
    # For each pixel, connect to its k nearest neighbors (including itself)
    for i in range(n_pixels):
        for j in indices[i, 1:]:  # Skip first (self)
            row_indices.append(i)
            col_indices.append(j)
            # Make symmetric
            row_indices.append(j)
            col_indices.append(i)
    
    # Create sparse adjacency matrix
    data = np.ones(len(row_indices))
    adjacency = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_pixels, n_pixels)
    )
    
    # Find connected components
    n_components, labels = connected_components(adjacency, directed=False)
    
    return labels, n_components


def visualize_knn_components(
    pixels: torch.Tensor,
    labels: np.ndarray,
    title: str = "k-NN Connected Components",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize connected components.
    
    Args:
        pixels: Pixel coordinates of shape (n_pixels, 2)
        labels: Component labels
        title: Title for the plot
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    pixels_np = pixels.cpu().numpy() if isinstance(pixels, torch.Tensor) else pixels
    n_components = len(np.unique(labels))
    
    # Create colormap with enough colors
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_components, 20)))
    if n_components > 20:
        # For more components, cycle through colors
        colors_extended = np.vstack([colors] * (n_components // 20 + 1))
        colors = colors_extended[:n_components]
    
    plt.figure(figsize=figsize)
    for component in np.unique(labels):
        mask = labels == component
        plt.scatter(
            pixels_np[mask, 1],
            pixels_np[mask, 0],
            c=[colors[component]],
            s=10,
            alpha=0.7,
            label=f"Component {component}"
        )
    
    plt.xlabel('Column (x)')
    plt.ylabel('Row (y)')
    plt.title(f"{title}\n({n_components} components)")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    from SpectralLearner import SpectralLearner
    
    img_path = "C:\\code\\learningExpts\\infodata\\clusterableImages\\interlockingCurves.png"
    device = "cpu"
    
    # Extract foreground pixels as (row, col) coordinates
    foreground_coords = getForeground(img_path, method="adaptive_threshold", device=device)
    print(f"Extracted {foreground_coords.shape[0]} foreground pixels.")
    
    # Downsample to target number (or use all if fewer than target)
    n_target = 1600
    n_objects = min(foreground_coords.shape[0], n_target)
    pixel_coords = downsamplePixels(foreground_coords, n_samples=n_objects)
    pixel_coords = pixel_coords.float()  # Convert to float for learning
    
    print(f"Using {n_objects} pixels for learning.")
    
    # Normalize coordinates to [-1, 1] range for numerical stability
    min_coords = pixel_coords.min(dim=0)[0]
    max_coords = pixel_coords.max(dim=0)[0]
    pixel_coords_normalized = 2 * (pixel_coords - min_coords) / (max_coords - min_coords + 1e-8) - 1
    
    print(f"\nInitializing SpectralLearner with {n_objects} pixel objects...")
    
    # Initialize SpectralLearner with distance-based similarity and normalized Laplacian
    learner = SpectralLearner(
        n_objects=n_objects,
        embedding_dim=2,  # 2D pixel coordinates
        param_dim=1,      # Not used for distance-based similarity
        device=device,
        use_distance_similarity=True,
        normalize_laplacian=True,  # Use random walk normalization
        spectral_regularization=0.01
    )
    
    # Optimizer: only optimize beta
    optimizer = torch.optim.Adam([learner.beta], lr=0.1)
    
    print(f"Initial beta: {learner.beta.item():.4f}")
    print("\nTraining to maximize spectral gap...")
    print("Epoch | Loss       | Spectral Gap | Beta")
    print("-" * 45)
    
    n_epochs = 100
    loss_history = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute similarity matrix from pixel coordinates
        similarity = learner.compute_similarity(pixel_coords_normalized)
        
        # Compute loss (negative spectral gap + regularization)
        loss = learner.compute_loss(similarity, None, pixel_coords_normalized)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Log progress
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            # Compute metrics without gradients
            with torch.no_grad():
                laplacian = learner.compute_laplacian(similarity)
                eigenvalues_torch, _ = learner.compute_spectral_gap(laplacian, k=2)
                spectral_gap = float(eigenvalues_torch[1] - eigenvalues_torch[0]) if len(eigenvalues_torch) >= 2 else 0.0
            
            print(f"{epoch+1:5d} | {loss.item():10.6f} | {spectral_gap:12.6f} | {learner.beta.item():.4f}")
    
    print("\n" + "=" * 45)
    print("Initial training completed!")
    print(f"Beta after initial training: {learner.beta.item():.4f}")
    
    # Iterative refinement
    print("\n" + "=" * 45)
    similarity_refined, clusters_refined = learner.iterative_refinement(
        pixel_coords_normalized,
        optimizer,
        n_iterations=10,
        verbose=True
    )
    
    print("\n" + "=" * 45)
    print("Refinement completed!")
    
    # Compute final metrics
    with torch.no_grad():
        laplacian_final = learner.compute_laplacian(similarity_refined)
        eigenvalues_final, eigenvectors_final = learner.compute_spectral_gap(laplacian_final, k=10)
        
        eigenvalues_final_np = eigenvalues_final.cpu().numpy()
        
        print(f"\nFinal Spectral Gap (λ₂ - λ₁): {eigenvalues_final_np[1] - eigenvalues_final_np[0]:.6f}")
        print(f"First 5 eigenvalues: {eigenvalues_final_np[:5]}")
        print(f"\nFinal clusters:")
        print(f"  Cluster 0: {(clusters_refined == 0).sum()} pixels")
        print(f"  Cluster 1: {(clusters_refined == 1).sum()} pixels")
        print(f"  Final beta: {learner.beta.item():.4f}")
        
        # Display the two clusters visually
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        colors = ['red' if c == 0 else 'blue' for c in clusters_refined]
        pixel_coords_original = pixel_coords.numpy()
        # Denormalize pixel coordinates back to original image space
        pixel_coords_original[:, 0] = pixel_coords_original[:, 0] * (max_coords.numpy()[0] - min_coords.numpy()[0]) + min_coords.numpy()[0]
        pixel_coords_original[:, 1] = pixel_coords_original[:, 1] * (max_coords.numpy()[1] - min_coords.numpy()[1]) + min_coords.numpy()[1]
        plt.scatter(pixel_coords_original[:, 1], pixel_coords_original[:, 0], c=colors, s=5, alpha=0.7)
        plt.xlabel('Column (x)')
        plt.ylabel('Row (y)')
        plt.title('Spectral Clustering Result (Iteratively Refined)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('spectral_clustering_result.png', dpi=100, bbox_inches='tight')
        print("\nSaved visualization to 'spectral_clustering_result.png'")
        plt.close()
    
    # Ground truth clustering via k-NN connected components
    print("\n" + "=" * 45)
    print("Computing ground truth via k-NN connected components...")
    knn_labels, n_knn_components = get_knn_connected_components(pixel_coords, k=6)
    print(f"Found {n_knn_components} connected components via k-NN")
    
    # Count component sizes
    unique_components, component_counts = np.unique(knn_labels, return_counts=True)
    print(f"Component sizes: min={component_counts.min()}, max={component_counts.max()}, mean={component_counts.mean():.1f}")
    
    # Visualize k-NN components
    print("\nGenerating k-NN component visualization...")
    plt_knn = visualize_knn_components(pixel_coords, knn_labels, title="Ground Truth: k-NN Connected Components (k=6)")
    plt_knn.savefig('knn_components.png', dpi=100, bbox_inches='tight')
    print("Saved k-NN visualization to 'knn_components.png'")
    plt_knn.close()