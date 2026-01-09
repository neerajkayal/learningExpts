#!/usr/bin/env python
"""Visualize foreground pixel extraction."""

import torch
import matplotlib.pyplot as plt
from segmentImage import getForeground

def main():
    img_path = "C:\\code\\learningExpts\\infodata\\clusterableImages\\interlockingCurves.png"
    device = "cpu"
    
    # Extract foreground pixels as (row, col) coordinates
    foreground_coords = getForeground(img_path, method="adaptive_threshold", device=device)
    print(f"Extracted {foreground_coords.shape[0]} foreground pixels")
    print(f"Coordinate ranges:")
    print(f"  Rows: [{foreground_coords[:, 0].min()}, {foreground_coords[:, 0].max()}]")
    print(f"  Cols: [{foreground_coords[:, 1].min()}, {foreground_coords[:, 1].max()}]")
    
    # Convert to numpy for plotting
    coords_np = foreground_coords.cpu().numpy()
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(coords_np[:, 1], coords_np[:, 0], c='red', s=1, alpha=0.5)
    plt.xlabel('Column (x)')
    plt.ylabel('Row (y)')
    plt.title(f'Foreground Pixels Extraction\n({foreground_coords.shape[0]} pixels)')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('foreground_extraction.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved to 'foreground_extraction.png'")
    plt.close()

if __name__ == "__main__":
    main()
