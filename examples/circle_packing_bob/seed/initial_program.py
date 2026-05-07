"""Initial circle packing program for N=26 circles."""
import numpy as np


def run():
    """
    Pack 26 circles into a unit square to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    
    # Simple grid-based initial packing
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place circles in a grid pattern
    # Start with a square grid layout
    side = int(np.ceil(np.sqrt(n)))
    
    grid_spacing = 1.0 / (side + 1)
    idx = 0
    
    for i in range(side):
        for j in range(side):
            if idx < n:
                x = (i + 1) * grid_spacing
                y = (j + 1) * grid_spacing
                centers[idx] = [x, y]
                # Small initial radius
                radii[idx] = grid_spacing / 3
                idx += 1
    
    # Compute sum of radii
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii
