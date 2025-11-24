import numpy as np
import cv2

def get_region_bounds(mask_channel):
    """
    Finds the top (start) and bottom (end) pixel coordinates of a region.
    mask_channel: 2D numpy array (0 or 1)
    """
    rows, cols = np.where(mask_channel > 0.5)
    
    if len(rows) == 0:
        return "Not Detected", "Not Detected"
    
    # Top-most point (min Y)
    start_point = (cols[np.argmin(rows)], np.min(rows))
    # Bottom-most point (max Y)
    end_point = (cols[np.argmax(rows)], np.max(rows))
    
    return start_point, end_point

def determine_spine_shape(full_mask):
    """
    Mathematically determines C-Shape vs S-Shape by fitting a curve to the spine centroid.
    full_mask: Combined binary mask of the spine
    """
    # 1. Extract skeleton/centroids of the spine
    h, w = full_mask.shape
    centroids_x = []
    centroids_y = []
    
    for y in range(0, h, 10): # Scan every 10 pixels
        row_pixels = np.where(full_mask[y, :] > 0.5)[0]
        if len(row_pixels) > 0:
            center_x = np.mean(row_pixels)
            centroids_x.append(center_x)
            centroids_y.append(y)
            
    if len(centroids_y) < 5:
        return "Unknown (Incomplete Data)"
        
    # 2. Fit a Polynomial Curve (3rd degree)
    # x = ay^3 + by^2 + cy + d
    try:
        poly_params = np.polyfit(centroids_y, centroids_x, 3)
        p = np.poly1d(poly_params)
        
        # 3. Calculate 2nd Derivative (Inflection points)
        # S-Shape has at least 1 inflection point (change in curvature)
        # C-Shape has 0 inflection points within the range
        
        # Simple heuristic: Check variation in X direction
        # Or check sign changes in 2nd derivative
        second_deriv = p.deriv(2)
        roots = second_deriv.roots
        
        real_roots = [r for r in roots if np.isreal(r) and 0 < r < h]
        
        if len(real_roots) > 0:
            return "S-Shape (Scoliosis)"
        else:
            return "C-Shape / Normal"
            
    except:
        return "Analysis Failed"