import cv2
import numpy as np
import os
from skimage.morphology import skeletonize, opening, closing, disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .structure_tensor import compute_structure_tensor_coherence

def process_skeletonization_st(input_path, output_dir, config):
    """
    Reads grayscale image, applies Structure Tensor Coherence, thresholds, morphs, and skeletonizes.
    
    Args:
        input_path (str): Path to grayscale input image.
        output_dir (str): Directory to save skeleton and debug plot.
        config (dict): Configuration dictionary with keys:
                       WINDOW_SIGMA, COHERENCE_PERCENTILE, MORPH_OPEN_K, MORPH_CLOSE_K
    
    Returns:
        str: Path to the generated skeleton image (0-255 uint8).
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return None

    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: Could not read image: {input_path}")
        return None

    # Normalization (0-1)
    img_norm = img_gray.astype(np.float32) / 255.0

    # Structure Tensor Coherence
    window_sigma = config.get("WINDOW_SIGMA", 10.0)
    print(f"  Computing Structure Tensor Coherence (window_sigma={window_sigma})...")
    
    # Coherence map (high at boundaries)
    edges, orientation, energy = compute_structure_tensor_coherence(img_norm, sigma=1.0, window_sigma=window_sigma)

    # Thresholding
    percentile = config.get("COHERENCE_PERCENTILE", 50)
    thresh_val = np.percentile(edges, percentile)
    channel_mask = (edges > thresh_val).astype(np.uint8)

    # Morphological cleanup
    open_k = config.get("MORPH_OPEN_K", 3)
    close_k = config.get("MORPH_CLOSE_K", 5)
    if open_k > 0:
        channel_mask = opening(channel_mask, disk(open_k))
    if close_k > 0:
        channel_mask = closing(channel_mask, disk(close_k))

    # Skeletonize
    skeleton = skeletonize(channel_mask > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # Save skeleton
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    skel_out_path = os.path.join(output_dir, f"{base_name}_st_skeleton.png")
    cv2.imwrite(skel_out_path, skeleton_uint8)
    
    # Create Color Overlay (Skeleton in Green)
    # Convert original grayscale to BGR
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # Create green mask where skeleton is 255
    img_bgr[skeleton_uint8 > 0] = [0, 255, 0] # BGR for Green
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, img_bgr)
    print(f"  Overlay saved: {overlay_path}")
    
    # Debug plot (Expanded Diagnostic)
    plot_path = os.path.join(output_dir, f"{base_name}_st_debug_plot.png")
    plt.figure(figsize=(20, 4))
    
    plt.subplot(1, 5, 1)
    plt.title("Energy (Structure Strength)")
    plt.imshow(energy, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 5, 2)
    plt.title("Orientation (Angle)")
    plt.imshow(orientation, cmap="hsv")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.title("Boundary Signal (Inv-Coh)")
    plt.imshow(edges, cmap="magma")
    plt.axis("off")
    
    plt.subplot(1, 5, 4)
    plt.title("Thresholded Mask")
    plt.imshow(channel_mask, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 5, 5)
    plt.title("Skeleton")
    plt.imshow(skeleton, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()

    print(f"  ST Skeleton saved: {skel_out_path}")
    return skel_out_path
