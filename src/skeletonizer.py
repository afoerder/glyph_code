import inspect
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import math
from skimage.morphology import skeletonize, opening, closing, disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gaussian_kernel_1d(sigma, kernel_size):
    """Generates a 1D Gaussian kernel."""
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def compute_hessian_eigenvalues_torch(image, sigma):
    """
    Computes Hessian eigenvalues for a given sigma scaling using PyTorch.
    Optimized for GPU using separable convolutions.
    """
    device = image.device
    k_size = int(6 * sigma + 1)
    if k_size % 2 == 0: k_size += 1
    
    # Create 1D Gaussian kernels
    g1d = gaussian_kernel_1d(sigma, k_size).to(device)
    g_row = g1d.view(1, 1, 1, k_size)
    g_col = g1d.view(1, 1, k_size, 1)
    
    # Image must be (1, 1, H, W)
    img_tensor = image.view(1, 1, image.shape[0], image.shape[1])
    
    # Separable Smoothing: Row then Col (O(2K) instead of O(K^2))
    pad = k_size // 2
    smooth_row = F.conv2d(img_tensor, g_row, padding=(0, pad))
    smooth = F.conv2d(smooth_row, g_col, padding=(pad, 0))
    
    # Gradients (central difference)
    # dx kernel: [-0.5, 0, 0.5]
    k_diff = torch.tensor([[-0.5, 0, 0.5]], device=device).view(1, 1, 1, 3)
    k_diff_t = k_diff.view(1, 1, 3, 1)
    
    # First derivatives
    Lx = F.conv2d(smooth, k_diff, padding=(0, 1))
    Ly = F.conv2d(smooth, k_diff_t, padding=(1, 0))
    
    # Second derivatives
    Lxx = F.conv2d(Lx, k_diff, padding=(0, 1))
    Lxy = F.conv2d(Lx, k_diff_t, padding=(1, 0))
    Lyy = F.conv2d(Ly, k_diff_t, padding=(1, 0))
    
    # Clean shapes and scale by sigma^2
    Dxx = Lxx.squeeze() * (sigma**2)
    Dxy = Lxy.squeeze() * (sigma**2)
    Dyy = Lyy.squeeze() * (sigma**2)
    
    # Eigenvalues of Hessian
    tmp = torch.sqrt((Dxx - Dyy)**2 + 4 * Dxy**2)
    l1 = 0.5 * (Dxx + Dyy + tmp)
    l2 = 0.5 * (Dxx + Dyy - tmp)
    
    # Ensure |l1| <= |l2| for Frangi standard
    mask = torch.abs(l1) > torch.abs(l2)
    l1_final = torch.where(mask, l2, l1)
    l2_final = torch.where(mask, l1, l2)
    
    return l1_final, l2_final

def frangi_torch(image_np, scale_range=(1, 10), scale_step=2, beta=0.5, gamma=15):
    """
    PyTorch implementation of Frangi vesselness filter using separable convolutions.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("Warning: CUDA not found, using CPU for Frangi (slow).")
        device = torch.device('cpu')
        
    img_tensor = torch.from_numpy(image_np).float().to(device)
    max_vesselness = torch.zeros_like(img_tensor)
    
    # Iterate scales
    for sigma in range(scale_range[0], scale_range[1], scale_step):
        if sigma == 0: continue
        
        l1, l2 = compute_hessian_eigenvalues_torch(img_tensor, sigma)
        
        l2_fixed = l2.clone()
        l2_fixed[l2_fixed == 0] = 1e-10
        
        rb_sq = (l1 / l2_fixed) ** 2
        s_sq = l1 ** 2 + l2 ** 2
        
        term1 = torch.exp(-rb_sq / (2 * beta**2))
        term2 = 1 - torch.exp(-s_sq / (2 * gamma**2))
        
        vesselness = term1 * term2
        
        # Look for bright lines on dark background: l2 should be negative
        vesselness[l2 > 0] = 0
        
        max_vesselness = torch.maximum(max_vesselness, vesselness)
        
    return max_vesselness.cpu().numpy()

def process_skeletonization(input_path, output_dir, config):
    """
    Reads grayscale image, applies Frangi, thresholds, morphs, and skeletonizes.
    
    Args:
        input_path (str): Path to grayscale input image.
        output_dir (str): Directory to save skeleton and debug plot.
        config (dict): Configuration dictionary with keys:
                       FRANGI_SCALE_RANGE, FRANGI_SCALE_STEP, FRANGI_BETA, FRANGI_GAMMA,
                       VESSEL_PERCENTILE, MORPH_OPEN_K, MORPH_CLOSE_K
    
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

    # Normalize
    # Assuming input is already grayscale and potentially equalized if done in preprocessing?
    # Actually preprocessing just did simple resize/gray. Let's do equalization here to be safe/consistent with fan_finder logic.
    img_eq = cv2.equalizeHist(img_gray)
    img_norm = img_eq.astype(np.float32) / 255.0

    # Frangi
    scale_range = config.get("FRANGI_SCALE_RANGE", (1, 50))
    scale_step = config.get("FRANGI_SCALE_STEP", 2)
    beta = config.get("FRANGI_BETA", 0.1)
    gamma = config.get("FRANGI_GAMMA", 25.0)
    
    print(f"  Running Frangi filter (GPU) on {os.path.basename(input_path)}...")
    vessels = frangi_torch(img_norm, scale_range, scale_step, beta, gamma)

    # Threshold
    percentile = config.get("VESSEL_PERCENTILE", 70)
    thresh_val = np.percentile(vessels, percentile)
    channel_mask = (vessels > thresh_val).astype(np.uint8)

    # Morph
    open_k = config.get("MORPH_OPEN_K", 3)
    close_k = config.get("MORPH_CLOSE_K", 3)
    if open_k > 0:
        channel_mask = opening(channel_mask, disk(open_k))
    if close_k > 0:
        channel_mask = closing(channel_mask, disk(close_k))

    # Skeletonize
    skeleton = skeletonize(channel_mask > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # Save
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    skel_out_path = os.path.join(output_dir, f"{base_name}_skeleton.png")
    cv2.imwrite(skel_out_path, skeleton_uint8)
    
    # Optional debug plot
    plot_path = os.path.join(output_dir, f"{base_name}_debug_plot.png")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Channel Mask")
    plt.imshow(channel_mask, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Skeleton")
    plt.imshow(skeleton, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()

    print(f"  Skeleton saved: {skel_out_path}")
    return skel_out_path
