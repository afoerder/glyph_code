import torch
import torch.nn.functional as F
import numpy as np

def compute_structure_tensor_coherence(image_np, sigma=5, window_sigma=10):
    """
    Computes a texture coherence map using Structure Tensor Analysis.
    Optimized with PyTorch for GPU acceleration and separable convolutions.
    
    Args:
        image_np (np.ndarray): 0-1 grayscale image.
        sigma (float): Sigma for derivative smoothing (fine scale).
        window_sigma (float): Sigma for the integration window (texture scale).
        
    Returns:
        np.ndarray: Inverse coherence map (high at boundaries).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.from_numpy(image_np).float().to(device).view(1, 1, *image_np.shape)

    def gaussian_blur_separable(tensor, s):
        if s <= 0: return tensor
        k_size = int(6 * s + 1)
        if k_size % 2 == 0: k_size += 1
        x = torch.arange(-k_size // 2 + 1, k_size // 2 + 1, dtype=torch.float32, device=device)
        kernel1d = torch.exp(-(x**2) / (2 * s**2))
        kernel1d = (kernel1d / kernel1d.sum()).view(1, 1, 1, k_size)
        
        pad = k_size // 2
        tensor = F.conv2d(tensor, kernel1d, padding=(0, pad))
        tensor = F.conv2d(tensor, kernel1d.transpose(2, 3), padding=(pad, 0))
        return tensor

    # 1. Pre-smooth image to remove pixel-level noise
    img_smooth = gaussian_blur_separable(img, sigma)

    # 2. Spatial Gradients (Ix, Iy) using Sobel-style weights
    # [1, 2, 1] smoothing in one dir, [-1, 0, 1] diff in the other
    k_diff = torch.tensor([[-1, 0, 1]], device=device).float().view(1, 1, 1, 3)
    k_smooth = torch.tensor([[1, 2, 1]], device=device).float().view(1, 1, 1, 3) / 4.0
    
    Ix = F.conv2d(F.conv2d(img_smooth, k_diff, padding=(0, 1)), k_smooth.transpose(2, 3), padding=(1, 0))
    Iy = F.conv2d(F.conv2d(img_smooth, k_smooth, padding=(0, 1)), k_diff.transpose(2, 3), padding=(1, 0))

    # 3. Products of derivatives
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 4. Integration Window (Gaussian smoothing of products)
    # This integrates the local neighborhood to form the Structure Tensor
    A = gaussian_blur_separable(Ixx, window_sigma).squeeze()
    C = gaussian_blur_separable(Iyy, window_sigma).squeeze()
    B = gaussian_blur_separable(Ixy, window_sigma).squeeze()

    # 5. Eigenvalues of J = [[A, B], [B, C]]
    tr = A + C
    det = torch.clamp(A * C - B * B, min=0)
    tmp = torch.sqrt(torch.clamp(tr*tr - 4*det, min=0))
    l1 = 0.5 * (tr + tmp)
    l2 = 0.5 * (tr - tmp)

    # 6. Coherence = (l1 - l2) / (l1 + l2)
    coherence = (l1 - l2) / (l1 + l2 + 1e-10)
    
    # Orientation angle
    # Since orientation is periodic (0-180), we use sin(2theta) and cos(2theta)
    # for robust gradient computation.
    denom = A - C + 1e-10
    theta_2 = torch.atan2(2 * B, denom) 
    
    U = torch.cos(theta_2)
    V = torch.sin(theta_2)
    
    # 7. Orientation Gradient (The key boundary signal)
    # Boundaries between texture plates are where the dominant orientation shifts.
    k_diff = torch.tensor([[-0.5, 0, 0.5]], device=device).float().view(1, 1, 1, 3)
    k_diff_t = k_diff.view(1, 1, 3, 1)
    
    # Gradient of the orientation vector (U, V)
    Ux = F.conv2d(U.view(1, 1, *U.shape), k_diff, padding=(0, 1)).squeeze()
    Uy = F.conv2d(U.view(1, 1, *U.shape), k_diff_t, padding=(1, 0)).squeeze()
    Vx = F.conv2d(V.view(1, 1, *V.shape), k_diff, padding=(0, 1)).squeeze()
    Vy = F.conv2d(V.view(1, 1, *V.shape), k_diff_t, padding=(1, 0)).squeeze()
    
    # Total orientation gradient magnitude
    # This is high exactly where the texture "snaps" to a new angle.
    grad_mag = torch.sqrt(Ux**2 + Uy**2 + Vx**2 + Vy**2)
    
    # Integration of the gradient over a small window to broaden the edge for skeletonization
    grad_mag = gaussian_blur_separable(grad_mag.view(1, 1, *grad_mag.shape), s=1.0).squeeze()
    
    # 8. Final signal weighting
    energy_norm = tr / (tr.max() + 1e-10)
    
    # Combine orientation gradient with coherence drops and energy
    # We want HIGH gradient OR LOW coherence, weighted by ENERGY.
    edges = grad_mag * 5.0 + (1.0 - coherence)
    edges = edges * torch.clamp(energy_norm * 3.0, 0, 1)
    
    # Mask out dead areas
    edges[energy_norm < 0.05] = 0
    
    # Robust normalization: normalize based on 99.5th percentile
    q995 = torch.quantile(edges, 0.995)
    if q995 > 0:
        edges = torch.clamp(edges / q995, 0, 1)

    return edges.cpu().numpy(), theta_2.cpu().numpy() * 0.5, energy_norm.cpu().numpy()
