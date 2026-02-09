import cv2
import numpy as np
import os
from skimage.morphology import skeletonize, opening, closing, disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def process_skeletonization_bin(input_path, output_dir, config, output_additional_name=None):
    """
    Skeletonization for high-contrast binary masks.
    
    Args:
        input_path (str): Path to the grayscale/binary image.
        output_dir (str): Directory to save results.
        config (dict): Configuration for thresholding and morphology.
        output_additional_name (str): Suffix for files if provided.
        
    Returns:
        str: Path to the skeleton image.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return None

    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: Could not read image: {input_path}")
        return None

    # 1. Thresholding
    # Since these are highly contrasted, simple binary threshold or Otsu's should work.
    # We invert if the background is light (assuming our skeletonize needs white on black).
    # Typical mask_4 often has white lines on black, but let's check config.
    invert = config.get("INVERT", False)
    if invert:
        img_gray = cv2.bitwise_not(img_gray)
    
    thresh_type = config.get("THRESH_TYPE", "OTSU")
    if thresh_type == "OTSU":
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        val = config.get("THRESH_VAL", 127)
        _, binary = cv2.threshold(img_gray, val, 255, cv2.THRESH_BINARY)

    # 2. Morphological Cleanup
    open_k = config.get("MORPH_OPEN_K", 0)
    close_k = config.get("MORPH_CLOSE_K", 0)
    
    mask = binary / 255.0
    if open_k > 0:
        mask = opening(mask, disk(open_k))
    if close_k > 0:
        mask = closing(mask, disk(close_k))

    # 3. Method Selection
    method = config.get("METHOD", "OUTLINE")
    print(f"  Processing {os.path.basename(input_path)} using {method}...")

    if method == "OUTLINE":
        # Extract outlines using contours
        # Convert mask to uint8 for cv2
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Draw outlines on blank image
        thickness = config.get("OUTLINE_THICKNESS", 1)
        outline_img = np.zeros_like(mask_uint8)
        cv2.drawContours(outline_img, contours, -1, 255, thickness)
        
        # Merge nearby lines via dilation
        dilate_size = config.get("OUTLINE_DILATE", 0)
        if dilate_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
            outline_img = cv2.dilate(outline_img, kernel)
            
        # Re-skeletonize if requested
        if config.get("RE_SKELETONIZE", True):
            skeleton = skeletonize(outline_img > 0)
            skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        else:
            skeleton_uint8 = outline_img
            skeleton = skeleton_uint8 > 0
    else:
        # Medial axis skeletonization
        skeleton = skeletonize(mask > 0)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # 4. Save
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    add_name = output_additional_name if output_additional_name else ""
    os.makedirs(output_dir, exist_ok=True)
    
    skel_out_path = os.path.join(output_dir, f"{base_name}{add_name}_skeleton.png")
    cv2.imwrite(skel_out_path, skeleton_uint8)
    
    # Debug Plot
    plot_path = os.path.join(output_dir, f"{base_name}{add_name}_debug_plot.png")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Binary Mask")
    plt.imshow(mask, cmap="gray")
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
