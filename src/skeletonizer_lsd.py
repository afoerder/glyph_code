import cv2
import numpy as np
import os
from skimage.morphology import skeletonize, thin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def process_skeletonization_lsd(input_path, output_dir, config, output_additional_name = None):
    """
    Multi-scale Line Segment Detection approach for skeleton production.
    
    Pipeline:
    1. Apply edge-preserving smoothing (bilateral filter)
    2. Run LSD at multiple blur scales to capture different line thicknesses
    3. Filter lines by length (remove short noise)
    4. Rasterize line segments to skeleton image
    5. Apply morphological thinning for clean 1-pixel skeleton
    
    Args:
        input_path (str): Path to grayscale input image.
        output_dir (str): Directory to save skeleton and debug plot.
        config (dict): Configuration dictionary with keys:
                       BLUR_SCALES, MIN_LINE_LENGTH, LINE_THICKNESS
    
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

    h, w = img_gray.shape
    
    # Configuration
    blur_scales = config.get("BLUR_SCALES", [1.0, 3.0, 5.0])
    min_line_length = config.get("MIN_LINE_LENGTH", 20)
    line_thickness = config.get("LINE_THICKNESS", 2)
    use_canny_boost = config.get("USE_CANNY_BOOST", True)
    canny_low = config.get("CANNY_LOW", 50)
    canny_high = config.get("CANNY_HIGH", 150)
    use_contour_refine = config.get("USE_CONTOUR_REFINE", False)
    contour_thickness = max(1, int(config.get("CONTOUR_THICKNESS", 1)))
    contour_dilate = max(0, int(config.get("CONTOUR_DILATE", 0)))
    contour_retr = str(config.get("CONTOUR_RETR", "EXTERNAL")).upper()
    
    print(f"  Running LSD (scales={blur_scales}, min_length={min_line_length})...")
    
    # Create LSD detector
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    
    # Accumulator for all detected lines
    all_lines = []
    
    # Multi-scale detection
    for sigma in blur_scales:
        if sigma > 0:
            # Gaussian blur at this scale
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)
        else:
            blurred = img_gray
        
        # Enhance contrast with CLAHE for better line detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Run LSD
        lines, widths, prec, nfa = lsd.detect(enhanced)
        
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Filter by minimum length
                if length >= min_line_length:
                    all_lines.append((x1, y1, x2, y2, length, sigma))
    
    print(f"    Detected {len(all_lines)} line segments across {len(blur_scales)} scales")
    
    # Optional: Add Canny edge boost for curved/organic lines
    canny_mask = None
    if use_canny_boost:
        # Bilateral filter for edge-preserving smoothing
        bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
        canny_edges = cv2.Canny(bilateral, canny_low, canny_high)
        canny_mask = canny_edges
    
    # Create rasterized line image
    line_image = np.zeros((h, w), dtype=np.uint8)
    
    for x1, y1, x2, y2, length, sigma in all_lines:
        # Draw lines with thickness based on scale
        thickness = max(1, line_thickness)
        cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    
    # Combine with Canny edges if enabled
    if canny_mask is not None:
        # Dilate Canny slightly to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        canny_dilated = cv2.dilate(canny_mask, kernel, iterations=1)
        line_image = cv2.bitwise_or(line_image, canny_dilated)
    
    # Gap bridging: morphological closing to connect nearby line endpoints
    gap_bridge_size = config.get("GAP_BRIDGE_SIZE", 0)
    if gap_bridge_size > 0:
        # Use elliptical kernel for more natural line bridging
        bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (gap_bridge_size, gap_bridge_size)
        )
        line_image = cv2.morphologyEx(line_image, cv2.MORPH_CLOSE, bridge_kernel)

    # Optional contour refinement (borrowed from binary OUTLINE strategy)
    if use_contour_refine:
        retr_map = {
            "EXTERNAL": cv2.RETR_EXTERNAL,
            "LIST": cv2.RETR_LIST,
            "TREE": cv2.RETR_TREE,
        }
        retr_mode = retr_map.get(contour_retr, cv2.RETR_EXTERNAL)
        contour_input = (line_image > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour_input, retr_mode, cv2.CHAIN_APPROX_NONE)

        contour_img = np.zeros_like(contour_input)
        cv2.drawContours(contour_img, contours, -1, 255, contour_thickness)

        if contour_dilate > 0:
            contour_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (contour_dilate, contour_dilate))
            contour_img = cv2.dilate(contour_img, contour_kernel)

        line_image = contour_img
        print(
            f"    Contour refine: {len(contours)} contours "
            f"(thickness={contour_thickness}, dilate={contour_dilate}, retr={contour_retr})"
        )
    
    # Skeletonize to get 1-pixel wide lines
    binary = line_image > 0
    skeleton = skeletonize(binary)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    # Save skeleton
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if output_additional_name is not None:
        base_name += f"_{output_additional_name}"

    os.makedirs(output_dir, exist_ok=True)

    skel_out_path = os.path.join(output_dir, f"{base_name}_lsd_skeleton.png")
    cv2.imwrite(skel_out_path, skeleton_uint8)
    
    # Create Color Overlay (Skeleton in Green)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_bgr[skeleton_uint8 > 0] = [0, 255, 0]  # BGR for Green
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, img_bgr)
    print(f"  Overlay saved: {overlay_path}")
    
    # Debug plot
    plot_path = os.path.join(output_dir, f"{base_name}_lsd_debug_plot.png")
    plt.figure(figsize=(20, 4))
    
    plt.subplot(1, 5, 1)
    plt.title("Original")
    plt.imshow(img_gray, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 5, 2)
    plt.title(f"LSD Lines ({len(all_lines)})")
    lsd_visual = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2, length, sigma in all_lines:
        cv2.line(lsd_visual, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    plt.imshow(lsd_visual, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 5, 3)
    plt.title("Canny Boost" if use_canny_boost else "No Canny")
    if canny_mask is not None:
        plt.imshow(canny_mask, cmap="gray")
    else:
        plt.imshow(np.zeros((h, w)), cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 5, 4)
    plt.title("Contour Mask" if use_contour_refine else "Combined Mask")
    plt.imshow(line_image, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 5, 5)
    plt.title("Final Skeleton")
    plt.imshow(skeleton, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()
    
    print(f"  LSD Skeleton saved: {skel_out_path}")
    return skel_out_path
