import cv2
import numpy as np
import os

def preprocess_image(input_path, output_path, target_size=(1622, 1248)):
    """
    Reads an image, converts to grayscale, resizes to target size, and saves it.
    
    Args:
        input_path (str): Path to input image.
        output_path (str): Path to save processed image.
        target_size (tuple): (width, height) target resolution.
    
    Returns:
        str: Path to the processed image, or None on failure.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return None

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image: {input_path}")
        return None

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Resize
    # cv2.resize expects (width, height)
    processed = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, processed)
    print(f"Processed: {input_path} -> {output_path} ({target_size})")
    
    return output_path
