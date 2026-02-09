import os
import sys
import argparse
from src import preprocessing, skeletonizer_bin, analyzer
from itertools import product

# =============================================================================
# ------------------------  BINARY CONFIGURATION  -----------------------------
# =============================================================================

# Image Processing
TARGET_RESOLUTION = (1622, 1248)  # (width, height)

# Binary Skeletonizer Configuration
BIN_CONFIG = {
    "METHOD": "OUTLINE",  # "OUTLINE" or "SKELETON"
    "OUTLINE_THICKNESS": 1,
    "OUTLINE_DILATE": 5,   # Higher value merges closer lines
    "RE_SKELETONIZE": True, # Bring merged lines back to 1px
    "INVERT": False,      # Set True if background is light
    "THRESH_TYPE": "OTSU", # "OTSU" or "SIMPLE"
    "THRESH_VAL": 127,    # Used if THRESH_TYPE is "SIMPLE"
    "MORPH_OPEN_K": 0,    # 0 to skip
    "MORPH_CLOSE_K": 0,   # 0 to skip
}

# Node Analysis
ANALYZER_CONFIG = {
    "OPP_THR_DEG": 160.0,
    "STRAIGHT_THR": 170.0,
    "Y_MIN": 75.0,
    "Y_MAX": 150.0,
    
    # Resolution-focused detection parameters
    "V_MIN_ARM_LEN": 12,
    "DEDUPE_RADIUS": 15,
    "CLUSTER_RADIUS": 20,
    "TRACE_MAX_LEN": 60,
    "MIN_ARM_LEN_CLASSIFY": 15,
    "FINAL_SNAP_RADIUS": 10,
    "OCR_MAX_WORKERS": 8,
    
    # NEW: OCR-specific parameters
    "OCR_PATCH_SIZE": 60,           # Larger patches for more context
    "OCR_DILATE_ITERATIONS": 5,     # Make lines thick like fonts (try 3-7)
    "OCR_SCALE_FACTOR": 4,          # Upscale for OCR (try 2-5)
    "DEBUG_SAVE_PATCHES": True,     # Save patches to see what OCR sees
    "DEBUG_DIR": "./ocr_debug",     # Where to save debug patches
    
    # Overlay visualization
    "OVERLAY_ALPHA": 0.2
}

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
GRAY_DIR = os.path.join(BASE_DIR, "grayscale_resized")
SKEL_DIR = os.path.join(BASE_DIR, "skeletons_bin")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images_binary")

# =============================================================================

def process_single_image(input_path, bin_config, name_additions=None, flag_skip_analysis=False, analyzer_config=None):
    """
    Process a single image through the binary pipeline.
    """
    # Use default config if none provided
    if analyzer_config is None:
        analyzer_config = ANALYZER_CONFIG
        
    try:
        base_name = os.path.basename(input_path)
        print(f"\n--- [BINARY] Processing {base_name} ---")
        
        # 1. Preprocessing
        name_only = os.path.splitext(base_name)[0]
        gray_path = os.path.join(GRAY_DIR, f"{name_only}.png")
        
        print(f"[{name_only}] 1. Preprocessing...")
        if not preprocessing.preprocess_image(input_path, gray_path, TARGET_RESOLUTION):
            return

        # 2. Skeletonization (Binary Method)
        print(f"[{name_only}] 2. Skeletonizing (Binary)...")
        skel_path = skeletonizer_bin.process_skeletonization_bin(gray_path, SKEL_DIR, bin_config, output_additional_name=name_additions)
        if not skel_path:
            return

        # 3. Analysis
        if not flag_skip_analysis:
            print(f"[{name_only}] 3. Analyzing...")
            analyzer.process_analysis(skel_path, gray_path, OUTPUT_DIR, analyzer_config, output_additional_name=name_additions)
        else:
            print(f"[{name_only}] Skipping analysis...")
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Glyph System: Binary Mask Analysis")
    parser.add_argument("--input", help="Optional: specific input file")
    parser.add_argument("--loop", action="store_true", help="Run with parameter loop")
    parser.add_argument("--tune-ocr", action="store_true", help="Run OCR parameter tuning sweep")
    args = parser.parse_args()
    
    flag_skip_analysis = False # Enable analysis

    # Ensure directories exist
    for d in [INPUT_DIR, GRAY_DIR, SKEL_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # Get files
    if args.input:
        if os.path.exists(args.input):
            files = [args.input]
        else:
            print(f"Error: Input {args.input} not found.")
            return
    else:
        files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if not files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} images to process.")

    if args.tune_ocr:
        # OCR parameter tuning sweep
        print("\n=== OCR PARAMETER TUNING MODE ===")
        
        patch_sizes = [50, 60, 70]
        dilations = [3, 5, 7]
        scales = [3, 4, 5]
        
        total_combos = len(patch_sizes) * len(dilations) * len(scales)
        print(f"Total OCR parameter combinations: {total_combos}")
        
        combo_idx = 0
        for patch_size, dilate, scale in product(patch_sizes, dilations, scales):
            combo_idx += 1
            
            print(f"\n[OCR Combo {combo_idx}/{total_combos}] patch={patch_size}, dilate={dilate}, scale={scale}")
            
            test_config = ANALYZER_CONFIG.copy()
            test_config.update({
                "OCR_PATCH_SIZE": patch_size,
                "OCR_DILATE_ITERATIONS": dilate,
                "OCR_SCALE_FACTOR": scale,
                "DEBUG_SAVE_PATCHES": True,
                "DEBUG_DIR": f"./ocr_debug/p{patch_size}_d{dilate}_s{scale}"
            })
            
            name_additions = f"_ocr_p{patch_size}_d{dilate}_s{scale}"
            
            # Process each image with this OCR config
            for input_path in files:
                process_single_image(input_path, BIN_CONFIG, name_additions, 
                                   flag_skip_analysis=False, analyzer_config=test_config)
        
        print("\n=== OCR TUNING COMPLETE ===")
        print(f"Check the ocr_debug/ subdirectories to see which parameters work best!")
        
    elif args.loop:
        # Define parameter variations to test
        method_options = ["OUTLINE"]
        thickness_options = [1, 2]
        dilate_options = [3, 5, 7, 9]
        reskel_options = [True]
        invert_options = [False]
        thresh_type_options = ["OTSU"]
        thresh_val_options = [127]
        morph_open_options = [0]
        morph_close_options = [0]
        
        all_options = [
            method_options,
            thickness_options,
            dilate_options,
            reskel_options,
            invert_options,
            thresh_type_options,
            thresh_val_options,
            morph_open_options,
            morph_close_options
        ]
        
        total_combos = 1
        for opt in all_options:
            total_combos *= len(opt)
        print(f"\nTotal parameter combinations: {total_combos}")
        
        combo_idx = 0
        for method, thickness, dilate, reskel, invert, thresh_type, thresh_val, morph_open, morph_close in product(*all_options):
            combo_idx += 1
            
            bin_config = {
                "METHOD": method,
                "OUTLINE_THICKNESS": thickness,
                "OUTLINE_DILATE": dilate,
                "RE_SKELETONIZE": reskel,
                "INVERT": invert,
                "THRESH_TYPE": thresh_type,
                "THRESH_VAL": thresh_val,
                "MORPH_OPEN_K": morph_open,
                "MORPH_CLOSE_K": morph_close,
            }
            
            reskel_str = "re" if reskel else "nore"
            name_additions = f"_{method}_t{thickness}_d{dilate}_{reskel_str}_v{thresh_val}"
            
            print(f"\n[Combo {combo_idx}/{total_combos}] {name_additions}")
            for input_path in files:
                process_single_image(input_path, bin_config, name_additions, flag_skip_analysis)
    else:
        # Use default config
        for input_path in files:
            process_single_image(input_path, BIN_CONFIG, flag_skip_analysis=flag_skip_analysis)

    print("\n[BINARY] Processing complete.")

if __name__ == "__main__":
    main()