import os
import argparse
from src import preprocessing, skeletonizer_lsd, analyzer
from itertools import product

# =============================================================================
# ------------------------  LSD CONFIGURATION  --------------------------------
# =============================================================================

# Image Processing
TARGET_RESOLUTION = (1622, 1248)  # (width, height)

# Line Segment Detection Configuration
LSD_CONFIG = {
    # Multi-scale blur sigmas for detecting lines at different thicknesses
    # Lower values = finer details, Higher values = thicker/bolder lines
    "BLUR_SCALES": [1.0, 3.0, 6.0],
    
    # Minimum line length in pixels (filters out short noise segments)
    # Increased to 25 to reduce minor line clutter
    "MIN_LINE_LENGTH": 25,
    
    # Thickness for rasterizing detected lines before skeletonization
    "LINE_THICKNESS": 3,
    
    # Whether to add Canny edge detection for curved/organic lines
    # LSD only detects straight lines, Canny helps with curves
    "USE_CANNY_BOOST": True,
    
    # Canny thresholds (only used if USE_CANNY_BOOST is True)
    # Raised thresholds to reduce minor edge detection
    "CANNY_LOW": 50,
    "CANNY_HIGH": 150,
    
    # Gap bridging: morphological closing to connect nearby line endpoints
    # Higher values bridge larger gaps but may merge distinct lines
    "GAP_BRIDGE_SIZE": 7,
}

# Node Analysis
ANALYZER_CONFIG = {
    "OPP_THR_DEG": 160.0,
    "STRAIGHT_THR": 170.0,
    "Y_MIN": 75.0,
    "Y_MAX": 150.0
}

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
GRAY_DIR = os.path.join(BASE_DIR, "grayscale_resized")
SKEL_DIR = os.path.join(BASE_DIR, "skeletons_lsd")  # Separate dir for LSD results
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images_lsd")

# =============================================================================

def process_single_image(input_path, lsd_config, name_additions=None, flag_skip_analysis=False):
    """
    Process a single image through the LSD pipeline.
    """
    try:
        base_name = os.path.basename(input_path)
        print(f"\n--- [LSD] Processing {base_name} ---")
        
        # 1. Preprocessing
        name_only = os.path.splitext(base_name)[0]
        gray_path = os.path.join(GRAY_DIR, f"{name_only}.png")
        
        print(f"[{name_only}] 1. Preprocessing...")
        if not preprocessing.preprocess_image(input_path, gray_path, TARGET_RESOLUTION):
            return

        # 2. Skeletonization (LSD Method)
        print(f"[{name_only}] 2. Skeletonizing (LSD)...")
        skel_path = skeletonizer_lsd.process_skeletonization_lsd(gray_path, SKEL_DIR, lsd_config, output_additional_name=name_additions)
        if not skel_path:
            return

        # 3. Analysis
        if not flag_skip_analysis:
            print(f"[{name_only}] 3. Analyzing...")
            analyzer.process_analysis(skel_path, gray_path, OUTPUT_DIR, ANALYZER_CONFIG, output_additional_name=name_additions)
        else:
            print(f"[{name_only}] Skipping analysis...")
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()

def _parse_int_values(raw, arg_name):
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            raise ValueError(f"{arg_name} must be a comma-separated list of integers. Got: {raw}")

    if not values:
        raise ValueError(f"{arg_name} cannot be empty.")
    return values

def _build_sweep_configs(args):
    # Baseline LSD parameter space
    blur_scales_options = [
        [1.0, 3.0, 6.0],
        [1.0, 2.0, 4.0],
        [0.5, 1.5, 3.0],
        [0.5, 1.0, 2.0],
        [0.5, 1.0, 1.5],
        [0.25, 0.5, 1.0],
    ]
    min_line_length_options = [40, 75, 100]
    line_thickness_options = [5, 6, 7]
    gap_bridge_size_options = [5, 7, 9, 11, 13]

    # Expanded tunable ranges
    canny_low_options = _parse_int_values(args.canny_low_values, "--canny-low-values")
    canny_high_options = _parse_int_values(args.canny_high_values, "--canny-high-values")
    contour_thickness_options = _parse_int_values(args.contour_thickness_values, "--contour-thickness-values")
    contour_dilate_options = _parse_int_values(args.contour_dilate_values, "--contour-dilate-values")

    if any(v < 0 for v in canny_low_options + canny_high_options + contour_thickness_options + contour_dilate_options):
        raise ValueError("Threshold, thickness, and dilation values must be non-negative.")

    if args.canny_mode == "on":
        use_canny_boost_options = [True]
    elif args.canny_mode == "both":
        use_canny_boost_options = [False, True]
    else:
        use_canny_boost_options = [False]

    if args.contour_mode == "on":
        use_contour_refine_options = [True]
    elif args.contour_mode == "both":
        use_contour_refine_options = [False, True]
    else:
        use_contour_refine_options = [False]

    valid_canny_pairs = [
        (low, high) for low, high in product(canny_low_options, canny_high_options) if high > low
    ]
    if (True in use_canny_boost_options) and not valid_canny_pairs:
        raise ValueError("No valid Canny threshold pairs where high > low.")

    sweep_configs = []

    for blur_scales in blur_scales_options:
        for min_line_length in min_line_length_options:
            for line_thickness in line_thickness_options:
                for gap_bridge_size in gap_bridge_size_options:
                    for use_canny_boost in use_canny_boost_options:
                        # Do not multiply no-canny runs by threshold pairs.
                        if use_canny_boost:
                            canny_pairs = valid_canny_pairs
                        else:
                            canny_pairs = [(canny_low_options[0], canny_high_options[0])]

                        for canny_low, canny_high in canny_pairs:
                            for use_contour_refine in use_contour_refine_options:
                                # Do not multiply contour-off runs by contour params.
                                if use_contour_refine:
                                    contour_pairs = list(product(contour_thickness_options, contour_dilate_options))
                                else:
                                    contour_pairs = [(1, 0)]

                                for contour_thickness, contour_dilate in contour_pairs:
                                    lsd_config = {
                                        "BLUR_SCALES": blur_scales,
                                        "MIN_LINE_LENGTH": min_line_length,
                                        "LINE_THICKNESS": line_thickness,
                                        "USE_CANNY_BOOST": use_canny_boost,
                                        "CANNY_LOW": canny_low,
                                        "CANNY_HIGH": canny_high,
                                        "GAP_BRIDGE_SIZE": gap_bridge_size,
                                        "USE_CONTOUR_REFINE": use_contour_refine,
                                        "CONTOUR_THICKNESS": contour_thickness,
                                        "CONTOUR_DILATE": contour_dilate,
                                        "CONTOUR_RETR": "EXTERNAL",
                                    }

                                    blur_str = "-".join(map(str, blur_scales))
                                    canny_str = f"canny{canny_low}-{canny_high}" if use_canny_boost else "nocanny"
                                    contour_str = (
                                        f"_ctr_t{contour_thickness}_d{contour_dilate}"
                                        if use_contour_refine else ""
                                    )
                                    name_additions = (
                                        f"_blur{blur_str}_minlen{min_line_length}_thick{line_thickness}"
                                        f"_{canny_str}_gap{gap_bridge_size}{contour_str}"
                                    )

                                    sweep_configs.append((lsd_config, name_additions))

    return sweep_configs

def main():
    parser = argparse.ArgumentParser(description="Glyph System: Line Segment Detection Analysis")
    parser.add_argument("--input", help="Optional: specific input file")
    parser.add_argument(
        "--canny-mode",
        choices=["off", "on", "both"],
        default="off",
        help="Generate parameter sweeps with Canny disabled, enabled, or both.",
    )
    parser.add_argument(
        "--contour-mode",
        choices=["off", "on", "both"],
        default="off",
        help="Apply contour refinement after LSD mask generation: disabled, enabled, or both.",
    )
    parser.add_argument(
        "--canny-low-values",
        default="30,50,70",
        help="Comma-separated Canny low threshold values used when Canny is enabled.",
    )
    parser.add_argument(
        "--canny-high-values",
        default="100,150,200",
        help="Comma-separated Canny high threshold values used when Canny is enabled.",
    )
    parser.add_argument(
        "--contour-thickness-values",
        default="1,2",
        help="Comma-separated contour draw thickness values used when contour refinement is enabled.",
    )
    parser.add_argument(
        "--contour-dilate-values",
        default="0,3,5",
        help="Comma-separated contour dilation kernel sizes used when contour refinement is enabled.",
    )
    args = parser.parse_args()
    
    flag_use_loop = True #TODO: Pull this out to an argument
    flag_skip_analysis = True

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
    print(f"Canny mode: {args.canny_mode}")
    print(f"Contour mode: {args.contour_mode}")

    # Loop through param options
    if flag_use_loop:
        try:
            sweep_configs = _build_sweep_configs(args)
        except ValueError as e:
            print(f"Configuration error: {e}")
            return

        total_combos = len(sweep_configs)
        print(f"\nTotal parameter combinations: {total_combos}")
        print(f"Total runs: {total_combos * len(files)} (combos × images)\n")
        
        for combo_idx, (lsd_config, name_additions) in enumerate(sweep_configs, start=1):
            print(f"\n{'='*60}")
            print(f"[Combo {combo_idx}/{total_combos}] {name_additions}")
            print(f"{'='*60}")
            
            for input_path in files:
                process_single_image(input_path, lsd_config, name_additions, flag_skip_analysis)
    else:
        # Use default config
        for input_path in files:
            process_single_image(input_path, LSD_CONFIG)

    print("\n[LSD] Processing complete.")

if __name__ == "__main__":
    main()
