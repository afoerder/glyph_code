import cv2
import numpy as np
import math
import csv
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from concurrent.futures import ThreadPoolExecutor

try:
    import pytesseract
except ImportError:
    pytesseract = None

OCR_AVAILABLE = (pytesseract is not None) and (shutil.which("tesseract") is not None)

# =============================================================================
# ------------------------  V-Detector & Analysis Logic  ----------------------
# =============================================================================
# Adapted from main2.py

V_MIN_ARM_LENGTH = 4  
V_MIN_ANGLE_DEG = 5
V_MAX_ANGLE_DEG = 160
V_ENDPOINT_TOL   = 4
HOUGH_THRESH     = 10
HOUGH_MAX_GAP    = 3

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))

def detect_v_shapes(skel_uint8, min_arm_len=V_MIN_ARM_LENGTH, min_angle=V_MIN_ANGLE_DEG,
                    max_angle=V_MAX_ANGLE_DEG, endpoint_tol=V_ENDPOINT_TOL,
                    hough_thresh=HOUGH_THRESH, max_gap=HOUGH_MAX_GAP):
    """Detect V shapes in a skeleton image using Hough line segments."""
    img = (skel_uint8 > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(img, 1, np.pi / 180.0, hough_thresh,
                            minLineLength=max(3, min_arm_len), maxLineGap=max_gap)
    detections = []
    if lines is None:
        return detections

    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        segments.append(((x1, y1), (x2, y2)))

    # Naive pairwise check
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            s1 = segments[i]
            s2 = segments[j]
            
            # Check for shared endpoint within tolerance
            vertex = None
            endpoints_i = [s1[0], s1[1]]
            endpoints_j = [s2[0], s2[1]]
            
            p1_far = None
            p2_far = None

            matched = False
            for ei in endpoints_i:
                for ej in endpoints_j:
                    if _dist(ei, ej) <= endpoint_tol:
                        vertex = ((ei[0] + ej[0]) / 2, (ei[1] + ej[1]) / 2)
                        p1_far = s1[1] if ei == s1[0] else s1[0]
                        p2_far = s2[1] if ej == s2[0] else s2[0]
                        matched = True
                        break
                if matched: break
            
            if matched and vertex:
                # vectors
                v1 = (p1_far[0] - vertex[0], p1_far[1] - vertex[1])
                v2 = (p2_far[0] - vertex[0], p2_far[1] - vertex[1])
                angle = _angle_between(v1, v2)
                l1 = _dist(vertex, p1_far)
                l2 = _dist(vertex, p2_far)

                if min_angle <= angle <= max_angle and min(l1, l2) >= min_arm_len:
                    detections.append({
                        'vertex': vertex,
                        'angle': angle,
                        'lengths': (l1, l2),
                        'segments': (s1, s2)
                    })
    return detections

def neighbor_count_8(skel01):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    # cv2 filter2D is fast
    neighbors = cv2.filter2D(skel01.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return neighbors * skel01  # mask by skeleton

def _snap_to_skeleton(pt, skel01, radius=3):
    px, py = pt
    px = int(round(px))
    py = int(round(py))
    h, w = skel01.shape
    
    # scan neighborhood
    best_pt = None
    min_d = float('inf')
    
    y_min, y_max = max(0, py - radius), min(h, py + radius + 1)
    x_min, x_max = max(0, px - radius), min(w, px + radius + 1)
    
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if skel01[y, x] > 0:
                d = (x - px)**2 + (y - py)**2
                if d < min_d:
                    min_d = d
                    best_pt = (x, y)
    return best_pt

def degree_candidates(skel01, deg_map, min_deg=3, dedupe_radius=6):
    ys, xs = np.where((skel01 > 0) & (deg_map >= min_deg))
    points = []
    for x, y in zip(xs, ys):
        points.append((x, y))
    
    # Dedupe
    final_pts = []
    processed = [False] * len(points)
    
    for i in range(len(points)):
        if processed[i]: continue
        processed[i] = True
        cluster = [points[i]]
        
        for j in range(i + 1, len(points)):
            if not processed[j]:
                if _dist(points[i], points[j]) <= dedupe_radius:
                    cluster.append(points[j])
                    processed[j] = True
        
        # Centroid
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        # Snap back to nearest skel pixel in cluster
        best = cluster[0]
        bd = float('inf')
        for p in cluster:
            d = (p[0]-cx)**2 + (p[1]-cy)**2
            if d < bd:
                bd = d
                best = p
        final_pts.append(best)
        
    return final_pts

def run_v_detector_and_filter(skel255, skel01, deg_map, min_arm_len, endpoint_tol, hough_thresh, hough_max_gap, deg_min_for_v, snap_radius):
    raw_dets = detect_v_shapes(skel255, min_arm_len=min_arm_len, endpoint_tol=endpoint_tol, hough_thresh=hough_thresh, max_gap=hough_max_gap)
    
    valid_pts = []
    for d in raw_dets:
        v = d['vertex']
        snapped = _snap_to_skeleton(v, skel01, radius=snap_radius)
        if snapped:
            sx, sy = snapped
            if deg_map[sy, sx] >= deg_min_for_v:
                valid_pts.append(snapped)
    return raw_dets, valid_pts

def cluster_points(points, radius=5):
    if not points: return []
    final = []
    # Greedy clustering
    pool = list(points)
    while pool:
        seed = pool.pop(0)
        cluster = [seed]
        keep = []
        for p in pool:
            if _dist(seed, p) <= radius:
                cluster.append(p)
            else:
                keep.append(p)
        pool = keep
        # Centroid
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        final.append((int(round(cx)), int(round(cy))))
    return final

def _process_ocr_patch(args):
    """Worker function with MUCH more aggressive preprocessing for OCR"""
    cx, cy, skel255_shape, skel255, skel01, patch_size, dilate_iterations, scale_factor, debug_config, idx, total = args
    
    # Extract LARGER patch for context
    half = patch_size // 2
    y_min, y_max = max(0, cy-half), min(skel255_shape[0], cy+half)
    x_min, x_max = max(0, cx-half), min(skel255_shape[1], cx+half)
    patch = skel255[y_min:y_max, x_min:x_max]
    patch_skel01 = skel01[y_min:y_max, x_min:x_max]
    
    if patch.size == 0:
        return None
    
    # === AGGRESSIVE OCR PREPROCESSING ===
    
    # 1. SMOOTH the skeleton first (remove noise)
    patch_smoothed = cv2.GaussianBlur(patch, (5, 5), 1.5)
    
    # 2. MUCH THICKER dilation (make it look like bold text)
    thick_kernel = np.ones((5, 5), np.uint8)  # Larger kernel
    patch_thick = cv2.dilate(patch_smoothed, thick_kernel, iterations=dilate_iterations)
    
    # 3. Another blur to make edges smooth like fonts
    patch_smooth = cv2.GaussianBlur(patch_thick, (5, 5), 2.0)
    
    # 4. Morphological closing to fill small gaps (make shapes more solid)
    close_kernel = np.ones((3, 3), np.uint8)
    patch_closed = cv2.morphologyEx(patch_smooth, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    
    # 5. MAJOR UPSCALING (OCR needs ~50-150px characters)
    patch_scaled = cv2.resize(patch_closed, None, fx=scale_factor, fy=scale_factor, 
                              interpolation=cv2.INTER_CUBIC)
    
    # 6. Threshold to clean binary
    _, patch_binary = cv2.threshold(patch_scaled, 100, 255, cv2.THRESH_BINARY)
    
    # 7. Invert for OCR (black text on white)
    patch_inv = cv2.bitwise_not(patch_binary)
    
    # 8. MUCH MORE padding (OCR needs whitespace)
    pad_size = 40  # Bigger padding
    patch_padded = cv2.copyMakeBorder(patch_inv, pad_size, pad_size, pad_size, pad_size, 
                                      cv2.BORDER_CONSTANT, value=255)
    
    # 9. One final blur to anti-alias (make it look printed)
    patch_final = cv2.GaussianBlur(patch_padded, (3, 3), 0.5)
    
    # === OCR ATTEMPT ===
    text = ""
    if OCR_AVAILABLE:
        custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=TYXU'
        try:
            text = pytesseract.image_to_string(patch_final, config=custom_config).strip()
        except Exception:
            text = ""
    
    # Debug save BOTH versions
    if debug_config.get("save_patches", False):
        debug_dir = debug_config.get("dir", "./ocr_debug")
        os.makedirs(debug_dir, exist_ok=True)
        # Save original skeleton patch
        cv2.imwrite(f"{debug_dir}/0_original_{cx}_{cy}.png", patch)
        # Save final OCR input
        cv2.imwrite(f"{debug_dir}/1_ocr_input_{cx}_{cy}_{text or 'NONE'}.png", patch_final)
    
    # === GEOMETRIC FALLBACK ===
    if text not in ['T', 'Y', 'X', 'U']:
        # Component counting method
        temp_patch = patch_skel01.copy()
        center_y, center_x = patch_skel01.shape[0] // 2, patch_skel01.shape[1] // 2
        
        # Remove larger area around junction (3x3 -> 5x5)
        temp_patch[max(0, center_y-2):center_y+3, max(0, center_x-2):center_x+3] = 0
        
        # Count in larger radius
        mask = np.zeros_like(patch_skel01, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), min(20, patch_skel01.shape[0]//2), 1, -1)
        
        local_region = temp_patch * mask
        n_components, _ = cv2.connectedComponents(local_region)
        k = n_components - 1
        
        if k == 2:
            label = "T"
        elif k == 3:
            label = "Y"
        elif k == 4:
            label = "X"
        else:
            label = "U"
        
        method = "GEOM_NO_OCR" if not OCR_AVAILABLE else "GEOM"
    else:
        label = text
        method = "OCR"
    
    print(f"  [{method}] Junction {idx+1}/{total} at ({cx},{cy}): {label}")
    
    return {'x': int(cx), 'y': int(cy), 'label': label, 'method': method}

def find_intersections_hybrid(skel01, skel255, config=None):
    """Use degree detection to find candidates, OCR to classify them (Parallel with improved preprocessing)"""
    if config is None:
        config = {}
    
    # 1. Find high-degree nodes
    kernel = np.ones((3,3), np.uint8)
    neighbors = cv2.filter2D(skel01.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT) * skel01
    candidates = np.argwhere(neighbors >= 4)  # y, x format
    
    # Deduplicate candidates
    raw_points = [(int(x), int(y)) for y, x in candidates]
    cluster_radius = config.get("CLUSTER_RADIUS", 15)
    candidates_clustered = cluster_points(raw_points, radius=cluster_radius)
    
    # OCR preprocessing parameters
    patch_size = config.get("OCR_PATCH_SIZE", 60)
    dilate_iterations = config.get("OCR_DILATE_ITERATIONS", 5)
    scale_factor = config.get("OCR_SCALE_FACTOR", 4)
    max_workers = config.get("OCR_MAX_WORKERS", 8)
    
    # Debug configuration
    debug_config = {
        "save_patches": config.get("DEBUG_SAVE_PATCHES", False),
        "dir": config.get("DEBUG_DIR", "./ocr_debug")
    }
    
    total = len(candidates_clustered)
    
    print(f"  [OCR] Processing {total} junctions using {max_workers} threads...")
    print(f"  [OCR] Patch size: {patch_size}px, Dilation: {dilate_iterations}x, Scale: {scale_factor}x")
    if not OCR_AVAILABLE:
        print("  [OCR] pytesseract/tesseract not available; using geometric fallback.")
    
    # Prepare arguments for parallel execution
    tasks = [
        (cx, cy, skel255.shape, skel255, skel01, patch_size, dilate_iterations, scale_factor, debug_config, i, total)
        for i, (cx, cy) in enumerate(candidates_clustered)
    ]
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch_results = list(executor.map(_process_ocr_patch, tasks))
    
    # Filter out None and add to results
    results = [r for r in batch_results if r is not None]
    
    return results

class Arm:
    def __init__(self, endpoint, path):
        self.endpoint = endpoint
        self.path = path
        # vector from start to end
        if len(path) >= 2:
            self.vec = (path[-1][0] - path[0][0], path[-1][1] - path[0][1])
        else:
            self.vec = (0, 0)
    
    def angle_deg(self):
        return math.degrees(math.atan2(self.vec[1], self.vec[0])) % 360

def trace_arms(skel01, center, max_len=30):
    cx, cy = center
    h, w = skel01.shape
    visited = np.zeros_like(skel01, dtype=bool)
    visited[cy, cx] = True
    
    # Initial neighbors
    stack = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and skel01[ny, nx] > 0:
                stack.append([(cx, cy), (nx, ny)])
                visited[ny, nx] = True
    
    arms = []
    for path in stack:
        curr = path[-1]
        while len(path) < max_len:
            # find next neighbor
            px, py = curr
            found_next = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h and skel01[ny, nx] > 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        path.append((nx, ny))
                        curr = (nx, ny)
                        found_next = True
                        break 
                if found_next: break
            if not found_next:
                break
        
        # Only keep if decent length
        if len(path) >= 3:
            arms.append(Arm(endpoint=path[-1], path=path))
            
    return arms

def classify_node(arms, opp_thr_deg, straight_thr, y_min, y_max, ring_opp_pairs=0):
    k = len(arms)
    angles = sorted([a.angle_deg() for a in arms])
    
    # Calculate angle differences
    diffs = []
    for i in range(k):
        diff = (angles[(i + 1) % k] - angles[i]) % 360
        diffs.append(diff)
    
    # Check for opposite pairs (angles ~180 apart)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(angles[i] - angles[j])
            diff = min(diff, 360 - diff)
            if diff >= opp_thr_deg:
                pairs.append((i, j, diff))
    
    info = {"k": k, "angles": diffs, "opp_pairs": [p[2] for p in pairs], "ring_opp_pairs": ring_opp_pairs}
    
    # Classification logic (mirroring main2.py)
    label = "U"
    
    if k == 1:
        label = "U" # Endpoint?
    elif k == 3:
        # T vs Y
        # T: any angle close to 180 (straight)
        # Y: angles ~120
        is_T = any(d >= straight_thr for d in diffs)
        if is_T:
            label = "T"
        else:
            # Check Y bounds
            if all(y_min <= d <= y_max for d in diffs):
                label = "Y"
            else:
                # Fallback: find nearest archetype? 
                # For now, default to Y if not T, or U if very skewed?
                # main2.py logic: "Otherwise choose nearest archetype" -- simplified here:
                label = "Y" 
    elif k >= 4:
        # X: two opposite pairs
        # Logic: do we have at least 2 distinct pairs?
        if len(pairs) >= 2:
            label = "X"
        elif ring_opp_pairs >= 2: # Rescue from ring heuristic
            label = "X"
        else:
            label = "U" # Complex?
            
    # Rescue rule for 3-arm X (e.g. one arm missed but geometry is X-like?)
    # main2.py: "Rescue rule from 8-neighbor ring if only 3 arms." -> if k=3 and ring says 2 pairs -> X
    if k == 3 and ring_opp_pairs >= 2:
        label = "X"

    return label, info

def process_analysis(skeleton_path, original_image_path, output_dir, config, output_additional_name=None):
    if not os.path.exists(skeleton_path):
        print(f"Error: Skeleton file not found: {skeleton_path}")
        return

    skel_img = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    if skel_img is None:
        print("Error reading skeleton.")
        return
        
    skel01 = (skel_img > 127).astype(np.uint8)
    skel255 = skel01 * 255
    
    print(f"  Running OCR-Hybrid analysis on {os.path.basename(skeleton_path)}...")
    results_list = find_intersections_hybrid(skel01, skel255, config)
    
    # Prepare overlay
    if os.path.exists(original_image_path):
        overlay = cv2.imread(original_image_path)
    else:
        overlay = cv2.cvtColor(skel255, cv2.COLOR_GRAY2BGR)
        
    for r in results_list:
        cx, cy, label = r['x'], r['y'], r['label']
        
        # Color based on label
        color = (0, 255, 0) # U (Green)
        if label == "T": color = (255, 0, 0)   # Blue
        if label == "Y": color = (0, 255, 255) # Yellow
        if label == "X": color = (0, 0, 255)   # Red
        
        cv2.circle(overlay, (cx, cy), 3, color, -1)
        cv2.putText(overlay, label, (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save
    base = os.path.splitext(os.path.basename(original_image_path))[0]
    if output_additional_name is not None:
        base += f"_{output_additional_name}"
        
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{base}_nodes.csv")
    out_img = os.path.join(output_dir, f"{base}_analyzed.png")
    
    # Add skeleton overlay in red with alpha per config or default
    skeleton_color = np.zeros_like(overlay)
    skeleton_color[skel01 > 0] = [0, 0, 255]  # Red in BGR
    alpha = config.get("OVERLAY_ALPHA", 0.2)
    overlay = cv2.addWeighted(overlay, 1.0, skeleton_color, alpha, 0)
    
    cv2.imwrite(out_img, overlay)
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "label", "method"])
        writer.writeheader()
        for r in results_list:
            writer.writerow(r)
            
    print(f"  Analysis saved: {out_img}")
    print(f"  Node data saved: {out_csv}")
    
    # Stats
    counts = {"T": 0, "Y": 0, "X": 0, "U": 0}
    for r in results_list:
        lbl = r['label']
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"  Summary: T={counts['T']}, Y={counts['Y']}, X={counts['X']}, U={counts['U']}")
