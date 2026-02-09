# glyph_code

A small image-processing project for turning visual textures into clean line skeletons and then comparing parameter settings to find the best result.

The main goal is to help you explore "glyph-like" structures from images (cracks, leaves, circuit traces, spider webs, etc.) by running repeatable processing pipelines and reviewing outputs side by side.

## What this repo does

This repo has two pipelines:

1. `main_lsd.py` (LSD pipeline)
2. `main_bin.py` (binary threshold pipeline)

Both pipelines:

1. Read images from `input_images/`
2. Resize + normalize to grayscale
3. Build a 1-pixel skeleton image
4. Optionally run node analysis
5. Write outputs to disk for inspection

## Acronyms and terms (plain language)

- **LSD**: **Line Segment Detector**. An OpenCV method that finds straight line segments in an image.
- **Canny**: A classic edge detector (finds boundaries/edges).
- **OCR**: **Optical Character Recognition** (text-reading system). In this repo it is used as part of node/shape analysis in the binary pipeline.
- **CLI**: **Command-Line Interface** (running scripts from terminal commands).
- **Skeletonization**: Reducing thick lines to a clean 1-pixel-wide centerline.
- **Contour refinement**: Re-drawing detected shapes using contour boundaries to simplify/clean masks.

## Repository layout

- `main_lsd.py`: parameter sweep runner for LSD-based skeleton generation
- `main_bin.py`: binary-mask pipeline and OCR tuning workflow
- `src/preprocessing.py`: resizing and grayscale preparation
- `src/skeletonizer_lsd.py`: LSD + optional Canny + optional contour refinement + skeletonization
- `src/skeletonizer_bin.py`: binary/outline skeletonization path
- `src/analyzer.py`: node/angle analysis and CSV/image output
- `param_viewer.html`: browser viewer for comparing generated parameter outputs
- `input_images/`: source images
- `grayscale_resized/`: intermediate grayscale images
- `skeletons_lsd/`: LSD pipeline output images
- `skeletons_bin/`: binary pipeline output images
- `output_images_lsd/`: analyzed LSD overlays/exports (when analysis is enabled)
- `output_images_binary/`: analyzed binary overlays/exports
- `ocr_debug/`: OCR debug patches and artifacts

## Quick start (local)

From repo root:

```bash
pip install opencv-python scikit-image matplotlib numpy
```

Run LSD pipeline on all images:

```bash
python main_lsd.py \
  --canny-mode on \
  --contour-mode on \
  --canny-low-values 50 \
  --canny-high-values 150 \
  --contour-thickness-values 1 \
  --contour-dilate-values 0
```

Run LSD pipeline on one image:

```bash
python main_lsd.py \
  --input input_images/Fingerprint.png \
  --canny-mode on \
  --contour-mode on \
  --canny-low-values 50 \
  --canny-high-values 150 \
  --contour-thickness-values 1 \
  --contour-dilate-values 0
```

Run binary pipeline (default settings):

```bash
python main_bin.py
```

Run binary pipeline OCR tuning sweep:

```bash
python main_bin.py --tune-ocr
```

## How many parameter combinations are generated?

`main_lsd.py` creates combinations from fixed options plus your CLI values.

Current built-in sweep sizes:

- blur scales: 6 options
- min line length: 3 options
- line thickness: 3 options
- gap bridge size: 5 options

Then multiplied by your Canny and contour choices.

Example with:

- `--canny-mode on`
- `--contour-mode on`
- one Canny pair (`50-150`)
- one contour thickness (`1`)
- one contour dilate (`0`)

Total combos = `6 * 3 * 3 * 5 * 1 * 1 * 1 = 270`

If you process 10 images, total runs = `2700` image-runs.

## Using the parameter viewer

Open:

- Local file: `param_viewer.html`
- GitHub Pages: `https://afoerder.github.io/glyph_code/`

The viewer builds filenames from selected parameters and tries to load matching files from:

- `skeletons_lsd/` for skeleton/overlay/debug
- `output_images_lsd/` for analyzed outputs

If an image says "not found", usually either:

1. That exact parameter combo was not generated, or
2. Analysis was skipped so analyzed images do not exist.

## Running on Google Colab (recommended for large sweeps)

Use Colab + Google Drive to avoid local disk pressure.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive
!git clone https://github.com/afoerder/glyph_code.git
%cd /content/drive/MyDrive/glyph_code
!pip -q install opencv-python scikit-image matplotlib numpy
```

Run a sweep command (example):

```bash
!python main_lsd.py \
  --canny-mode on \
  --contour-mode on \
  --canny-low-values 50 \
  --canny-high-values 150 \
  --contour-thickness-values 1 \
  --contour-dilate-values 0
```

Notes:

- This code path is mostly CPU-based in standard Colab.
- GPU does not provide big speed gains unless the pipeline is rewritten for GPU APIs.

## Practical tips

- Start with one image first (`--input ...`) to confirm output quality.
- Keep Canny and contour value lists short while tuning.
- Large sweeps create many output files quickly; monitor Drive/storage usage.
- Keep generated outputs out of git unless you intentionally want to publish a curated subset.

## Current behavior to know

In `main_lsd.py`:

- `flag_use_loop` is currently hardcoded `True` (always runs sweep mode)
- `flag_skip_analysis` is currently hardcoded `True` (analysis step skipped)

So by default, LSD analyzed outputs may not be produced unless that flag is changed in code.

## License

No license file is currently included. Add one if you plan to share or reuse this code publicly.
