# Depixelizing Pixel Art

A Python implementation of the **Kopf-Lischinski 2011** algorithm for converting pixel art into resolution-independent vector graphics.

## Overview

This project implements the algorithm described in the paper ["Depixelizing Pixel Art"](https://johanneskopf.de/publications/pixelart/) by Johannes Kopf and Dani Lischinski (SIGGRAPH 2011). The algorithm converts low-resolution pixel art images into smooth, scalable vector graphics (SVG).

## Algorithm Pipeline

The implementation follows the 5-phase pipeline from the paper:

### Phase 1: Similarity Graph Construction
- Builds an 8-connected graph where pixels with similar colors are connected
- Uses YUV color space with thresholds from the paper:
  - Y (luminance): 48/255
  - U: 7/255
  - V: 6/255

### Phase 2: Ambiguity Removal (Planarization)
- Resolves diagonal crossings to make the graph planar
- Handles fully-connected 2×2 blocks (removes both diagonals)
- Applies three heuristics to decide which diagonal to remove:
  - **Island Heuristic**: Preserves diagonals that prevent isolated pixels
  - **Curve Heuristic**: Favors smaller connected components
  - **Sparse Heuristic**: Considers local pixel density

### Phase 3: Vertex Splitting (Cell Reshaping)
- Splits corner vertices to create adaptive cells
- Pixels connected diagonally get split vertices offset by 0.25 units
- Creates smooth transitions between connected regions

### Phase 4: Contour Extraction
- Extracts visible boundaries between disconnected components
- Uses cell geometry to determine boundary vertices
- Chains segments into continuous paths

### Phase 5: Spline Fitting & Optimization
- Converts paths to **cubic B-spline curves**
- Applies **Ramer-Douglas-Peucker** simplification
- **Energy minimization** to smooth curves while preserving corners
- Corner detection based on angle threshold (default: 100°)

## Installation

```bash
pip install -r requirements.txt
```

### Minimal installation (vectorization only)

```bash
pip install pillow numpy
```

### Full installation (with Deep Learning upscaling)

```bash
pip install pillow numpy tensorflow-cpu matplotlib
```

## Usage

### Basic Usage (Vectorization - Kopf-Lischinski 2011)

```bash
python main.py path/to/your/image.png
```

### Deep Neural Network Upscaling

```bash
python main.py path/to/your/image.png --nn --upscale 16 --epochs 1000
```

### Run Both Pipelines

```bash
python main.py path/to/your/image.png --both
```

### Examples

```bash
# Vectorization only (default)
python main.py input/megaman.png

# Deep NN upscaling with 16x factor
python main.py input/megaman.png --nn -u 16 -e 1000

# Run both pipelines
python main.py input/megaman.png --both --upscale 16

# Save trained model
python main.py input/megaman.png --nn --save-model
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--nn`, `--neural-network` | Use Deep Neural Network instead of vectorization |
| `--both` | Run both vectorization AND Deep NN pipelines |
| `--upscale`, `-u` | Upscale factor for Deep NN (default: 16) |
| `--epochs`, `-e` | Training epochs for Deep NN (default: 1000) |
| `--save-model` | Save the trained Deep NN model (.h5 file) |

Output files are saved in `output/<image_name>_<timestamp>/` directory.

## Output Files

### Kopf-Lischinski Vectorization Pipeline

| File | Description |
|------|-------------|
| `grafo_similaridade.svg` | Phase 1: 8-connected similarity graph |
| `celulas_voronoi.svg` | Phase 3: Adaptive cells with vertex splitting |
| `contornos_splines.svg` | Phase 5: Final optimized B-spline curves |
| `contornos_splines_debug.svg` | Debug view with control points and corners |

### Deep Neural Network Pipeline

| File | Description |
|------|-------------|
| `upscaled_Nx.png` | Upscaled image (N = upscale factor) |
| `reconstruction.png` | Reconstruction at original resolution |
| `comparison.png` | Side-by-side: original vs upscaled |
| `training_history.png` | Loss and accuracy curves |
| `model.h5` | Saved trained model (if --save-model) |

## Project Structure

```
vibe-depixel/
├── main.py                 # Entry point and pipeline orchestration
├── requirements.txt        # Python dependencies
├── core/
│   ├── color.py           # YUV conversion and color similarity
│   ├── graph.py           # Similarity graph and planarization (Phases 1-4)
│   ├── spline.py          # Cubic B-splines and optimization (Phase 5)
│   ├── deep_nn.py         # Deep Neural Network upscaling (alternative)
│   └── render.py          # SVG export
├── input/                 # Input images
└── output/                # Generated SVG/PNG files
```

## Key Classes

### Kopf-Lischinski Pipeline

#### `SimilarityGraph` (core/graph.py)
- `_build_initial_graph()`: Creates 8-connected pixel graph
- `planarize()`: Resolves diagonal crossings (Phase 2)
- `reshape_cells()`: Vertex splitting for adaptive cells (Phase 3)
- `extract_visible_contours()`: Extracts boundaries (Phase 4)

#### `CubicBSpline` (core/spline.py)
- `evaluate(t)`: Evaluates cubic B-spline at parameter t
- `_detect_corners()`: Identifies sharp corners to preserve
- `optimize()`: Energy minimization with corner preservation
- `to_svg_path()`: Converts to SVG path string

#### `SplineOptimizer` (core/spline.py)
- `_chain_segments()`: Connects segments into continuous paths
- `_simplify_path()`: Ramer-Douglas-Peucker simplification
- `get_splines()`: Creates optimized B-splines from contours

### Deep Neural Network Pipeline

#### `DeepNNDepixelizer` (core/deep_nn.py)
- `fit()`: Trains the network on input image coordinates → colors
- `predict()`: Generates upscaled image at target resolution
- `predict_train()`: Reconstructs original image
- `save_model()` / `load_model()`: Model persistence
- `plot_training_history()`: Visualizes training progress

## Algorithm Parameters

### Kopf-Lischinski Vectorization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Y_THRESHOLD` | 48/255 | Luminance similarity threshold |
| `U_THRESHOLD` | 7/255 | U channel similarity threshold |
| `V_THRESHOLD` | 6/255 | V channel similarity threshold |
| `CORNER_ANGLE_THRESHOLD` | 100° | Minimum angle for corner detection |
| `simplify_tolerance` | 0.1 | RDP simplification tolerance |
| `optimize_iterations` | 3 | Energy optimization iterations |

### Deep Neural Network Upscaling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `upscale_factor` | 16 | Output image magnification |
| `epochs` | 1000 | Training iterations |
| `batch_size` | 32 | Mini-batch size |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `dropout_rate` | 0.125 | Dropout regularization rate |
| `validation_split` | 0.1 | Fraction for validation |
| `patience` | 50 | Early stopping patience |

## Example Results

### Kopf-Lischinski Vectorization

The algorithm successfully converts pixel art into smooth vector graphics while:
- Preserving sharp corners and features
- Removing pixel aliasing artifacts
- Creating resolution-independent output

### Deep Neural Network Upscaling

The Deep NN approach:
- Learns the discrete color palette from input
- Maps 2D coordinates to colors via one-hot encoding
- Generates upscaled images without blur or interpolation artifacts
- Can upscale to arbitrary resolutions

## References

- Kopf, J., & Lischinski, D. (2011). **Depixelizing Pixel Art**. ACM SIGGRAPH 2011 Papers.
  - [Project Page](https://johanneskopf.de/publications/pixelart/)
  - [Paper PDF](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Kopf11.pdf)

- Inacio, D. **Depixelizing Pixel Art using Deep Neural Networks**.
  - [GitHub Notebook](https://github.com/diegoinacio/creative-coding-notebooks/blob/master/ML-and-AI/pixel-art-depixelization-deepNN.ipynb)

## License

This implementation is for educational purposes. The original algorithm is described in the SIGGRAPH 2011 paper by Kopf and Lischinski.
