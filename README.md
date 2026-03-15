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
pip install pillow numpy
```

## Usage

```bash
python main.py path/to/your/image.png
```

### Example

```bash
python main.py input/megaman.png
```

Output files are saved in `output/<image_name>_<timestamp>/` directory.

## Output Files

| File | Description |
|------|-------------|
| `grafo_similaridade.svg` | Phase 1: 8-connected similarity graph |
| `celulas_voronoi.svg` | Phase 3: Adaptive cells with vertex splitting |
| `contornos_splines.svg` | Phase 5: Final optimized B-spline curves |
| `contornos_splines_debug.svg` | Debug view with control points and corners |

## Project Structure

```
vibe-depixel/
├── main.py                 # Entry point and pipeline orchestration
├── core/
│   ├── color.py           # YUV conversion and color similarity
│   ├── graph.py           # Similarity graph and planarization (Phases 1-4)
│   ├── spline.py          # Cubic B-splines and optimization (Phase 5)
│   └── render.py          # SVG export
├── input/                 # Input images
└── output/                # Generated SVG files
```

## Key Classes

### `SimilarityGraph` (core/graph.py)
- `_build_initial_graph()`: Creates 8-connected pixel graph
- `planarize()`: Resolves diagonal crossings (Phase 2)
- `reshape_cells()`: Vertex splitting for adaptive cells (Phase 3)
- `extract_visible_contours()`: Extracts boundaries (Phase 4)

### `CubicBSpline` (core/spline.py)
- `evaluate(t)`: Evaluates cubic B-spline at parameter t
- `_detect_corners()`: Identifies sharp corners to preserve
- `optimize()`: Energy minimization with corner preservation
- `to_svg_path()`: Converts to SVG path string

### `SplineOptimizer` (core/spline.py)
- `_chain_segments()`: Connects segments into continuous paths
- `_simplify_path()`: Ramer-Douglas-Peucker simplification
- `get_splines()`: Creates optimized B-splines from contours

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Y_THRESHOLD` | 48/255 | Luminance similarity threshold |
| `U_THRESHOLD` | 7/255 | U channel similarity threshold |
| `V_THRESHOLD` | 6/255 | V channel similarity threshold |
| `CORNER_ANGLE_THRESHOLD` | 100° | Minimum angle for corner detection |
| `simplify_tolerance` | 0.1 | RDP simplification tolerance |
| `optimize_iterations` | 3 | Energy optimization iterations |

## Example Results

The algorithm successfully converts pixel art into smooth vector graphics while:
- Preserving sharp corners and features
- Removing pixel aliasing artifacts
- Creating resolution-independent output

## References

- Kopf, J., & Lischinski, D. (2011). **Depixelizing Pixel Art**. ACM SIGGRAPH 2011 Papers.
  - [Project Page](https://johanneskopf.de/publications/pixelart/)
  - [Paper PDF](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Kopf11.pdf)

## License

This implementation is for educational purposes. The original algorithm is described in the SIGGRAPH 2011 paper by Kopf and Lischinski.
