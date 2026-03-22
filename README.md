# vibe-depixel

Pixel art upscaling research project. Combines classical vectorization, coordinate-based neural networks, and learned super-resolution (SR) models trained on synthetic LR/HR pairs generated with Stable Diffusion.

---

## Pipelines

### 1. Kopf-Lischinski Vectorization
Implements the depixelization pipeline from Kopf & Lischinski (2011): similarity graph → planarization → cell reshaping → contour extraction → cubic B-spline fitting.

```bash
python main.py input/sprite.png
```

Outputs SVG files: similarity graph, spline contours, debug overlay, Voronoi cells.

### 2. Deep NN (coordinate-based)
Per-image neural network that learns the color palette and maps 2D coordinates to colors. Useful for single-image upscaling without a dataset.

```bash
python main.py input/sprite.png --nn --upscale 16 --epochs 1000
```

### 3. Super-Resolution (learned, dataset-based)
The main research pipeline. Trains convolutional and transformer models on paired LR/HR datasets to generalize across sprites.

```bash
# Train
python main.py --train --lr-dir dataset/train/lr --hr-dir dataset/train/hr --preset gemini

# Inference
python main.py input/sprite.png --sr --checkpoint checkpoints/gemini/sr_model_best.pth --preset gemini
```

---

## SR Models

| Preset | Architecture | Params | Notes |
|--------|-------------|--------|-------|
| `default` | EDSRLite 64ch / 16 blocks | ~1.5M | Fast baseline |
| `gemini` | EDSRLite 128ch / 32 blocks | ~11M | Main research model |
| `esrgan` | RRDBNet via spandrel | ~16.7M | Fine-tuned from pretrained |
| `swinir` | Swin Transformer via spandrel | varies | Fine-tuned from pretrained |

Spandrel presets (`esrgan`, `swinir`) require a pretrained `.pth` file placed in the project root and passed via `--pretrained`.

---

## Loss Functions

| Name | Description |
|------|-------------|
| `SRLoss` | L1 + gradient + optional VGG19 perceptual. Standard RGB loss. |
| `LumSRLoss` | YCbCr-weighted: Y channel gets full L1+gradient penalty, Cb/Cr get 10x less. Robust to color drift in synthetic datasets. |

`LumSRLoss` perceptual options: `freq` (FFT magnitude on Y, no pretrained weights), `dists` (Deep Image Structure and Texture Similarity, requires `piq`), `vgg` (VGG19 on RGB), `none`.

Enable with `--lum-loss` flag or `"lum_loss": True` in experiment config.

---

## Training Defaults per Preset

Defined in `core/sr_model.py → TRAINING_DEFAULTS`. Applied automatically — override per experiment config if needed.

| Preset | patch_size | grad_clip | weight_decay | optimizer |
|--------|-----------|-----------|--------------|-----------|
| default | 64 | 1.0 | 0.0 | Adam |
| gemini | 64 | 1.0 | 0.0 | Adam |
| esrgan | 64 | 0.1 | 0.01 | AdamW |
| swinir | 128 | 0.1 | 0.01 | AdamW |

Prodigy optimizer (adaptive LR) is also supported: set `"use_adamw": "prodigy"` in the experiment config. Requires `pip install prodigyopt`.

LR schedule: CosineAnnealingLR from initial LR down to `1e-6` over all epochs.

---

## Hyperparameter Sweep

```bash
# Run all experiments
python experiments.py

# Run specific preset group
python experiments.py --only esrgan_lum --epochs 200

# Re-run even if already in leaderboard
python experiments.py --only swinir_lum --epochs 200 --force

# Show current leaderboard
python experiments.py --show

# Preview without training
python experiments.py --dry-run
```

Experiments are tracked in `checkpoints/experiments/leaderboard.json`. Only the top-K checkpoints (default: 4) are kept on disk — lower-ranked `.pth` files are pruned automatically.

PSNR is measured in RGB for standard experiments and in Y (luminance) for `lum_loss` experiments, since the model optimizes for Y. Both values are logged per epoch.

---

## Dataset Tools

### Sync from ComfyUI output
```bash
python sync_dataset.py
```
Copies new LR/HR pairs from the ComfyUI output folder to `dataset/train/`.

### Analyze dataset quality
```bash
python analyze_dataset.py --top-bad 20 --save-report report.csv --save-plots
```
Computes per-pair metrics (SSIM, color drift in LAB, sharpness ratio) and flags problematic pairs. Color drift is the most reliable indicator for SD-generated datasets.

### Filter by color drift
```bash
python filter_dataset.py --top-pct 20
```
Copies the cleanest 80% of pairs (by color drift) to `dataset/train_clean/`. Does not modify the original dataset.

### Color correction
```bash
python color_correct_dataset.py --method combined --strength 0.5
```
Corrects HR color drift relative to LR using Reinhard color transfer + histogram matching. Output in `dataset/train_cc/`.

| Method | Effect |
|--------|--------|
| `reinhard` | Shifts LAB mean/std — gentle, preserves SD gradients |
| `histogram` | Forces per-channel histogram match — stronger, may cause banding |
| `combined` | Reinhard + partial histogram blend (default) |
| `adaptive` | Like combined but scales with drift severity |

---

## Evaluation

```bash
python eval_model.py --checkpoint checkpoints/experiments/gemini_lr1e-4_bs16_perc_best.pth --preset gemini
python eval_model.py --checkpoint path/to/model.pth --save-report eval.csv --save-plots --save-grid
```

Metrics: PSNR, SSIM, edge score (Sobel cosine similarity), color error (LAB MAE), optional LPIPS (`--lpips`).

Uses the same val split as training (seed=42, 10% holdout) so results are comparable across checkpoints.

---

## GUI

```bash
python gui_sr.py
python gui_sr.py --checkpoint checkpoints/gemini/sr_model_best.pth --preset gemini
```

Three tabs:
- **LR vs SR** — click an image, drag the slider to compare original and upscaled
- **Model A vs B** — load two checkpoints, compare side by side with training curves
- **GIF** — load animated GIFs, preview upscaled animation with FPS control

---

## Dataset Structure

```
dataset/
├── train/
│   ├── lr/          # pixel art originals
│   └── hr/          # SD-upscaled at 4x resolution
├── train_clean/     # filtered by filter_dataset.py
├── train_cc/        # color-corrected by color_correct_dataset.py
└── train_cc_clean/  # color-corrected + filtered (recommended for training)
```

Filenames must match between `lr/` and `hr/`. Supported formats: png, jpg, bmp, webp.
Minimum LR size: 64×64 (128×128 for SwinIR). HR must be exactly 4× the LR resolution.

---

## Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy scipy scikit-image matplotlib spandrel PyQt6 piq prodigyopt
```

---

## TODO

- [ ] **LR warmup** — ramp LR from near-zero to target over first N epochs before cosine decay. Especially useful for fine-tuning ESRGAN/SwinIR to avoid disrupting pretrained weights on early batches.
- [ ] Fix SD color drift at the source (ComfyUI): img2img with low denoising strength or ControlNet Tile to preserve original color palette during HR generation.
- [ ] Evaluate on standardized pixel art benchmark (e.g. PixelPerfect set) for cross-project comparison.
- [ ] Per-character/batch drift analysis to identify which ComfyUI generation sessions produced the worst pairs.
