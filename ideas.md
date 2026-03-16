# Ideas for Future Development

This document outlines potential improvements and alternative approaches for the Depixelizing Pixel Art project.

---

## Current State

The project currently implements **two pipelines**:

### 1. Kopf-Lischinski (2011) - Vectorization
- **Approach:** Geometry-based contour extraction with B-spline curves
- **Pros:** Resolution-independent SVG output, preserves sharp corners
- **Cons:** Requires planarization heuristics, may lose fine details
- **Best for:** Clean pixel art with distinct color regions

### 2. Deep NN (Diego Inacio) - Coordinate Learning
- **Approach:** Neural network learns coordinates → color mapping per image
- **Pros:** No dataset needed, works on single images
- **Cons:** Slow (trains from scratch per image), image-specific model
- **Best for:** Small pixel art with limited color palettes

---

## Idea 1: Integrate Pre-trained Super-Resolution Models

### Real-ESRGAN Integration

**What:** Add support for Real-ESRGAN, a pre-trained GAN for image super-resolution.

**Implementation:**
```bash
pip install realesrgan
```

```python
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Load pre-trained model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
upsampler = RealESRGANer(scale=4, model_path='realesrgan-x4.pth', model=model)

# Upscale image
upscaled, _ = upsampler.enhance(low_res_image)
```

**Pros:**
- ✅ Works out-of-the-box
- ✅ Fast inference (< 1 second)
- ✅ No training required
- ✅ Good generalization

**Cons:**
- ❌ Not specialized for pixel art
- ❌ May introduce artifacts
- ❌ Large model (~67MB)

**Estimated effort:** 2-3 hours

---

## Idea 2: SwinIR for Pixel Art

**What:** SwinIR is a transformer-based model that performs better on pixel art than GANs.

**Why SwinIR:**
- Better edge preservation
- Less artifacts than ESRGAN
- Works well on anime/pixel art styles

**Implementation:**
```bash
pip install swinir
```

**Pre-trained models available:**
- Classical SR (div2k)
- Real-world SR (jpeg compression)
- Anime/art (specialized)

**Pros:**
- ✅ State-of-the-art results
- ✅ Better for pixel art than ESRGAN
- ✅ Pre-trained models available

**Cons:**
- ❌ Larger model (~120MB)
- ❌ Slower than ESRGAN
- ❌ Still not pixel-art specific

**Estimated effort:** 2-3 hours

---

## Idea 3: Train Custom Pixel Art Super-Resolution Model

### Dataset Generation

**Problem:** No large dataset of (pixel art → HD) pairs exists.

**Solution:** Generate synthetic training data from existing HD images.

**Pipeline:**
```
HD Image Dataset
       ↓
Downscale + Pixelate (bicubic/nearest)
       ↓
Create pairs: (pixelated, original)
       ↓
Train Super-Resolution model
```

**Data sources:**
- DIV2K (800 HD images)
- Flickr2K (2650 images)
- Anime datasets (Danbooru, etc.)
- Game spritesheets (with permission)

**Code for dataset generation:**
```python
from PIL import Image

def create_pixel_art_pair(hd_image, scale_factor=4):
    # Downscale
    low_res = hd_image.resize(
        (hd_image.width // scale_factor, 
         hd_image.height // scale_factor),
        Image.Resampling.BICUBIC
    )
    # Pixelate (nearest neighbor upscale)
    pixelated = low_res.resize(
        (hd_image.width, hd_image.height),
        Image.Resampling.NEAREST
    )
    return pixelated, hd_image
```

**Estimated dataset size:**
- Minimum: 1,000 images
- Recommended: 10,000+ images
- Training time: 2-7 days on GPU

---

### Model Architecture Options

#### Option A: ESRGAN (Enhanced Super-Resolution GAN)

```
Generator: RRDB blocks (Residual-in-Residual Dense Block)
Discriminator: PatchGAN classifier
Loss: Perceptual + Adversarial + Pixel loss
```

**Pros:**
- Proven architecture
- Good for textures

**Cons:**
- May introduce GAN artifacts
- Harder to train

#### Option B: SwinIR (Swin Transformer IR)

```
Architecture: Swin Transformer blocks
Self-attention for long-range dependencies
```

**Pros:**
- State-of-the-art results
- Better edge preservation

**Cons:**
- More complex
- Slower training

#### Option C: Simple CNN (for pixel art)

```
Input → Conv → ReLU → Conv → ... → PixelShuffle → Output
```

**Pros:**
- Fast training
- Simple architecture
- Good for pixel art (sharp edges)

**Cons:**
- May not capture complex patterns

---

### Training Pipeline

**Requirements:**
- GPU with 8GB+ VRAM (RTX 3060 or better)
- Python 3.8+
- PyTorch 2.0+

**Dependencies:**
```bash
pip install torch torchvision
pip install basicsr  # Basic Super-Resolution library
pip install lpips   # Perceptual loss
```

**Training script outline:**
```python
# train.py
import torch
from torch.utils.data import DataLoader
from basicsr.models.sr_model import SRModel
from basicsr.data.paired_image_dataset import PairedImageDataset

# Load dataset
train_dataset = PairedImageDataset('datasets/pixel_art_train')
train_loader = DataLoader(train_dataset, batch_size=16)

# Initialize model
model = SRModel(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    scale=4
)

# Train
for epoch in range(100):
    for batch in train_loader:
        loss = model.train_step(batch)
        loss.backward()
        optimizer.step()
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Scale factor | 4x |
| Batch size | 16-32 |
| Learning rate | 2e-4 |
| Epochs | 100-200 |
| Loss weights | L1: 1.0, Perceptual: 1.0, GAN: 5e-3 |

**Estimated training time:**
- 1,000 images: ~12 hours
- 10,000 images: ~5 days
- 50,000 images: ~3 weeks

---

## Idea 4: Hybrid Approach (Vectorization + NN Refinement)

**What:** Combine Kopf-Lischinski vectorization with neural refinement.

**Pipeline:**
```
Pixel Art
    ↓
Kopf-Lischinski → SVG contours
    ↓
Rasterize at target resolution
    ↓
Neural Refinement → Clean artifacts
    ↓
Final HD Image
```

**Pros:**
- Best of both worlds
- Sharp vector edges + smooth fills
- Can fix vectorization errors

**Cons:**
- Complex pipeline
- Two models to maintain

---

## Idea 5: Color Palette Learning

**What:** Train a model to predict the "correct" color palette for pixel art upscaling.

**Motivation:** Pixel art often uses dithering and limited palettes. A model could learn to:
- Identify the intended palette
- Remove dithering artifacts
- Fill smooth regions with solid colors

**Approach:**
```
Input: Pixel art region (8x8)
Output: Dominant color + variance
```

**Dataset:** Extract palettes from professional pixel art games.

---

## Idea 6: Interactive Tool / GUI

**What:** Build a user-friendly interface for the depixelization tools.

**Features:**
- Drag-and-drop image upload
- Real-time preview
- Parameter sliders (smoothness, corners, etc.)
- Side-by-side comparison
- Export options (SVG, PNG, WebP)

**Tech stack options:**
- **Desktop:** PyQt, Tkinter, or Tauri
- **Web:** Streamlit, Gradio, or React + Flask
- **CLI enhancement:** Rich library for terminal UI

**Example with Gradio:**
```python
import gradio as gr
from main import run_kopf_lischinski, run_deep_nn

def depixelize(image, method, scale):
    if method == "Vectorization":
        result = run_kopf_lischinski(image)
    else:
        result = run_deep_nn(image, scale)
    return result

gr.Interface(
    fn=depixelize,
    inputs=["image", "dropdown", "slider"],
    outputs="image"
).launch()
```

---

## Comparison Table

| Approach | Dataset | Training | Inference | Quality | Effort |
|----------|---------|----------|-----------|---------|--------|
| **Current: Kopf-Lischinski** | None | N/A | ~5 sec | ⭐⭐⭐⭐ | Done |
| **Current: Deep NN (per-image)** | Self | 2 min/image | 1 sec | ⭐⭐⭐ | Done |
| **Real-ESRGAN** | Pre-trained | N/A | 0.5 sec | ⭐⭐⭐⭐ | 2 hours |
| **SwinIR** | Pre-trained | N/A | 1 sec | ⭐⭐⭐⭐⭐ | 2 hours |
| **Custom ESRGAN** | Generate 10k | 5 days | 0.5 sec | ⭐⭐⭐⭐⭐ | 2 weeks |
| **Custom SwinIR** | Generate 10k | 7 days | 1 sec | ⭐⭐⭐⭐⭐ | 3 weeks |
| **Hybrid** | Optional | Varies | ~10 sec | ⭐⭐⭐⭐⭐ | 1 month |

---

## Recommended Next Steps

### Quick Win (1 day)
1. Integrate Real-ESRGAN or SwinIR
2. Add `--sr` flag to CLI
3. Compare results with current methods

### Medium Project (2 weeks)
1. Generate synthetic dataset (DIV2K + Flickr2K)
2. Train simple CNN for pixel art SR
3. Evaluate against pre-trained models

### Long-term Project (1-3 months)
1. Curate professional pixel art dataset
2. Train custom SwinIR model
3. Build interactive GUI
4. Publish results/paper

---

## References

### Papers
- **Kopf & Lischinski (2011):** Depixelizing Pixel Art - [Link](https://johanneskopf.de/publications/pixelart/)
- **Wang et al. (2021):** Real-ESRGAN - [arXiv](https://arxiv.org/abs/2107.10833)
- **Liang et al. (2021):** SwinIR - [arXiv](https://arxiv.org/abs/2108.10257)

### Codebases
- **BasicSR:** [GitHub](https://github.com/XPixelGroup/BasicSR)
- **Real-ESRGAN:** [GitHub](https://github.com/xinntao/Real-ESRGAN)
- **SwinIR:** [GitHub](https://github.com/JingyunLiang/SwinIR)

### Datasets
- **DIV2K:** [Link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Flickr2K:** [Link](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)
- **Danbooru (anime):** [Link](https://www.gwern.net/Danbooru2019)

---

## Notes

- All pre-trained model integrations should be optional dependencies
- Consider model size and download time for end users
- Document GPU requirements clearly
- Add benchmark comparisons in README
