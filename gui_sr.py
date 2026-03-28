"""
Super-Resolution comparison GUI.

Tab 1 — LR vs SR:   pick a model, click an image to compare with original.
Tab 2 — A vs B:     pick two models, click an image to compare them head-to-head.
                    Training metrics (loss/PSNR) shown alongside the visual comparison.

Usage:
    python gui_sr.py
    python gui_sr.py --checkpoint checkpoints/default/sr_model_best.pth --preset default
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageSequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QLabel, QSplitter,
    QListWidgetItem, QTabWidget, QComboBox, QGroupBox, QSizePolicy,
    QScrollArea, QProgressBar, QSpinBox, QCheckBox,
)
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QTimer

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core.sr_model import EDSRLite, load_model, ESRGAN_PRESET, PRESETS, SPANDREL_PRESETS

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
OUTPUT_DIR = Path("output/sr")


# ── Inference worker ──────────────────────────────────────────────────────────

class InferenceWorker(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, image_path: Path, model_a: torch.nn.Module,
                 device: torch.device, model_b: torch.nn.Module | None = None):
        super().__init__()
        self.image_path = image_path
        self.model_a = model_a
        self.model_b = model_b
        self.device = device

    def _infer(self, model, arr: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            out = model(t.to(self.device))
        return (out.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def run(self):
        try:
            arr = np.array(Image.open(self.image_path).convert('RGB'))
            self.model_a.eval()
            a = self._infer(self.model_a, arr)
            b = self._infer(self.model_b, arr) if self.model_b else arr
            self.finished.emit(a, b)
        except Exception as e:
            self.error.emit(str(e))


# ── Slider compare widget ─────────────────────────────────────────────────────

class CompareWidget(QWidget):
    def __init__(self, left_label="A", right_label="B"):
        super().__init__()
        self.left_label = left_label
        self.right_label = right_label
        self.left_px: QPixmap | None = None
        self.right_px: QPixmap | None = None
        self.slider_pos = 0.5
        self._status = "Select an image from the list"
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

    def set_images(self, left: np.ndarray, right: np.ndarray):
        self.left_px = self._arr_to_px(left)
        self.right_px = self._arr_to_px(right)
        self.slider_pos = 0.5
        self._status = ""
        self.update()

    def set_status(self, msg: str):
        self._status = msg
        self.left_px = self.right_px = None
        self.update()

    @staticmethod
    def _arr_to_px(arr: np.ndarray) -> QPixmap:
        h, w, ch = arr.shape
        return QPixmap.fromImage(QImage(arr.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888))

    def _rect(self) -> QRect:
        if not self.right_px:
            return QRect(0, 0, self.width(), self.height())
        iw, ih = self.right_px.width(), self.right_px.height()
        s = min(self.width() / iw, self.height() / ih)
        dw, dh = int(iw * s), int(ih * s)
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self.left_px:
            p.setPen(Qt.GlobalColor.gray)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._status)
            p.end(); return
        r = self._rect()
        mode = Qt.TransformationMode.SmoothTransformation
        ls = self.left_px.scaled(r.width(), r.height(), Qt.AspectRatioMode.IgnoreAspectRatio, mode)
        rs = self.right_px.scaled(r.width(), r.height(), Qt.AspectRatioMode.IgnoreAspectRatio, mode)
        sp = int(r.width() * self.slider_pos)
        p.drawPixmap(r.x(), r.y(), ls, 0, 0, sp, r.height())
        p.drawPixmap(r.x() + sp, r.y(), rs, sp, 0, r.width() - sp, r.height())
        p.setPen(Qt.GlobalColor.white)
        p.drawLine(r.x() + sp, r.y(), r.x() + sp, r.y() + r.height())
        if sp > 50:           p.drawText(r.x() + 8,      r.y() + 20, self.left_label)
        if r.width()-sp > 50: p.drawText(r.x() + sp + 8, r.y() + 20, self.right_label)
        p.end()

    def mouseMoveEvent(self, e):
        if not self.left_px: return
        r = self._rect()
        self.slider_pos = max(0.0, min(1.0, (e.position().x() - r.x()) / r.width()))
        self.update()


# ── Metrics panel (matplotlib embedded) ──────────────────────────────────────

class MetricsPanel(QWidget):
    def __init__(self, title: str = ""):
        super().__init__()
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        self._draw_empty()

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center", color="gray",
                transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()

    def load(self, metrics_path: Path | str | None):
        self.fig.clear()
        path = Path(metrics_path) if metrics_path else None
        if not path or not path.exists():
            self._draw_empty()
            return

        data = json.loads(path.read_text())
        epochs = list(range(1, len(data["loss"]) + 1))

        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        ax1.plot(epochs, data["loss"], color="#e05c5c", linewidth=1.0)
        ax1.set_ylabel("Loss", fontsize=8)
        ax1.set_yscale("symlog")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=7)
        if self._title:
            ax1.set_title(self._title, fontsize=9, fontweight="bold")

        ax2.plot(epochs, data["psnr"], color="#4c8bf5", linewidth=1.0)
        best_ep = data["psnr"].index(max(data["psnr"])) + 1
        best_v = max(data["psnr"])
        ax2.axvline(best_ep, color="#4c8bf5", linestyle="--", alpha=0.4)
        ax2.set_ylabel("PSNR (dB)", fontsize=8)
        ax2.set_xlabel("Epoch", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=7)
        ax2.annotate(f"{best_v:.2f} dB", xy=(best_ep, best_v),
                     xytext=(4, -10), textcoords="offset points", fontsize=7, color="#4c8bf5")

        self.canvas.draw()


# ── Model selector widget ─────────────────────────────────────────────────────

class ModelSelector(QGroupBox):
    model_ready = pyqtSignal(object, object)  # model, metrics_path

    def __init__(self, title="Model", default_preset="default", default_checkpoint=""):
        super().__init__(title)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: torch.nn.Module | None = None
        self._checkpoint = default_checkpoint

        layout = QVBoxLayout(self)

        # Preset row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Preset:"))
        self.combo = QComboBox()
        self.combo.addItems(list(PRESETS.keys()) + sorted(SPANDREL_PRESETS))
        self.combo.setCurrentText(default_preset)
        self.combo.currentTextChanged.connect(self.load_model)
        row1.addWidget(self.combo)
        layout.addLayout(row1)

        # Checkpoint row
        row2 = QHBoxLayout()
        self.btn_ckpt = QPushButton("Checkpoint…")
        self.btn_ckpt.clicked.connect(self._pick_ckpt)
        self.lbl_ckpt = QLabel(Path(default_checkpoint).name if default_checkpoint else "—")
        self.lbl_ckpt.setStyleSheet("color:gray;font-size:10px;")
        self.lbl_ckpt.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row2.addWidget(self.btn_ckpt)
        row2.addWidget(self.lbl_ckpt)
        layout.addLayout(row2)

        # Status
        self.lbl_status = QLabel("Not loaded")
        self.lbl_status.setStyleSheet("color:gray;font-size:10px;")
        layout.addWidget(self.lbl_status)

        if default_checkpoint and os.path.exists(default_checkpoint):
            self.load_model()

    def _pick_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", "", "Model (*.pth)")
        if p:
            self._checkpoint = p
            self.lbl_ckpt.setText(Path(p).name)
            self._model = None
            # Auto-detectar preset pelo conteúdo do arquivo
            try:
                raw = torch.load(p, map_location="cpu", weights_only=True)
                if isinstance(raw, dict) and "preset" in raw:
                    detected = raw["preset"]
                else:
                    detected = sorted(SPANDREL_PRESETS)[0]  # arquivo externo → esrgan
                self.combo.blockSignals(True)
                self.combo.setCurrentText(detected)
                self.combo.blockSignals(False)
            except Exception:
                pass
            self.load_model()

    def load_model(self):
        preset = self.combo.currentText()

        if preset in SPANDREL_PRESETS and not (self._checkpoint and os.path.exists(self._checkpoint)):
            self.lbl_status.setText("Select a .pth file")
            self.lbl_status.setStyleSheet("color:orange;font-size:10px;")
            return

        self.lbl_status.setText("Loading…")
        QApplication.processEvents()
        try:
            scale = 4
            state_dict = None
            if self._checkpoint and os.path.exists(self._checkpoint):
                if preset in SPANDREL_PRESETS:
                    # Spandrel carrega tudo via pretrained_path, sem state_dict separado
                    pass
                else:
                    raw = torch.load(self._checkpoint, map_location=self.device, weights_only=True)
                    if isinstance(raw, dict) and "state_dict" in raw:
                        scale = raw.get("scale", 4)
                        state_dict = raw["state_dict"]
                    else:
                        state_dict = raw  # legacy plain state_dict
            model = load_model(preset, self.device,
                               pretrained_path=self._checkpoint or None,
                               scale=scale)
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._model = model
            n = sum(p.numel() for p in model.parameters())
            self.lbl_status.setText(f"Ready — {n:,} params")
            self.lbl_status.setStyleSheet("color:green;font-size:10px;")
            metrics_path = Path(self._checkpoint).parent / "metrics.json" if self._checkpoint else None
            self.model_ready.emit(model, str(metrics_path) if metrics_path and metrics_path.exists() else None)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.lbl_status.setText(f"Error: {e}")
            self.lbl_status.setStyleSheet("color:red;font-size:10px;")

    @property
    def model(self):
        return self._model


# ── Tab 1: LR vs SR ───────────────────────────────────────────────────────────

class LRvsSRTab(QWidget):
    def __init__(self, device, default_preset, default_checkpoint):
        super().__init__()
        self.device = device
        self._worker = None
        self._current_path = None

        self.selector = ModelSelector("Model", default_preset, default_checkpoint)
        self.compare = CompareWidget("LR (original)", "SR (model)")
        self.metrics = MetricsPanel("Training metrics")

        self.selector.model_ready.connect(lambda m, mp: self.metrics.load(mp))

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.compare)
        right.addWidget(self.metrics)
        right.setSizes([500, 250])

        layout = QHBoxLayout(self)
        layout.addWidget(self.selector, 0)
        layout.addWidget(right, 1)

    def process_image(self, path: Path):
        if not self.selector.model:
            self.compare.set_status("Load a model first")
            return
        self._current_path = path
        self.compare.set_status(f"Processing {path.name}…")
        self._worker = InferenceWorker(path, self.selector.model, self.device)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(lambda e: self.compare.set_status(f"Error: {e}"))
        self._worker.start()

    def _on_done(self, sr_arr, _):
        lr_arr = np.array(Image.open(self._current_path).convert('RGB'))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sr_arr).save(OUTPUT_DIR / self._current_path.name)
        self.compare.set_images(lr_arr, sr_arr)


# ── Tab 2: Model A vs B ───────────────────────────────────────────────────────

class AvsBTab(QWidget):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self._worker = None

        self.selector_a = ModelSelector("Model A")
        self.selector_b = ModelSelector("Model B")
        self.compare = CompareWidget("Model A", "Model B")
        self.metrics_a = MetricsPanel("Model A — training")
        self.metrics_b = MetricsPanel("Model B — training")

        self.selector_a.model_ready.connect(lambda m, mp: self.metrics_a.load(mp))
        self.selector_b.model_ready.connect(lambda m, mp: self.metrics_b.load(mp))

        # Left column: selector A + metrics A
        col_a = QWidget()
        col_a_layout = QVBoxLayout(col_a)
        col_a_layout.addWidget(self.selector_a)
        col_a_layout.addWidget(self.metrics_a)

        # Right column: selector B + metrics B
        col_b = QWidget()
        col_b_layout = QVBoxLayout(col_b)
        col_b_layout.addWidget(self.selector_b)
        col_b_layout.addWidget(self.metrics_b)

        # Bottom columns side by side
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.addWidget(col_a)
        bottom_layout.addWidget(col_b)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.compare)
        splitter.addWidget(bottom)
        splitter.setSizes([450, 350])

        QVBoxLayout(self).addWidget(splitter)

    def process_image(self, path: Path):
        if not self.selector_a.model or not self.selector_b.model:
            self.compare.set_status("Load both models first")
            return
        self.compare.set_status(f"Processing {path.name}…")
        self._worker = InferenceWorker(
            path, self.selector_a.model, self.device, model_b=self.selector_b.model
        )
        self._worker.finished.connect(self.compare.set_images)
        self._worker.error.connect(lambda e: self.compare.set_status(f"Error: {e}"))
        self._worker.start()


# ── Tab 3: GIF temporal coherence ────────────────────────────────────────────

class GifWorker(QThread):
    """Loads GIF frames and runs SR model on each one."""
    progress = pyqtSignal(int, int)           # current, total
    finished = pyqtSignal(list, list, int)    # nn_frames, sr_frames, fps
    error = pyqtSignal(str)

    def __init__(self, gif_path: Path, model: torch.nn.Module, device: torch.device, scale: int = 4):
        super().__init__()
        self.gif_path = gif_path
        self.model = model
        self.device = device
        self.scale = scale

    def run(self):
        try:
            print(f"[GifWorker] opening {self.gif_path}")
            gif = Image.open(self.gif_path)
            print(f"[GifWorker] format={gif.format} mode={gif.mode} size={gif.size}")
            frames_raw = []
            durations = []
            for frame in ImageSequence.Iterator(gif):
                frames_raw.append(np.array(frame.convert('RGB')))
                durations.append(frame.info.get('duration', 100))

            print(f"[GifWorker] {len(frames_raw)} frames, durations sample={durations[:5]}")
            avg_duration = sum(durations) / len(durations) if durations else 0
            fps = max(1, round(1000 / avg_duration)) if avg_duration > 0 else 12
            print(f"[GifWorker] fps={fps}")

            nn_frames = []
            sr_frames = []
            self.model.eval()
            total = len(frames_raw)

            for i, arr in enumerate(frames_raw):
                self.progress.emit(i, total)
                if i == 0:
                    print(f"[GifWorker] frame shape={arr.shape} dtype={arr.dtype}")
                # Nearest-neighbor upscale
                nn = np.kron(arr, np.ones((self.scale, self.scale, 1), dtype=np.uint8))
                nn_frames.append(nn)
                # SR model
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    out = self.model(t.to(self.device))
                sr = (out.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                sr_frames.append(sr)
                if i == 0:
                    print(f"[GifWorker] first SR frame shape={sr.shape}")

            print(f"[GifWorker] done, emitting {len(nn_frames)} frames")
            self.progress.emit(total, total)
            self.finished.emit(nn_frames, sr_frames, fps)
        except Exception as e:
            import traceback
            print(f"[GifWorker] ERROR: {e}")
            traceback.print_exc()
            self.error.emit(str(e))


class FrameCanvas(QLabel):
    """Simple label that shows numpy frames scaled to fit."""
    def __init__(self, title: str, smooth: bool = False):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: black;")
        self.setMinimumSize(300, 200)
        self._transform = (Qt.TransformationMode.SmoothTransformation if smooth
                           else Qt.TransformationMode.FastTransformation)
        self.setText(title)

    def show_frame(self, arr: np.ndarray):
        h, w, ch = arr.shape
        qimg = QImage(arr.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            self._transform,
        )
        self.setPixmap(px)


RIFE_BASE = Path("F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/comfyui-frame-interpolation")
RIFE_CKPT = RIFE_BASE / "ckpts/rife/rife49.pth"


def _load_rife(device: torch.device):
    """Import IFNet from the ComfyUI RIFE node, mocking the comfy dependency."""
    import types
    if "comfy" not in sys.modules:
        comfy_mock = types.ModuleType("comfy")
        mm_mock = types.ModuleType("comfy.model_management")
        mm_mock.get_torch_device = lambda: device
        mm_mock.soft_empty_cache = lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None
        sys.modules["comfy"] = comfy_mock
        sys.modules["comfy.model_management"] = mm_mock
        comfy_mock.model_management = mm_mock

    rife_dir = str(RIFE_BASE / "vfi_models" / "rife")
    if rife_dir not in sys.path:
        sys.path.insert(0, rife_dir)

    from rife_arch import IFNet  # noqa
    model = IFNet(arch_ver="4.7")
    state = torch.load(str(RIFE_CKPT), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


class RifeWorker(QThread):
    """Runs RIFE 2x interpolation on a list of SR frames."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list, list)   # sr_interp, nn_sync
    error = pyqtSignal(str)

    def __init__(self, sr_frames: list, nn_frames: list, device: torch.device):
        super().__init__()
        self.sr_frames = sr_frames
        self.nn_frames = nn_frames
        self.device = device

    def run(self):
        try:
            model = _load_rife(self.device)
            scale_list = [8.0, 4.0, 2.0, 1.0]
            n = len(self.sr_frames)
            sr_out = []
            nn_out = []

            with torch.inference_mode():
                for i in range(n - 1):
                    self.progress.emit(i, n - 1)

                    f0 = torch.from_numpy(self.sr_frames[i]).permute(2, 0, 1).unsqueeze(0).float().div(255).to(self.device)
                    f1 = torch.from_numpy(self.sr_frames[i + 1]).permute(2, 0, 1).unsqueeze(0).float().div(255).to(self.device)
                    ts = torch.tensor([[[[0.5]]]], dtype=torch.float32, device=self.device)

                    mid = model(f0, f1, ts, scale_list)
                    mid_np = (mid.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    sr_out.append(self.sr_frames[i])
                    sr_out.append(mid_np)
                    nn_out.append(self.nn_frames[i])
                    nn_out.append(self.nn_frames[i])   # duplicate original for sync

            # append last frame
            sr_out.append(self.sr_frames[-1])
            nn_out.append(self.nn_frames[-1])

            self.progress.emit(n - 1, n - 1)
            self.finished.emit(sr_out, nn_out)
        except Exception as e:
            self.error.emit(str(e))


class GifTab(QWidget):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self._worker = None
        self._nn_frames: list[np.ndarray] = []
        self._sr_frames: list[np.ndarray] = []
        self._nn_frames_interp: list[np.ndarray] = []
        self._sr_frames_interp: list[np.ndarray] = []
        self._frame_idx = 0
        self._rife_worker = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._next_frame)

        # Controls
        self.selector = ModelSelector("Model")
        self.btn_gif = QPushButton("Load GIF…")
        self.btn_gif.clicked.connect(self._pick_gif)
        self.lbl_gif = QLabel("No GIF loaded")
        self.lbl_gif.setStyleSheet("color:gray;font-size:10px;")

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(12)
        self.fps_spin.valueChanged.connect(self._update_fps)
        fps_row.addWidget(self.fps_spin)
        fps_row.addStretch()

        self.chk_interpolate = QCheckBox("Interpolate (RIFE)")
        self.chk_interpolate.setChecked(False)
        self.chk_interpolate.toggled.connect(self._on_interpolate_toggled)

        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.addWidget(self.selector)
        cl.addWidget(self.btn_gif)
        cl.addWidget(self.lbl_gif)
        cl.addWidget(self.progress)
        cl.addLayout(fps_row)
        cl.addWidget(self.chk_interpolate)
        cl.addStretch()
        controls.setFixedWidth(260)

        # Canvases
        self.canvas_nn = FrameCanvas("Nearest Neighbor (original)", smooth=False)
        self.canvas_sr = FrameCanvas("SR Model output", smooth=True)

        canvases = QWidget()
        canvas_layout = QHBoxLayout(canvases)
        canvas_layout.addWidget(self.canvas_nn)
        canvas_layout.addWidget(self.canvas_sr)

        layout = QHBoxLayout(self)
        layout.addWidget(controls)
        layout.addWidget(canvases, 1)

    def _pick_gif(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GIF", "", "GIF (*.gif)")
        if not path:
            return
        if not self.selector.model:
            self.lbl_gif.setText("Load a model first")
            return
        self._timer.stop()
        self._nn_frames.clear()
        self._sr_frames.clear()
        self.lbl_gif.setText(Path(path).name)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._worker = GifWorker(Path(path), self.selector.model, self.device)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(lambda e: self.lbl_gif.setText(f"Error: {e}"))
        self._worker.start()

    def _on_progress(self, current: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.progress.setFormat(f"Processing frame {current}/{total}")

    def _on_done(self, nn_frames: list, sr_frames: list, fps: int):
        self._nn_frames = nn_frames
        self._sr_frames = sr_frames
        self._nn_frames_interp.clear()
        self._sr_frames_interp.clear()
        self._original_fps = fps
        self._frame_idx = 0
        self.fps_spin.setValue(fps)
        self.progress.setVisible(False)
        self.lbl_gif.setText(f"{len(nn_frames)} frames @ {fps} fps")
        self._timer.start(1000 // fps)
        if self.chk_interpolate.isChecked():
            self._run_rife()

    def _on_interpolate_toggled(self, checked: bool):
        if not self._sr_frames:
            return
        self._frame_idx = 0
        if checked:
            if self._sr_frames_interp:
                # already cached — just switch to interpolated playback
                fps = self._original_fps * 2
                self.fps_spin.blockSignals(True)
                self.fps_spin.setValue(fps)
                self.fps_spin.blockSignals(False)
                self.lbl_gif.setText(f"{len(self._sr_frames_interp)} frames @ {fps} fps (RIFE 2×)")
                self._timer.start(1000 // fps)
            else:
                self._run_rife()
        else:
            # restore original playback
            fps = self._original_fps
            self.fps_spin.blockSignals(True)
            self.fps_spin.setValue(fps)
            self.fps_spin.blockSignals(False)
            self.lbl_gif.setText(f"{len(self._nn_frames)} frames @ {fps} fps")
            self._timer.start(1000 // fps)

    def _run_rife(self):
        self._timer.stop()
        self.progress.setVisible(True)
        self.progress.setFormat("RIFE %v/%m")
        self.lbl_gif.setText("Running RIFE…")
        self._rife_worker = RifeWorker(self._sr_frames, self._nn_frames, self.device)
        self._rife_worker.progress.connect(self._on_rife_progress)
        self._rife_worker.finished.connect(self._on_rife_done)
        self._rife_worker.error.connect(self._on_rife_error)
        self._rife_worker.start()

    def _on_rife_progress(self, current: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_rife_done(self, sr_interp: list, nn_sync: list):
        self._sr_frames_interp = sr_interp
        self._nn_frames_interp = nn_sync
        self._frame_idx = 0
        self.progress.setVisible(False)
        if self.chk_interpolate.isChecked():
            fps = self._original_fps * 2
            self.fps_spin.blockSignals(True)
            self.fps_spin.setValue(fps)
            self.fps_spin.blockSignals(False)
            self.lbl_gif.setText(f"{len(sr_interp)} frames @ {fps} fps (RIFE 2×)")
            self._timer.start(1000 // fps)
        else:
            # user unchecked while RIFE was running — stay on original
            self._timer.start(1000 // self._original_fps)

    def _on_rife_error(self, msg: str):
        self.lbl_gif.setText(f"RIFE error: {msg}")
        self.progress.setVisible(False)
        self.chk_interpolate.blockSignals(True)
        self.chk_interpolate.setChecked(False)
        self.chk_interpolate.blockSignals(False)
        self._timer.start(1000 // self._original_fps)

    def _next_frame(self):
        if self.chk_interpolate.isChecked() and self._sr_frames_interp:
            nn = self._nn_frames_interp
            sr = self._sr_frames_interp
        else:
            nn = self._nn_frames
            sr = self._sr_frames
        if not nn:
            return
        self.canvas_nn.show_frame(nn[self._frame_idx])
        self.canvas_sr.show_frame(sr[self._frame_idx])
        self._frame_idx = (self._frame_idx + 1) % len(nn)

    def _update_fps(self, fps: int):
        if self._timer.isActive():
            self._timer.start(1000 // fps)


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, default_checkpoint="", default_preset="default"):
        super().__init__()
        self.setWindowTitle("SR Compare")
        self.resize(1500, 900)
        self._image_paths: dict[str, Path] = {}

        self.btn_folder = QPushButton("Select Folder")
        self.btn_folder.clicked.connect(self.select_folder)
        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setStyleSheet("color:gray;font-size:10px;")

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self._on_image_selected)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tab1 = LRvsSRTab(device, default_preset, default_checkpoint)
        self.tab2 = AvsBTab(device)
        self.tab3 = GifTab(device)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.tab1, "LR vs SR")
        self.tabs.addTab(self.tab2, "Model A vs B")
        self.tabs.addTab(self.tab3, "GIF Temporal")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        left = QWidget()
        left.setFixedWidth(240)
        ll = QVBoxLayout(left)
        ll.addWidget(self.btn_folder)
        ll.addWidget(self.lbl_folder)
        ll.addWidget(self.image_list)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)

        central = QWidget()
        QHBoxLayout(central).addWidget(splitter)
        self.setCentralWidget(central)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        paths = {p.name: p for p in sorted(Path(folder).iterdir())
                 if p.suffix.lower() in EXTENSIONS}
        if not paths:
            self.lbl_folder.setText("No images found")
            return
        self._image_paths = paths
        self.lbl_folder.setText(f"{Path(folder).name}  ({len(paths)})")
        self.image_list.clear()
        for name in paths:
            self.image_list.addItem(QListWidgetItem(name))

    def _on_image_selected(self, current, _):
        if current is None:
            return
        path = self._image_paths.get(current.text())
        if path and hasattr(self.tabs.currentWidget(), 'process_image'):
            self.tabs.currentWidget().process_image(path)

    def _on_tab_changed(self, _):
        item = self.image_list.currentItem()
        if item and hasattr(self.tabs.currentWidget(), 'process_image'):
            self._on_image_selected(item, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--preset", default="default")
    args = parser.parse_args()
    checkpoint = args.checkpoint or f"checkpoints/{args.preset}/sr_model_best.pth"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(default_checkpoint=checkpoint, default_preset=args.preset)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
