import sys
import os
import torch
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QColorDialog, QFrame, QSplitter, QFileDialog, QComboBox
)
from PyQt6.QtGui import QImage, QPainter, QPixmap, QColor, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint

from core.sr_model import load_model, PRESETS, SPANDREL_PRESETS

class CanvasWidget(QWidget):
    """Low-resolution canvas for drawing pixels."""
    changed = pyqtSignal(QImage)

    def __init__(self, size=64):
        super().__init__()
        self.canvas_size = size
        self.image = QImage(size, size, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.color = QColor(Qt.GlobalColor.black)
        self.brush_size = 1
        self.last_point = QPoint()
        self.setMinimumSize(400, 400)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Draw image scaled to fit widget
        # Use FastTransformation (Nearest Neighbor) for that "pixel art" feel
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.drawImage(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = self._to_canvas_coords(event.position())
            self._draw_pixel(self.last_point)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            current_point = self._to_canvas_coords(event.position())
            self._draw_line(self.last_point, current_point)
            self.last_point = current_point

    def _to_canvas_coords(self, pos):
        x = int(pos.x() * self.canvas_size / self.width())
        y = int(pos.y() * self.canvas_size / self.height())
        return QPoint(x, y)

    def _draw_pixel(self, pt):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPoint(pt)
        painter.end()
        self.update()
        self.changed.emit(self.image)

    def _draw_line(self, p1, p2):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(p1, p2)
        painter.end()
        self.update()
        self.changed.emit(self.image)

    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()
        self.changed.emit(self.image)

class SRWorker(QThread):
    """Inference thread to avoid blocking UI."""
    finished = pyqtSignal(np.ndarray)

    def __init__(self, model, device, qimage):
        super().__init__()
        self.model = model
        self.device = device
        
        # Convert QImage to format we can use with torch
        tmp = qimage.convertToFormat(QImage.Format.Format_RGB888)
        width, height = tmp.width(), tmp.height()
        ptr = tmp.bits()
        ptr.setsize(height * width * 3)
        self.arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3)).copy()

    def run(self):
        t = torch.from_numpy(self.arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            out = self.model(t.to(self.device))
        sr_arr = (out.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        self.finished.emit(sr_arr)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Super-Resolution Paint")
        self.resize(1000, 600)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.worker = None
        self.sr_enabled = True
        
        self.latest_qimage = None
        self._init_ui()
        self._load_default_model()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Toolbar
        toolbar = QHBoxLayout()
        
        self.btn_color = QPushButton("Pick Color")
        self.btn_color.clicked.connect(self._pick_color)
        toolbar.addWidget(self.btn_color)
        
        toolbar.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS.keys()) + sorted(SPANDREL_PRESETS))
        self.combo_preset.setCurrentText("default")
        toolbar.addWidget(self.combo_preset)
        
        self.btn_ckpt = QPushButton("Load Checkpoint")
        self.btn_ckpt.clicked.connect(self._pick_checkpoint)
        toolbar.addWidget(self.btn_ckpt)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_canvas)
        toolbar.addWidget(self.btn_clear)
        
        self.chk_sr = QCheckBox("Real-time SR Filter")
        self.chk_sr.setChecked(True)
        self.chk_sr.toggled.connect(self._toggle_sr)
        toolbar.addWidget(self.chk_sr)
        
        self.lbl_status = QLabel("Loading model...")
        toolbar.addWidget(self.lbl_status)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Main area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Canvas
        left_box = QFrame()
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(QLabel("Artist Canvas (LR)"))
        self.canvas = CanvasWidget(64)
        self.canvas.changed.connect(self._on_canvas_changed)
        left_layout.addWidget(self.canvas)
        splitter.addWidget(left_box)
        
        # Right: SR Preview
        right_box = QFrame()
        right_layout = QVBoxLayout(right_box)
        right_layout.addWidget(QLabel("SR Filtered Output (HR)"))
        self.preview = QLabel()
        self.preview.setMinimumSize(400, 400)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: #222;")
        right_layout.addWidget(self.preview)
        splitter.addWidget(right_box)
        
        layout.addWidget(splitter)

    def _load_default_model(self):
        preset = "default"
        checkpoint = f"checkpoints/{preset}/sr_model_best_yugioh.pth"
        if not os.path.exists(checkpoint):
            checkpoint = f"checkpoints/{preset}/sr_model_best.pth"

        if os.path.exists(checkpoint):
            self._load_checkpoint(checkpoint)
        else:
            try:
                self.model = load_model(preset, self.device)
                self.model.eval()
                self.lbl_status.setText("Model loaded (default architecture)")
            except Exception as e:
                self.lbl_status.setText(f"Error: {e}")

    def _pick_checkpoint(self):
        initial_dir = "checkpoints"
        if not os.path.exists(initial_dir):
            os.makedirs(initial_dir, exist_ok=True)
            
        path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", initial_dir, "Model (*.pth)")
        if path:
            self._load_checkpoint(path)

    def _load_checkpoint(self, path):
        preset = self.combo_preset.currentText()
        try:
            # Re-initialize model architecture based on selected preset
            # SPANDREL models (esrgan, swinir) are loaded directly from path
            self.model = load_model(
                preset, 
                self.device, 
                pretrained_path=path if preset in SPANDREL_PRESETS else None
            )
            
            # For non-spandrel models, we load state_dict separately
            if preset not in SPANDREL_PRESETS:
                sd = torch.load(path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(sd)
            
            self.model.eval()
            self.lbl_status.setText(f"Loaded [{preset}]: {Path(path).name}")
            self._on_canvas_changed(self.canvas.image)
        except Exception as e:
            self.lbl_status.setText(f"Error: {e}")
            self.lbl_status.setToolTip(str(e))

    def _pick_color(self):
        color = QColorDialog.getColor(self.canvas.color, self)
        if color.isValid():
            self.canvas.color = color
            self.btn_color.setStyleSheet(f"background-color: {color.name()};")

    def _clear_canvas(self):
        self.canvas.clear()

    def _toggle_sr(self, enabled):
        self.sr_enabled = enabled
        if not enabled:
            self.preview.clear()
        else:
            self._on_canvas_changed(self.canvas.image)

    def _on_canvas_changed(self, qimage):
        self.latest_qimage = qimage.copy()
        self._trigger_inference()

    def _trigger_inference(self):
        if not self.sr_enabled or self.model is None:
            return
        
        if self.worker and self.worker.isRunning():
            return
            
        if self.latest_qimage is None:
            return

        qimg = self.latest_qimage
        self.latest_qimage = None # Mark as consumed
        
        self.worker = SRWorker(self.model, self.device, qimg)
        self.worker.finished.connect(self._update_preview)
        self.worker.finished.connect(self._trigger_inference) # Recurse once done
        self.worker.start()

    def _update_preview(self, sr_arr):
        h, w, ch = sr_arr.shape
        qimg = QImage(sr_arr.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit the preview label
        scaled_pixmap = pixmap.scaled(
            self.preview.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
