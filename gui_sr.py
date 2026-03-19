"""
Super-Resolution comparison GUI.
Select a folder, run the SR model on all images, click to compare with a before/after slider.

Usage:
    python gui_sr.py
    python gui_sr.py --checkpoint checkpoints/sr_model_best.pth
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QLabel, QProgressBar,
    QSplitter, QListWidgetItem,
)
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect

from core.sr_model import EDSRLite

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


OUTPUT_DIR = Path("output/sr")


class InferenceWorker(QThread):
    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(dict)  # {filename: (lr_array, sr_array)}

    def __init__(self, image_paths: list[Path], model: EDSRLite, device: torch.device):
        super().__init__()
        self.image_paths = image_paths
        self.model = model
        self.device = device

    def run(self):
        results = {}
        total = len(self.image_paths)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            for i, path in enumerate(self.image_paths):
                self.progress.emit(i, total, path.name)
                lr_img = Image.open(path).convert('RGB')
                lr_arr = np.array(lr_img)

                # Check for cached SR output
                sr_path = OUTPUT_DIR / path.name
                if sr_path.exists():
                    sr_arr = np.array(Image.open(sr_path).convert('RGB'))
                else:
                    tensor = torch.from_numpy(lr_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    tensor = tensor.to(self.device)
                    sr = self.model(tensor)
                    sr_arr = sr.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    sr_arr = (sr_arr * 255).astype(np.uint8)
                    Image.fromarray(sr_arr).save(sr_path)

                results[path.name] = (lr_arr, sr_arr)
        self.progress.emit(total, total, "Done")
        self.finished.emit(results)


class CompareWidget(QWidget):
    """Before/after image comparison with a vertical slider controlled by mouse."""

    def __init__(self):
        super().__init__()
        self.lr_pixmap: QPixmap | None = None
        self.sr_pixmap: QPixmap | None = None
        self.slider_pos = 0.5  # 0.0 = all LR, 1.0 = all SR
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

    def set_images(self, lr_array: np.ndarray, sr_array: np.ndarray):
        self.lr_pixmap = self._array_to_pixmap(lr_array)
        self.sr_pixmap = self._array_to_pixmap(sr_array)
        self.slider_pos = 0.5
        self.update()

    def clear_images(self):
        self.lr_pixmap = None
        self.sr_pixmap = None
        self.update()

    @staticmethod
    def _array_to_pixmap(arr: np.ndarray) -> QPixmap:
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        qimg = QImage(arr.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _display_rect(self) -> tuple[QRect, float]:
        """Compute the rect that fits the image centered in the widget, return (rect, scale)."""
        if self.sr_pixmap is None:
            return QRect(0, 0, self.width(), self.height()), 1.0
        img_w, img_h = self.sr_pixmap.width(), self.sr_pixmap.height()
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / img_w, widget_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        x = (widget_w - disp_w) // 2
        y = (widget_h - disp_h) // 2
        return QRect(x, y, disp_w, disp_h), scale

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self.sr_pixmap is None or self.lr_pixmap is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select an image to compare")
            painter.end()
            return

        rect, scale = self._display_rect()
        img_w = self.sr_pixmap.width()
        img_h = self.sr_pixmap.height()

        # Scale LR to same display size as SR (nearest neighbor to show pixels)
        lr_scaled = self.lr_pixmap.scaled(
            rect.width(), rect.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        sr_scaled = self.sr_pixmap.scaled(
            rect.width(), rect.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        split_x = int(rect.width() * self.slider_pos)

        # Draw LR (left side)
        lr_src = QRect(0, 0, split_x, rect.height())
        painter.drawPixmap(rect.x(), rect.y(), lr_scaled, 0, 0, split_x, rect.height())

        # Draw SR (right side)
        sr_src = QRect(split_x, 0, rect.width() - split_x, rect.height())
        painter.drawPixmap(
            rect.x() + split_x, rect.y(), sr_scaled,
            split_x, 0, rect.width() - split_x, rect.height(),
        )

        # Slider line
        line_x = rect.x() + split_x
        painter.setPen(Qt.GlobalColor.white)
        painter.drawLine(line_x, rect.y(), line_x, rect.y() + rect.height())

        # Labels
        painter.setPen(Qt.GlobalColor.white)
        if split_x > 40:
            painter.drawText(rect.x() + 8, rect.y() + 20, "LR (original)")
        if rect.width() - split_x > 40:
            painter.drawText(rect.x() + split_x + 8, rect.y() + 20, "SR (model)")

        painter.end()

    def mouseMoveEvent(self, event):
        if self.sr_pixmap is None:
            return
        rect, _ = self._display_rect()
        x = event.position().x() - rect.x()
        self.slider_pos = max(0.0, min(1.0, x / rect.width()))
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, checkpoint: str):
        super().__init__()
        self.setWindowTitle("SR Compare")
        self.resize(1200, 800)

        self.checkpoint = checkpoint
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: EDSRLite | None = None
        self.results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.worker: InferenceWorker | None = None

        # Widgets
        self.btn_folder = QPushButton("Select Folder")
        self.btn_folder.clicked.connect(self.select_folder)

        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setStyleSheet("color: gray;")

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)

        self.compare = CompareWidget()

        # Layout
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.btn_folder)
        left_layout.addWidget(self.lbl_folder)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.image_list)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.compare)
        splitter.setSizes([250, 950])

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.checkpoint):
            self.lbl_folder.setText(f"Checkpoint not found: {self.checkpoint}")
            self.btn_folder.setEnabled(False)
            return
        self.model = EDSRLite(scale=4).to(self.device)
        self.model.load_state_dict(
            torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        paths = sorted(
            p for p in Path(folder).iterdir()
            if p.suffix.lower() in EXTENSIONS
        )
        if not paths:
            self.lbl_folder.setText("No images found in folder")
            return

        self.lbl_folder.setText(f"{folder}  ({len(paths)} images)")
        self.image_list.clear()
        self.compare.clear_images()
        self.results.clear()

        self.progress.setVisible(True)
        self.progress.setMaximum(len(paths))
        self.progress.setValue(0)
        self.btn_folder.setEnabled(False)

        self.worker = InferenceWorker(paths, self.model, self.device)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, current: int, total: int, name: str):
        self.progress.setValue(current)
        self.progress.setFormat(f"{current}/{total}  {name}")

    def _on_finished(self, results: dict):
        self.results = results
        self.progress.setVisible(False)
        self.btn_folder.setEnabled(True)

        for name in sorted(results.keys()):
            self.image_list.addItem(QListWidgetItem(name))

        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)

    def on_image_selected(self, current: QListWidgetItem, previous):
        if current is None:
            return
        name = current.text()
        if name in self.results:
            lr, sr = self.results[name]
            self.compare.set_images(lr, sr)


def main():
    parser = argparse.ArgumentParser(description="SR comparison GUI")
    parser.add_argument(
        "--checkpoint", default="checkpoints/sr_model_best.pth",
        help="Path to SR model checkpoint",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(args.checkpoint)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
