#!/usr/bin/env python3
"""
ComfyUI Queue Manager GUI
Visualize jobs before sending, remove duplicates manually.
"""
import json
import sys
import uuid
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict

from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QLabel, QPushButton, QSplitter,
    QStatusBar, QMessageBox, QProgressBar,
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal

COMFY_URL = "http://127.0.0.1:8188"
BACKUP_DIR = Path("F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/output/dataset/upscale_backup")
THUMB_SIZE = 80
PREVIEW_MAX = 480


# ── helpers ──────────────────────────────────────────────────────────────────

def pil_to_qpixmap(img: Image.Image, max_size: int) -> QPixmap:
    img = img.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qi = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


def collect_jobs(backup_dir: Path) -> list[dict]:
    jobs = []
    for folder in sorted(backup_dir.rglob("*")):
        if not folder.is_dir() or folder.name == "lr":
            continue
        pngs = sorted(
            [p for p in folder.glob("*.png") if not p.name.startswith(".")],
            key=lambda p: p.stat().st_mtime,
        )
        if not pngs:
            continue
        by_size = defaultdict(list)
        for png in pngs:
            try:
                img = Image.open(png)
                if "prompt" not in img.info:
                    continue
                by_size[img.size].append(png)
            except Exception:
                continue
        for size, files in by_size.items():
            rep = min(files, key=lambda p: p.stat().st_mtime)
            try:
                p = json.loads(Image.open(rep).info["prompt"])
                gif = p.get("329", {}).get("inputs", {}).get("gif", "?")
            except Exception:
                gif = "?"
            jobs.append({
                "path": rep,
                "size": size,
                "gif": gif,
                "rel": rep.relative_to(backup_dir),
                "prompt": None,  # lazy load
            })
    jobs.sort(key=lambda j: j["path"].stat().st_mtime)
    return jobs


def queue_prompt(prompt: dict, client_id: str) -> str | None:
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read()).get("prompt_id")
    except Exception as e:
        return None


def get_queue_size() -> int:
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/queue") as r:
            d = json.loads(r.read())
            return len(d.get("queue_running", [])) + len(d.get("queue_pending", []))
    except Exception:
        return -1


# ── sender thread ─────────────────────────────────────────────────────────────

class SenderThread(QThread):
    progress = pyqtSignal(int, int, str)   # current, total, label
    finished = pyqtSignal(int, int)         # queued, errors

    def __init__(self, jobs: list[dict]):
        super().__init__()
        self.jobs = jobs
        self.client_id = str(uuid.uuid4())

    def run(self):
        queued = errors = 0
        total = len(self.jobs)
        for i, job in enumerate(self.jobs, 1):
            # throttle
            while True:
                qs = get_queue_size()
                if qs == -1 or qs < 4:
                    break
                time.sleep(3)

            try:
                img = Image.open(job["path"])
                prompt = json.loads(img.info["prompt"])
            except Exception:
                errors += 1
                self.progress.emit(i, total, f"[error] {job['rel']}")
                continue

            pid = queue_prompt(prompt, self.client_id)
            if pid:
                queued += 1
                self.progress.emit(i, total, f"{job['rel']}")
            else:
                errors += 1
                self.progress.emit(i, total, f"[error] {job['rel']}")
            time.sleep(0.3)

        self.finished.emit(queued, errors)


# ── main window ───────────────────────────────────────────────────────────────

class QueueManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ComfyUI Queue Manager")
        self.resize(1100, 700)

        self.jobs: list[dict] = []
        self._sender: SenderThread | None = None

        # ── layout ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: job list
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_widget.currentItemChanged.connect(self._on_select)
        lv.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.btn_remove = QPushButton("Remove  [Del]")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_queue_sel = QPushButton("Queue selected")
        self.btn_queue_sel.setStyleSheet("font-weight:bold; background:#2a4e6e; color:white;")
        self.btn_queue_sel.clicked.connect(self._queue_selected)
        self.btn_queue = QPushButton("Queue all →")
        self.btn_queue.setStyleSheet("font-weight:bold; background:#2a6e2a; color:white;")
        self.btn_queue.clicked.connect(self._queue_all)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_queue_sel)
        btn_row.addWidget(self.btn_queue)
        lv.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        splitter.addWidget(left)

        # Right: preview + info
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 8, 8, 8)

        self.lbl_preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumSize(200, 200)
        self.lbl_preview.setStyleSheet("background:#1a1a1a;")
        rv.addWidget(self.lbl_preview, stretch=1)

        self.lbl_info = QLabel()
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#aaa; font-size:11px;")
        rv.addWidget(self.lbl_info)

        splitter.addWidget(right)
        splitter.setSizes([600, 500])

        self.setCentralWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._load_jobs()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self._remove_selected()
        super().keyPressEvent(event)

    def _load_jobs(self):
        self.status.showMessage("Loading jobs…")
        QApplication.processEvents()
        self.jobs = collect_jobs(BACKUP_DIR)
        self.list_widget.clear()
        for job in self.jobs:
            item = QListWidgetItem()
            item.setText(f"{job['rel']}\n{job['size'][0]}×{job['size'][1]}  gif: {job['gif']}")
            # thumbnail
            try:
                thumb = pil_to_qpixmap(Image.open(job["path"]), THUMB_SIZE)
                item.setIcon(QIcon(thumb))
            except Exception:
                pass
            item.setData(Qt.ItemDataRole.UserRole, id(job))
            self.list_widget.addItem(item)
        self._update_status()

    def _job_for_row(self, row: int) -> dict:
        return self.jobs[row]

    def _on_select(self, current: QListWidgetItem, _prev):
        if current is None:
            return
        row = self.list_widget.row(current)
        job = self._job_for_row(row)
        try:
            img = Image.open(job["path"])
            px = pil_to_qpixmap(img, PREVIEW_MAX)
            self.lbl_preview.setPixmap(px)
        except Exception:
            self.lbl_preview.setText("(preview error)")
        self.lbl_info.setText(
            f"<b>{job['rel']}</b><br>"
            f"Size: {job['size'][0]}×{job['size'][1]}<br>"
            f"GIF: {job['gif']}<br>"
            f"Path: {job['path']}"
        )

    def _remove_selected(self):
        rows = sorted(
            [self.list_widget.row(i) for i in self.list_widget.selectedItems()],
            reverse=True,
        )
        for row in rows:
            self.list_widget.takeItem(row)
            self.jobs.pop(row)
        self._update_status()

    def _update_status(self):
        self.status.showMessage(f"{len(self.jobs)} jobs in queue")

    def _queue_selected(self):
        rows = sorted({self.list_widget.row(i) for i in self.list_widget.selectedItems()})
        jobs = [self.jobs[r] for r in rows]
        if not jobs:
            QMessageBox.information(self, "Queue", "Nothing selected.")
            return
        self._start_sender(jobs, label=f"{len(jobs)} selected job(s)")

    def _queue_all(self):
        if not self.jobs:
            QMessageBox.information(self, "Queue", "No jobs to queue.")
            return
        self._start_sender(list(self.jobs), label=f"all {len(self.jobs)} jobs")

    def _start_sender(self, jobs: list[dict], label: str):
        reply = QMessageBox.question(
            self, "Confirm",
            f"Send {label} to ComfyUI?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_queue.setEnabled(False)
        self.btn_queue_sel.setEnabled(False)
        self.btn_remove.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)

        self._sender = SenderThread(jobs)
        self._sender.progress.connect(self._on_progress)
        self._sender.finished.connect(self._on_finished)
        self._sender.start()

    def _on_progress(self, current: int, total: int, label: str):
        self.progress.setValue(current)
        self.status.showMessage(f"[{current}/{total}] {label}")

    def _on_finished(self, queued: int, errors: int):
        self.btn_queue.setEnabled(True)
        self.btn_queue_sel.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self.progress.setVisible(False)
        self.status.showMessage(f"Done — queued: {queued}  errors: {errors}")
        QMessageBox.information(self, "Done", f"Queued: {queued}\nErrors: {errors}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = QueueManager()
    w.show()
    sys.exit(app.exec())
