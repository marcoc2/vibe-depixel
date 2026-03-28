#!/usr/bin/env python3
"""Standalone training preview viewer. Launched by sr_train.py as a subprocess.

Usage:
    python preview_viewer.py --watch path/to/preview_latest.png --title "my experiment"
"""
import sys
import argparse
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QScrollArea
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QFileSystemWatcher, QTimer, Qt


class PreviewWindow(QMainWindow):
    def __init__(self, watch_path: str, title: str):
        super().__init__()
        self.watch_path = watch_path
        self._base_title = title
        self.setWindowTitle(title)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(self.label)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        self.resize(960, 360)

        self.watcher = QFileSystemWatcher()
        self.watcher.fileChanged.connect(self._on_changed)
        if Path(watch_path).exists():
            self.watcher.addPath(watch_path)
            self._reload()

        # Polling fallback — QFileSystemWatcher can miss atomic writes on Windows
        self._last_mtime = 0.0
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(2000)

    def _poll(self):
        p = Path(self.watch_path)
        if not p.exists():
            return
        mtime = p.stat().st_mtime
        if mtime != self._last_mtime:
            self._last_mtime = mtime
            self._reload()
        if self.watch_path not in self.watcher.files():
            self.watcher.addPath(self.watch_path)

    def _on_changed(self, path: str):
        self._reload()
        # Re-add: atomic overwrites remove the inode watch on Windows
        if path not in self.watcher.files():
            self.watcher.addPath(path)

    def _reload(self):
        p = Path(self.watch_path)
        if not p.exists():
            return
        px = QPixmap(str(p))
        if px.isNull():
            return
        self.label.setPixmap(px)
        self.label.resize(px.size())
        # Update title to show the file mtime as a rough epoch indicator
        self.setWindowTitle(f"{self._base_title}  —  {p.stat().st_mtime:.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training preview viewer")
    parser.add_argument("--watch", required=True, help="Path to preview_latest.png")
    parser.add_argument("--title", default="Training Preview")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    w = PreviewWindow(args.watch, args.title)
    w.show()
    sys.exit(app.exec())
