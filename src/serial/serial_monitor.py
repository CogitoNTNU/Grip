# serial_monitor.py
# deps: PySide6, pyqtgraph, numpy

import queue
import threading
import time
from collections import deque
from typing import Any, Callable, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


# ---------------- internal: parse 8-int CSV ----------------
def _parse_csv8(line: bytes) -> Optional[List[int]]:
    parts = line.decode("utf-8", "ignore").strip().split(",")
    if len(parts) != 8:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


# ---------------- internal: broadcast hub -------------------
# Subscribers get (timestamp, raw_line: bytes, parsed_values: Optional[List[int]])
_subscribers: List[Callable[[float, bytes, Optional[List[int]]], None]] = []


def _subscribe(cb: Callable[[float, bytes, Optional[List[int]]], None]) -> None:
    _subscribers.append(cb)


def _unsubscribe(cb: Callable[[float, bytes, Optional[List[int]]], None]) -> None:
    try:
        _subscribers.remove(cb)
    except ValueError:
        pass


def _broadcast(ts: float, raw: bytes, parsed: Optional[List[int]]) -> None:
    for cb in list(_subscribers):
        try:
            cb(ts, raw, parsed)
        except Exception:
            pass  # never let a subscriber crash the sender


# ---------------- internal: serial wrapper (tee) ------------
class _SerialTee:
    """Wraps a serial-like object; tees .readline() to internal broadcast."""

    def __init__(
        self,
        ser: Any,
        parse_fn: Optional[Callable[[bytes], Optional[List[int]]]] = None,
    ):
        self._ser = ser
        self._parse = parse_fn or _parse_csv8

    def readline(self, *a, **kw) -> bytes:
        line = self._ser.readline(*a, **kw)
        if line:
            try:
                parsed = self._parse(line)
            except Exception:
                parsed = None
            _broadcast(time.time(), line, parsed)
        return line

    def read(self, *a, **kw) -> bytes:
        return self._ser.read(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._ser, name)

    def __enter__(self):
        self._ser.__enter__()
        return self

    def __exit__(self, *a):
        return self._ser.__exit__(*a)


# ---------------- internal: minimal plotting UI -------------
_CH = 8
_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
_HIST = 300


class _SerialMonitorWidget(QtWidgets.QWidget):
    """Neutral serial monitor: 4 graphs (one per pair), thicker pen = stabler channel."""

    def __init__(self, fs: int = 1000):
        super().__init__()
        self.setWindowTitle("Serial Monitor")
        self.fs = fs

        root = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("Waiting for data…")
        root.addWidget(self.status)

        pg.setConfigOptions(antialias=True)
        self.glw = pg.GraphicsLayoutWidget()
        root.addWidget(self.glw)

        # Create 4 plots, each with two curves (pair)
        self.plots: List[pg.PlotItem] = []
        self.curves = []  # will be indexed by channel 0..7
        for r, (i, j) in enumerate(_PAIRS):
            p = self.glw.addPlot(row=r, col=0, title=f"Pair {r + 1}")
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setYRange(0, 1023, padding=0)
            p.enableAutoRange("y", False)
            p.getViewBox().setLimits(yMin=0, yMax=1023)
            p.setLabel("bottom", "Samples")
            p.setLabel("left", "Amplitude")
            # create 2 curves
            c_i = p.plot(pen=pg.mkPen(width=1))
            c_j = p.plot(pen=pg.mkPen(width=4))
            self.plots.append(p)
            # ensure index alignment in self.curves list
            while len(self.curves) <= max(i, j):
                self.curves.append(None)
            self.curves[i] = c_i
            self.curves[j] = c_j

        # Buffers per channel
        self.buffers = [deque(maxlen=_HIST) for _ in range(_CH)]
        self.sample_idx = 0

        # queue for parsed lines
        self._q: "queue.Queue[List[int]]" = queue.Queue()

        # subscribe to broadcast
        _subscribe(self._on_broadcast)

        # repaint timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)  # ~33 fps

    def _on_broadcast(self, ts: float, raw: bytes, parsed: Optional[List[int]]) -> None:
        if parsed is None or len(parsed) != _CH:
            return
        self._q.put(parsed)

    def _tick(self):
        got = False
        for _ in range(500):  # drain a bunch per frame
            try:
                vals = self._q.get_nowait()
            except queue.Empty:
                break
            for ch in range(_CH):
                self.buffers[ch].append(vals[ch])
            self.sample_idx += 1
            got = True

        if not got:
            return

        n = len(self.buffers[0])
        if n <= 1:
            return

        x = np.arange(self.sample_idx - n, self.sample_idx, dtype=float)

        # Update curves
        for ch in range(_CH):
            y = np.fromiter(self.buffers[ch], dtype=float, count=n)
            self.curves[ch].setData(x, y)

        self.status.setText(f"Samples: {self.sample_idx}")

    def closeEvent(self, e):
        try:
            _unsubscribe(self._on_broadcast)
        finally:
            super().closeEvent(e)


# ---------------- internal: UI manager (bg thread) ----------
_ui_lock = threading.Lock()
_ui_started = False


def _ui_thread_main(fs: int):
    app = QtWidgets.QApplication([])
    w = _SerialMonitorWidget(fs=fs)
    w.resize(1100, 850)
    w.show()
    app.exec()


def _ensure_ui_started(fs: int):
    global _ui_started
    with _ui_lock:
        if _ui_started:
            return
        t = threading.Thread(target=_ui_thread_main, args=(fs,), daemon=True)
        t.start()
        _ui_started = True


# ---------------- public: single entrypoint ------------------
def serial_monitor(
    ser: Any,
    *,
    parse_fn: Optional[Callable[[bytes], Optional[List[int]]]] = None,
    fs: int = 1000,
    autostart: bool = True,
) -> Any:
    """
    Wrap a serial-like object so each .readline() is plotted automatically.
    Shows 4 graphs (pairs 0–1, 2–3, 4–5, 6–7); the stabler channel in each pair
    (lower short-term variance) is rendered with a thicker pen.

    - Assumes each line is 8 comma-separated ints (override via parse_fn).
    - If autostart=True, starts a background Qt window on first use.

    Usage:
        port = serial_monitor(open_port("MOCK", baudrate=115200, timeout=0.1))
        line = port.readline()  # also appears in the live plot
    """
    if autostart:
        _ensure_ui_started(fs=fs)
    return _SerialTee(ser, parse_fn=parse_fn)
