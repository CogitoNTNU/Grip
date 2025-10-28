# serial_monitor.py
# deps: PySide6, pyqtgraph, numpy

import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from rpi.src.serial.port_accessor import PortAccessor, Subscription


# ---------------- internal: parsers ----------------
def _parse_csv_int(line: bytes) -> Optional[List[int]]:
    """
    Parse a CSV-encoded int sequence.

    Args:
        line (bytes): The line to parse.
    """
    parts = line.decode("utf-8", "ignore").strip().split(",")
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


# ---------------- plotting config ----------------
_CH = 8
_SERVOS = 6
_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
_HIST = 300  # history window (samples)


# TODO: Rewrite to use event.t as x axis instead of sample index for consistency.
class SerialMonitorWidget(QtWidgets.QWidget):
    """
    Live monitor for a PortAccessor subscription.

    - 'in'  events: csv8 -> four paired plots (0–1, 2–3, 4–5, 6–7)
    - 'out' events: csv6 -> six servo plots (individual values)

    Sample clock is driven by 'in' frames only:
    - On each 'in' frame we append 8 channel values AND also append the *latest known*
      6 servo values, keeping x aligned across both plot groups.
    - On 'out' frames we only update the "last known" servo values; no x advance.
      Servo plots therefore "hold" their last graphed line until the next 'in'.
    """

    def __init__(
        self,
        *,
        pa: PortAccessor,
        sub: Subscription,
        parse_fn_in: Optional[Callable[[bytes], Optional[List[int]]]] = None,
        parse_fn_out: Optional[Callable[[bytes], Optional[List[int]]]] = None,
        plot_in: bool = True,
        plot_out: bool = True,
    ):
        """
        Initialize the widget.

        Args:
            pa (PortAccessor): The port accessor.
            sub (Subscription): The subscription to monitor.
            parse_fn_in (Optional[Callable[[bytes], Optional[List[int]]]]): Function to parse 'in' frames.
            parse_fn_out (Optional[Callable[[bytes], Optional[List[int]]]]): Function to parse 'out' frames.
            plot_in (bool): Whether to plot 'in' frames.
            plot_out (bool): Whether to plot 'out' frames.
        """
        super().__init__()
        self.setWindowTitle("Serial Monitor")
        self._pa = pa
        self._sub = sub

        # Fetch event queue from the subscription
        self._q: queue.Queue[Any] = sub.queue

        # Bind parse functions
        self._parse_in = parse_fn_in or _parse_csv_int
        self._parse_out = parse_fn_out or _parse_csv_int

        # Which plots to show?
        self._plot_in = plot_in
        self._plot_out = plot_out

        # Initialize GUI
        root = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("Waiting for data…")
        root.addWidget(self.status)
        pg.setConfigOptions(antialias=True)
        self.glw = pg.GraphicsLayoutWidget()
        root.addWidget(self.glw)

        # add a bit of spacing to avoid title/axis collisions
        lyt = self.glw.ci.layout
        lyt.setHorizontalSpacing(24)
        lyt.setVerticalSpacing(28)

        # ---------------- left column: 4 pair plots, each "3 units" tall ----------------
        if self._plot_in:
            # Create a sub-layout in the left column (col=0) to hold the pair plots.
            left = self.glw.addLayout(row=0, col=0)  # pairs
            # Give the left column equal stretch weight as the right (set individually per col).
            self.glw.ci.layout.setColumnStretchFactor(0, 1)

            PAIR_UNITS = 3  # row stretch weight for each pair plot row
            self.pair_plots: list[pg.PlotItem] = []
            # One curve handle per input channel (i and j will write into these slots).
            self.curves: list[Optional[pg.PlotDataItem]] = [None] * _CH

            for r, (i, j) in enumerate(_PAIRS):
                # Add a plot widget for the r-th pair in the left sub-layout.
                p = left.addPlot(row=r, col=0, title=f"Pair {r + 1}")

                # Move the title a bit up so it doesn't overlap the plot area.
                p.titleLabel.item.setPos(0, -12)

                # Show only horizontal grid lines (y), make them faint.
                p.showGrid(x=False, y=True, alpha=0.25)

                # Fix the y-axis range to 10-bit ADC values (0..1023) with no padding.
                p.setYRange(0, 1023, padding=0)
                p.enableAutoRange("y", False)

                # Also clamp panning/zooming so users can’t go outside [0, 1024].
                p.getViewBox().setLimits(yMin=0, yMax=1024)

                # Only the bottom-most plot gets an x-axis label.
                if r == len(_PAIRS) - 1:
                    p.setLabel("bottom", "Samples")

                # All pair plots get a y-axis label.
                p.setLabel("left", "Amplitude")

                # Show top/right frame lines but hide tick marks/values there for a clean frame.
                for edge in ["top", "right"]:
                    ax = p.getAxis(edge)
                    ax.setStyle(showValues=False)  # no numbers
                    ax.setTicks([])  # no tick marks
                    ax.setPen(pg.mkPen("w", width=0.5))  # thin white frame line
                    p.showAxis(edge, True)  # ensure the edge is visible

                # Add two overlaid curves: i (thin) and j (thick) to visually distinguish them.
                c_i = p.plot(pen=pg.mkPen(width=1))
                c_j = p.plot(pen=pg.mkPen(width=3))

                # Track plot and curve handles for later updates.
                self.pair_plots.append(p)
                self.curves[i] = c_i
                self.curves[j] = c_j

                # Make this row consume 3 "units" of vertical space (relative to others).
                left.layout.setRowStretchFactor(r, PAIR_UNITS)

        # ---------------- right column: 6 servo plots, each "2 units" tall ---------------
        if self._plot_out:
            # Create a sub-layout in the right column (col=1) to hold the servo plots.
            right = self.glw.addLayout(row=0, col=1)  # servos
            # Give the right column equal stretch weight as the left.
            self.glw.ci.layout.setColumnStretchFactor(1, 1)

            SERVO_UNITS = 2  # row stretch weight per servo plot row
            self.servo_plots: list[pg.PlotItem] = []
            self.servo_curves: list[pg.PlotDataItem] = []

            for s in range(_SERVOS):
                # Add one plot per servo (stacked vertically).
                p = right.addPlot(row=s, col=0, title=f"Servo {s + 1}")

                # Lift the title up a bit for visual clarity.
                p.titleLabel.item.setPos(0, -12)

                # Horizontal grid only; faint.
                p.showGrid(x=False, y=True, alpha=0.25)

                # Servo rotation values assumed 7-bit (0..127).
                p.setYRange(0, 127, padding=0)
                p.enableAutoRange("y", False)
                p.getViewBox().setLimits(yMin=0, yMax=128)

                # Label the y-axis for all servo plots.
                p.setLabel("left", "Rotation")

                # Show a clean frame on top/right without ticks/values.
                for edge in ["top", "right"]:
                    ax = p.getAxis(edge)
                    ax.setStyle(showValues=False)
                    ax.setTicks([])
                    ax.setPen(pg.mkPen("w", width=0.5))
                    p.showAxis(edge, True)

                # Only the bottom-most servo plot gets the x-axis label.
                if s == _SERVOS - 1:
                    p.setLabel("bottom", "Samples")

                # Each servo has a single curve (moderate thickness).
                c = p.plot(pen=pg.mkPen(width=2))

                # Track handles for updates.
                self.servo_plots.append(p)
                self.servo_curves.append(c)

                # Make this row consume 2 "units" of vertical space.
                right.layout.setRowStretchFactor(s, SERVO_UNITS)

        # --- data buffers ---
        self.buffers = [deque(maxlen=_HIST) for _ in range(_CH)]
        self.servo_buffers = [deque(maxlen=_HIST) for _ in range(_SERVOS)]

        # latest known servo values (for hold-last behavior)
        self._last_servo = [0] * _SERVOS
        self.sample_idx = 0  # advances ONLY on 'in' frames

        # QT stuff
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def _handle_in(self, payload: bytes) -> bool:
        """Process an 'in' (csv8) frame; append to channel buffers and mirror servos."""
        vals = self._parse_in(payload)
        if vals is None or len(vals) != _CH:
            return False

        for ch in range(_CH):
            self.buffers[ch].append(vals[ch])

        # mirror latest servo values at this same x to keep alignment
        for s in range(_SERVOS):
            self.servo_buffers[s].append(self._last_servo[s])

        # Advance the sample index only on 'in' frames (x advance)
        self.sample_idx += 1
        return True

    def _handle_out(self, payload: bytes) -> None:
        """Update the latest-known servo values from an 'out' (csv6) frame (no x advance)."""
        vals = self._parse_out(payload)
        if vals is None or len(vals) != _SERVOS:
            return
        self._last_servo = vals

    def _update_plots(self) -> None:
        # use the channel history length as canonical x size
        n = len(self.buffers[0])
        if n <= 1:
            return

        x = np.arange(self.sample_idx - n, self.sample_idx, dtype=float)

        # update channel (pairs) curves
        if self._plot_in:
            for ch in range(_CH):
                if self.curves[ch] is None:
                    continue
                y = np.fromiter(self.buffers[ch], dtype=float, count=n)
                self.curves[ch].setData(x, y)

        # update servo curves (same x, hold-last already enforced in buffers)
        if self._plot_out:
            for s in range(_SERVOS):
                y = np.fromiter(self.servo_buffers[s], dtype=float, count=n)
                self.servo_curves[s].setData(x, y)

        self.status.setText(f"Samples: {self.sample_idx}")

    def _tick(self) -> None:
        got_in = False

        # Drain a bunch of events quickly; fold many 'out' updates between 'in' frames
        for _ in range(1000):
            try:
                evt = self._q.get_nowait()
            except queue.Empty:
                break

            direction = getattr(evt, "direction", None)
            payload = getattr(evt, "data", b"")

            if direction == "in" and self._plot_in:
                if self._handle_in(payload):
                    got_in = True
            elif direction == "out" and self._plot_out:
                self._handle_out(payload)
            else:
                # Unknown direction; ignore
                continue

        # Only redraw if we actually appended an 'in' frame (x advanced)
        if got_in:
            self._update_plots()

    def closeEvent(self, e):
        try:
            # Unsubscribe when the window closes (existing behavior)
            self._pa.unsubscribe(self._sub)
        finally:
            super().closeEvent(e)


# ------------- Threaded runner (all Qt lives in a worker thread) -------------
@dataclass
class MonitorHandle:
    """
    Handle returned by register_monitor. Call .stop() to close the window and
    end the Qt event loop. You may also pa.unsubscribe(handle.sub) if desired.

    Attributes:
        thread: The thread running the GUI.
        sub: The subscription used by the GUI.
        _stop_evt: The stop event used to signal the GUI thread to exit.

    """

    thread: threading.Thread
    sub: Subscription
    _stop_evt: threading.Event

    def stop(self, join_timeout: float = 2.0) -> None:
        """Signal the GUI thread to exit and wait briefly for it to join."""
        self._stop_evt.set()
        if self.thread.is_alive():
            self.thread.join(timeout=join_timeout)


def register_monitor(
    pa: PortAccessor,
    *,
    parse_fn_in: Optional[Callable[[bytes], Optional[List[int]]]] = None,
    parse_fn_out: Optional[Callable[[bytes], Optional[List[int]]]] = None,
    fs: int = 1000,
    max_queue: int = 5000,
    title: str = "Serial Monitor",
    plot_in: bool = True,
    plot_out: bool = True,
) -> MonitorHandle:
    """
    Start the SerialMonitor in its own *thread* where all Qt objects live.
    Nothing Qt runs in your main thread. Only becomes active when called.

    Returns a MonitorHandle. When finished, optionally call:
        pa.unsubscribe(handle.sub)    # redundant: widget unsubscribes on close
        handle.stop()                 # signal the GUI thread to exit
    """
    stop_evt = threading.Event()
    started_evt = threading.Event()
    holder: dict[str, Optional[Subscription]] = {"sub": None}

    def gui_thread_main():
        # All Qt lives here:
        app = QtWidgets.QApplication([])

        # Subscribe from within the GUI thread so the widget owns the lifecycle.
        sub = pa.subscribe(max_queue=max_queue)
        holder["sub"] = sub
        started_evt.set()

        w = SerialMonitorWidget(
            pa=pa,
            sub=sub,
            parse_fn_in=parse_fn_in,
            parse_fn_out=parse_fn_out,
            #fs=fs,
            plot_in=plot_in,
            plot_out=plot_out,
        )
        w.setWindowTitle(title)
        w.resize(1100, 850)
        w.show()

        # Periodically check for stop signal from the parent thread.
        timer = QtCore.QTimer()
        timer.setInterval(50)

        def check_stop():
            if stop_evt.is_set():
                try:
                    w.close()  # triggers pa.unsubscribe(sub) in closeEvent
                finally:
                    app.quit()

        timer.timeout.connect(check_stop)
        timer.start()

        app.exec()

    t = threading.Thread(
        target=gui_thread_main, name="QtSerialMonitorThread", daemon=True
    )
    t.start()

    # Wait briefly for the GUI thread to create the subscription
    if not started_evt.wait(timeout=2.0):
        raise RuntimeError("Timed out starting serial monitor thread")

    sub_obj = holder["sub"]
    if sub_obj is None:
        raise RuntimeError("Serial monitor thread failed to initialize subscription")

    return MonitorHandle(thread=t, sub=sub_obj, _stop_evt=stop_evt)


# ------------- Optional: factory API preserved for parity -------------
def create_monitor_widget(
    pa: PortAccessor,
    *,
    parse_fn_in: Optional[Callable[[bytes], Optional[List[int]]]] = None,
    parse_fn_out: Optional[Callable[[bytes], Optional[List[int]]]] = None,
    fs: int = 1000,
    max_queue: int = 5000,
    plot_in: bool = True,
    plot_out: bool = True,
) -> QtWidgets.QWidget:
    """
    Creates and returns a SerialMonitorWidget. Caller must manage QApplication and call .show().
    """
    sub = pa.subscribe(max_queue=max_queue)
    w = SerialMonitorWidget(
        pa=pa,
        sub=sub,
        parse_fn_in=parse_fn_in,
        parse_fn_out=parse_fn_out,
        fs=fs,
        plot_in=plot_in,
        plot_out=plot_out,
    )
    w.resize(1100, 850)
    return w
