import atexit
import itertools
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Callable, Generator, Literal, Optional

import serial
from serial import SerialException

Direction = Literal["in", "out"]


@dataclass(frozen=True)
class PortEvent:
    t: float
    direction: Direction
    data: bytes
    port: str


@dataclass
class Subscription:
    id: int
    # Either callback is set, or queue is set
    callback: Optional[Callable[[PortEvent], None]] = None
    queue: Optional[Queue] = None


class PortAccessor:
    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 9600,
        timeout: float = 0.05,
        retries: int = 3,
        backoff: float = 0.1,
        stream_fn: Optional[Callable[[], Generator[Any, None, None]]] = None,
        **kwargs: Any,
    ) -> None:
        self.port: str = port
        self.kwargs: dict[str, Any] = dict(kwargs) | {
            "baudrate": baudrate,
            "timeout": timeout,
        }
        self._mock_stream_fn = stream_fn

        self._serial: Optional[serial.SerialBase] = None
        self._retries: int = max(1, retries)
        self._backoff: float = max(0.0, backoff)
        self._lock = threading.RLock()

        # pub/sub state
        self._subs_lock = threading.RLock()
        self._subs: dict[int, Subscription] = {}
        self._sub_ids = itertools.count(1)
        self._pub_lock = threading.RLock()

        # reader thread
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None

        atexit.register(self._atexit_close)

    @property
    def ser(self) -> serial.SerialBase:
        with self._lock:
            if self._serial is None:
                self.open()
            return self._serial

    def open(self) -> None:
        prev_err: Optional[BaseException] = None

        for i in range(self._retries):
            try:
                if self.port.upper() == "MOCK":
                    from src.serial.mock_port import MockPort  # adjust if needed

                    with self._lock:
                        self._serial = MockPort(
                            stream_fn=self._mock_stream_fn, **self.kwargs
                        )

                        if self._reader is None or not self._reader.is_alive():
                            self._start_reader()
                    return

                # Real serial: strip testing-only args just in case

                with self._lock:
                    self._serial = serial.serial_for_url(self.port, **self.kwargs)

                    if self._reader is None or not self._reader.is_alive():
                        self._start_reader()
                    return
            except (SerialException, OSError) as e:
                prev_err = e
                time.sleep(self._backoff * 2**i)
        assert prev_err is not None
        raise prev_err

    def _start_reader(self) -> None:
        self._stop.clear()
        self._reader = threading.Thread(
            target=self._reader_loop, name=f"PortReader:{self.port}", daemon=True
        )
        self._reader.start()

    def close(self) -> None:
        self._stop.set()
        reader = self._reader
        if reader and reader.is_alive():
            reader.join(timeout=1.5)
        self._reader = None

        with self._lock:
            if self._serial:
                try:
                    self._serial.close()
                finally:
                    self._serial = None

    def _reopen_once(self) -> None:
        # runs under _with_retry
        with self._lock:
            self.close()
            self.open()

    def _atexit_close(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        prev_err: Optional[BaseException] = None
        for i in range(self._retries):
            try:
                with self._lock:
                    return fn()
            except (SerialException, OSError) as e:
                prev_err = e

            try:
                self.close()
            except Exception:
                pass
            try:
                self.open()
            except Exception as e2:
                prev_err = e2

            time.sleep(self._backoff * 2**i)

        assert prev_err is not None
        raise prev_err

    def __enter__(self) -> "PortAccessor":
        _ = self.ser
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def subscribe(
        self,
        callback: Optional[Callable[[PortEvent], None]] = None,
        *,
        max_queue: int = 1000,
    ) -> Subscription:
        """
        Subscribe to inbound and outbound events.

        - If callback is provided, it will be invoked for every event (in caller's thread for writes, and reader thread for reads).
        - Otherwise a bounded Queue is created; use sub.queue.get() to consume events.
          Queue is non-blocking on publish and will drop the oldest item on overflow.
        """
        if callback is None:
            q: Queue = Queue(maxsize=max_queue)
            sub = Subscription(id=next(self._sub_ids), queue=q)
        else:
            sub = Subscription(id=next(self._sub_ids), callback=callback)

        with self._subs_lock:
            self._subs[sub.id] = sub
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._subs_lock:
            self._subs.pop(sub.id, None)

    def _publish(self, evt: PortEvent) -> None:
        # snapshot to avoid holding lock during callbacks
        with self._subs_lock:
            subs = list(self._subs.values())
        for s in subs:
            if s.callback is not None:
                try:
                    s.callback(evt)
                except Exception:
                    # Don't let a misbehaving subscriber kill the bus
                    pass
            elif s.queue is not None:
                q = s.queue
                try:
                    q.put_nowait(evt)
                except Full:
                    with self._pub_lock:
                        try:
                            _ = q.get_nowait()  # drop oldest
                        except Empty:
                            pass
                        q.put_nowait(evt)

    # ---------- I/O surface ----------

    def writeline(self, line: bytes, *, ensure_newline: bool = True) -> int:
        data = line if not ensure_newline or line.endswith(b"\n") else line + b"\n"
        return self.write(data)

    def write(self, data: bytes) -> int:
        def do_write() -> int:
            n = self.ser.write(data)
            # flush might raise too; keep under retry
            try:
                self.ser.flush()
            finally:
                pass
            return n

        n = self._with_retry(do_write)
        # Publish only after a successful write
        self._publish(
            PortEvent(t=time.time(), direction="out", data=data, port=self.port)
        )
        return n

    def _reader_loop(self) -> None:
        """
        Background loop that reads lines and publishes them.
        Uses ser.timeout to wake regularly and check for stop signal.
        """
        while not self._stop.is_set():
            try:
                # Reading outside the lock; pyserial is not guaranteed thread-safe,
                # but we only ever write under lock; read is single-threaded here.
                line = self.ser.readline()
                time.sleep(0.05)
            except (SerialException, OSError):
                # Try to reopen with backoff; _with_retry handles reopen
                try:
                    self._with_retry(self._reopen_once)  # triggers reopen logic
                except Exception:
                    # brief pause before trying again
                    time.sleep(min(0.5, self._backoff))
                continue

            if not line:  # timeout tick
                continue

            evt = PortEvent(t=time.time(), direction="in", data=line, port=self.port)
            self._publish(evt)
