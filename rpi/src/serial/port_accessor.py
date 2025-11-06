import atexit
import itertools
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Callable, Generator, Literal, Optional

import serial
from serial import SerialException
from .mock_port import MockPort

Direction = Literal["in", "out"]


@dataclass(frozen=True)
class PortEvent:
    """
    An event passed to subscribers.

    Attributes:
        t (float): The time of the event.
        direction (Direction): Whether the event is incoming or outgoing.
        data (bytes): The data associated with the event.
        port (str): The name of the port the event originated from.
    """

    t: float
    direction: Direction
    data: bytes
    port: str


@dataclass
class Subscription:
    """
    Represents a subscription to handle events either via a callback or a queue.

    Only one of callback or queue should be set.
    A callback should be used for trivial handling of events, while a queue is
    used to pipe events to separate threads or processes.

    Attributes:
        id (int): The unique identifier for the subscription.
        callback (Optional[Callable[[PortEvent], None]]): A callable object to handle
            events, if provided.
        queue (Optional[Queue]): A queue to store events, if provided.
    """

    id: int
    # Either callback is set, or queue is set
    callback: Optional[Callable[[PortEvent], None]] = None
    queue: Optional[Queue] = None


class PortAccessor:
    """
    PortAccessor provides an interface to manage the connection to a serial port.

    This class creates a background thread to read lines from the serial port and
    then publishes them to subscribers. It also provides a simple interface to write
    to the serial port and subscribe to inbound and outbound events.

    Attributes:
        open: Opens the serial port and starts the reader thread.
        close: Closes the serial port and stops the reader thread.
        subscribe: Subscribe to inbound and outbound events.
        unsubscribe: Unsubscribe from inbound and outbound events.
        writeline: Write a line to the serial port.
        write: Write data to the serial port.
    """

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
        """
        Initializes a PortAccessor.

        Args:
            port (str): The name of the serial port to connect to.
            baudrate (int): The baud rate to use for the serial connection.
            timeout (float): The timeout to use for the serial connection.
            retries (int): The number of retries to attempt when opening the port.
            backoff (float): The backoff factor to use when retrying.
            stream_fn (Optional[Callable[[], Generator[Any, None, None]]]): A function to use for mocking the serial stream.
        """
        self.port: str = port
        self.kwargs: dict[str, Any] = dict(kwargs) | {
            "baudrate": baudrate,
            "timeout": timeout,
        }  # Unify the **kwargs with defaults
        self._mock_stream_fn = stream_fn

        self._serial: Optional[serial.SerialBase] = None
        self._retries: int = max(1, retries)  # Bound retries to [1, inf)
        self._backoff: float = max(0.0, backoff)  # Bound backoff to non-negative
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
        """
        Opens the serial port and starts the reader thread.

        Tries to open the port multiple times with exponential backoff. Raises the last
        exception if all retries fail.
        """
        # Keep track of previous error for so we can raise at the end
        prev_err: Optional[BaseException] = None

        for i in range(self._retries):
            try:
                # TODO: Split this into a separate PortAccessor for mocking
                if self.port.upper() == "MOCK":
                    with self._lock:
                        self._serial = MockPort(
                            stream_fn=self._mock_stream_fn, **self.kwargs
                        )

                        if self._reader is None or not self._reader.is_alive():
                            self._start_reader()
                    return

                with self._lock:
                    # Fetch port from serial_for_url to ensure it's valid'
                    self._serial = serial.serial_for_url(self.port, **self.kwargs)

                    # Start reader if not already running
                    if self._reader is None or not self._reader.is_alive():
                        self._start_reader()
                    return
            except (SerialException, OSError) as e:
                prev_err = e
                time.sleep(self._backoff * 2**i)  # Exponential backoff

        # Raise the last error if all retries failed
        assert prev_err is not None
        raise prev_err

    def _start_reader(self) -> None:
        """
        Starts the reader thread for the port.
        """
        self._stop.clear()
        self._reader = threading.Thread(
            target=self._reader_loop, name=f"PortReader:{self.port}", daemon=True
        )
        self._reader.start()

    def close(self) -> None:
        """
        Closes the serial port and stops the reader thread.
        """
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
        """
        Close and reopen the serial port under lock.
        """
        with self._lock:
            try:
                self.close()
            except Exception:
                pass
            self.open()

    def _atexit_close(self) -> None:
        """
        Function to be registered as an atexit handler.
        """
        try:
            self.close()
        except Exception:
            pass

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        """
        Retry a function call with exponential backoff.

        Reopens the port if it fails.
        Raises the last exception if all retries fail.

        Args:
            fn (Callable[[], Any]): The function to be called.
        """
        # Keep track of previous error for so we can raise at the end
        prev_err: Optional[BaseException] = None
        for i in range(self._retries):
            # Try the function
            try:
                with self._lock:
                    return fn()
            except (SerialException, OSError) as e:
                prev_err = e

            # Reopen the port before trying again
            try:
                self._reopen_once()
            except Exception as e:
                prev_err = e

            time.sleep(self._backoff * 2**i)  # Exponential backoff

        # Raise the last error if all retries failed
        assert prev_err is not None
        raise prev_err

    # ---------- Context manager interface ----------
    def __enter__(self) -> "PortAccessor":
        _ = self.ser
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- Pub/sub interface ----------
    def subscribe(
        self,
        callback: Optional[Callable[[PortEvent], None]] = None,
        *,
        max_queue: int = 1000,
    ) -> Subscription:
        """
        Subscribe to inbound and outbound events.

        If callback is provided, it will be invoked for every event (in caller's thread for writes, and reader thread for reads).
        Otherwise a bounded Queue is created; use sub.queue.get() to consume events.
        Queue is non-blocking on publish and will drop the oldest item on overflow.

        Args:
            callback (Optional[Callable[[PortEvent], None]]): A callable object to handle
            the event, if provided.
            max_queue (int): The maximum size of the queue, if callback is not provided.
        """
        if callback is None:
            q: Queue = Queue(maxsize=max_queue)
            sub = Subscription(id=next(self._sub_ids), queue=q)
        else:
            sub = Subscription(id=next(self._sub_ids), callback=callback)

        # Store the subscription by subscription id
        with self._subs_lock:
            self._subs[sub.id] = sub
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        """
        Unsubscribe from inbound and outbound events.

        Args:
            sub (Subscription): The subscription that unsubscribes.
        """
        # Remove the subscription by subscription id
        with self._subs_lock:
            self._subs.pop(sub.id, None)

    def _publish(self, evt: PortEvent) -> None:
        """
        Publishes the given event to all subscribers.

        This method runs the subscribers' callbacks or appends the event to their
        queues, depending on which is provided.

        Args:
            evt (PortEvent): The event to be published to subscribers.

        Raises:
            Full: If the subscribers' queue is full and cannot accept the event.
            Empty: If the subscribers' queue is empty when attempting to drop the
                   oldest event.
        """
        # snapshot to avoid holding lock during callbacks
        with self._subs_lock:
            subs = list(self._subs.values())

        for s in subs:
            # Run callbacks or append to queues
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
        """
        Write a line to the serial port.

        Args:
            line (bytes): The line to be written.
            ensure_newline (bool): Whether to ensure the line ends with a newline.
        """
        data = line if not ensure_newline or line.endswith(b"\n") else line + b"\n"
        return self.write(data)

    def write(self, data: bytes) -> int:
        """
        Write data to the serial port.

        Args:
            data (bytes): The data to be written.
        """

        def do_write() -> int:
            n = self.ser.write(data)
            # flush might raise; keep under retry
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

        Continually reads lines from the serial port and publishes them to subscribers.
        """
        while not self._stop.is_set():
            try:
                # Reading outside the lock; pyserial is not guaranteed thread-safe,
                # but we only ever write under lock; read is single-threaded here.
                line = self.ser.readline()
                time.sleep(0.05)
            except (SerialException, OSError):
                # Try to open with backoff; _with_retry handles open
                try:
                    self._with_retry(self.open)  # triggers open
                except Exception:
                    # brief pause before trying again
                    time.sleep(min(0.5, self._backoff))
                continue

            if not line:  # timeout tick
                continue

            evt = PortEvent(t=time.time(), direction="in", data=line, port=self.port)
            self._publish(evt)
