import atexit
import threading
import time
from typing import Any

import serial


class PortAccessor:
    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        timeout: float = 1.0,
        retries: int = 3,
        backoff: float = 0.1,
        **kwargs,
    ) -> None:
        self.port: str = port
        self.kwargs: dict[str, Any] = dict(baudrate=baudrate, timeout=timeout)

        self._serial: serial.Serial | None = None
        self._retries: int = retries
        self._backoff: float = backoff
        self._lock = threading.RLock()

        atexit.register(self.close)

    @property
    def ser(self) -> serial.Serial:
        with self._lock:
            if self._serial is None:
                self._open()
            return self._serial

    def _open(self) -> None:
        prev_err = None
        for i in range(self._retries):
            try:
                self._serial = serial.serial_for_url(self.port, **self.kwargs)
                return
            except Exception as e:
                prev_err = e
                time.sleep(self._backoff * 2**i)
        raise prev_err

    def close(self) -> None:
        with self._lock:
            if self._serial:
                try:
                    self._serial.close()
                finally:
                    self._serial = None

    # Transparent passthrough
    def __getattr__(self, name: str) -> Any:
        try:
            attr = getattr(self.ser, name)
        except (serial.SerialException, OSError):
            self.close()
            attr = getattr(self.ser, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            try:
                return getattr(self.ser, name)(*args, **kwargs)
            except (serial.SerialException, OSError):
                self.close()
                return getattr(self.ser, name)(*args, **kwargs)

        return wrapper
