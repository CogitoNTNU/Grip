from typing import Generator, Any, Callable


class MockPort:
    def __init__(
        self,
        stream_fn: Callable[[], Generator[Any, None, None]],
        **kwargs,
    ) -> None:
        self._source = stream_fn()
        self._buf = bytearray()

    @staticmethod
    def _byte_ify(x: Any) -> bytes:
        # Fast paths
        if isinstance(x, (bytes, bytearray, memoryview)):
            return bytes(x)
        if isinstance(x, str):
            return x.encode("utf-8")
        if isinstance(x, int):
            # If it's a single byte value, emit that byte; else encode its text
            return bytes([x % 256]) if 0 <= x <= 255 else str(x).encode("utf-8")
        # Fallback: text-encode (e.g., float -> "1.23")
        return str(x).encode("utf-8")

    def _refill(self) -> None:
        chunk = self._byte_ify(next(self._source))
        if chunk:
            self._buf.extend(chunk)
            self._buf.extend(b"\n")

    def read(self, size: int = 1) -> bytes:
        if size <= 0:
            return b""
        out = bytearray()
        while len(out) < size:
            if not self._buf:
                self._refill()
            need = size - len(out)
            take = self._buf[:need]
            out.extend(take)
            del self._buf[: len(take)]  # mutate the same bytearray
        return bytes(out)

    def read_until(self, expected: bytes = b"\n") -> bytes:
        if not expected:
            return self.read(1)
        out = bytearray()

        while (idx := self._buf.find(expected)) == -1:
            self._refill()

        end = idx + len(expected)
        out = bytes(self._buf[:end])
        del self._buf[:end]
        return out

    def readline(self) -> bytes:
        return self.read_until(b"\n")

    def close(self):
        pass

    # --- Context manager ---
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
