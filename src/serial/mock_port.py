import random
import statistics as stats
from collections import deque
from typing import Any, Callable, Generator


def default_stream() -> Generator[Any, None, None]:
    """
    Generate a stream of pseudo-random data sequences based on dynamic constraints.

    Each sequence is a comma-separated string of eight integers representing the
    env and raw output from the four EMG sensors. The string is formatted as:
    ENV0,RAW0,ENV1,RAW1,ENV2,RAW2,ENV3,RAW3

    Returns:
        Generator: A generator object yielding strings that encapsulate two sets
        of computed integer values (`b_emit` and `c`) for each of four channels.

    Raises:
        N/A
    """
    rng = random.Random()

    # --- fixed settings ---
    MOMENTUM = 0.90  # base 'a' friction
    VEL_JITTER_MAX = 8.0  # base 'a' velocity jitter per step
    A_ENVELOPE = 64  # allow 'a' ∈ [-128, 1151]

    B_DIV = [512, 256, 128, 64]  # b uniform range half-width around 'a'
    B_HOLD = 50  # delay (samples) applied to b *only* for output
    MEDIAN_LEN = 100  # c = median of last MEDIAN_LEN generated b's

    # --- state ---
    # base = [rng.uniform(200, 800) for _ in range(4)]
    # vel = [0.0 for _ in range(4)]

    base = [500, 500, 500, 500]
    vel = [0.0, 0.0, 0.0, 0.0]

    # delay buffer for *emitted* b (no maxlen so we can test len > B_HOLD)
    delay_b = [deque() for _ in range(4)]
    # history of *generated* b for c's median (per pair)
    hist_b = [deque(maxlen=MEDIAN_LEN) for _ in range(4)]

    # seed with an initial b so early outputs are defined
    for i in range(4):
        a0 = int(round(base[i]))
        lo = max(0, a0 - B_DIV[i])
        hi = min(1023, a0 + B_DIV[i])
        b0 = rng.randint(lo, hi)
        delay_b[i].append(b0)  # queue starts with 1 sample
        hist_b[i].append(b0)

    while True:
        # update base 'a' (smooth), constrain to ±A_ENVELOPE outside [0,1023]
        # rnd = rng.uniform(-VEL_JITTER_MAX, VEL_JITTER_MAX)

        for i in range(4):
            rnd = rng.uniform(-VEL_JITTER_MAX, VEL_JITTER_MAX)
            vel[i] = MOMENTUM * vel[i] + rnd
            base[i] = base[i] + vel[i]
            if base[i] < -A_ENVELOPE:
                base[i] = -A_ENVELOPE
                vel[i] = abs(vel[i]) * 0.5
            elif base[i] > 1023 + A_ENVELOPE:
                base[i] = 1023 + A_ENVELOPE
                vel[i] = -abs(vel[i]) * 0.5

        out_vals = []
        for i in range(4):
            a = int(round(base[i]))
            lo = max(0, a - B_DIV[i])
            hi = min(1023, a + B_DIV[i])

            # generate b (pre-delay)
            b_gen = rng.randint(lo, hi)
            hist_b[i].append(b_gen)  # c is based on generated b (not delayed)
            c = int(round(stats.median(hist_b[i])))

            # apply true fixed delay to b:
            # only pop once we have > B_HOLD samples; until then, emit the oldest we have
            if B_HOLD > 0:
                delay_b[i].append(b_gen)
                if len(delay_b[i]) > B_HOLD:
                    b_emit = delay_b[i].popleft()  # now delayed by B_HOLD iterations
                else:
                    b_emit = delay_b[i][0]  # warm-up: no full delay yet
            else:
                b_emit = b_gen

            out_vals.extend((b_emit, c))

        yield ",".join(str(x) for x in out_vals) + "\n"


class MockPort:
    def __init__(
        self,
        stream_fn: Callable[[], Generator[Any, None, None]] = None,
        **kwargs,
    ) -> None:
        self._source = stream_fn() if stream_fn else default_stream()
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
            if not chunk.endswith(b"\n"):
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

    def write(self, data: bytes) -> int:
        # accept writes so higher layers work; return bytes "sent"
        return len(data)

    def flush(self) -> None:
        # nothing buffered for the mock
        pass

    # --- Context manager ---
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
