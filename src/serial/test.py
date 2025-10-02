import time
from typing import Generator, Any
import random
from src.serial.ports import open_port
from src.serial.serial_monitor import serial_monitor

from collections import deque
import statistics as stats


def stream_fn() -> Generator[Any, None, None]:
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

        yield ",".join(str(x) for x in out_vals)


if __name__ == "__main__":
    # port = serial_monitor(open_port("/dev/ttyUSB0", baudrate=9600, timeout=0.1))
    port = serial_monitor(open_port("MOCK", stream_fn=stream_fn))
    
    print("Serial monitor started. Press Ctrl+C to quit.")

    while True:
        line = port.readline()
        print(line)
        time.sleep(0.005)
