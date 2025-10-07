from src.serial.port_accessor import PortAccessor, PortEvent
from src.serial.serial_monitor import register_monitor

import time
import random
from typing import Any, Generator


def print_data(event: PortEvent):
    print(event.data)


def stream() -> Generator[Any, None, None]:
    rng = random.Random()

    values = [0] * 6

    while True:
        if rng.random() < 0.2:
            for i in range(6):
                if rng.random() < 0.1:
                    values[i] = rng.randint(0, 127)
        yield (",".join([str(v) for v in values]) + "\n").encode("utf-8")


if __name__ == "__main__":
    pa = PortAccessor(port="MOCK")
    pa.open()

    handle = register_monitor(pa, fs=1000, title="Throughput Monitor", plot_out=False)

    source = stream()

    for i in range(5000):
        payload = next(source)
        pa.write(payload)
        time.sleep(0.05)

    handle.stop()
    pa.close()
