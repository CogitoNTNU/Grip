from src.serial.port_accessor import PortAccessor, PortEvent
from src.serial.serial_monitor import register_monitor

import time
import random
from typing import Any, Generator


def print_data(event: PortEvent):
    """
    Small callback that prints the data from the serial port.

    Args:
        event (PortEvent): The event that triggered this callback.
    """
    print(event.data)


def stream() -> Generator[Any, None, None]:
    """
    Function that simulates a stream of data.
    """

    rng = random.Random()

    values = [0] * 6

    while True:
        if rng.random() < 0.2:
            for i in range(6):
                if rng.random() < 0.1:
                    values[i] = rng.randint(0, 127)
        yield (",".join([str(v) for v in values]) + "\n").encode("utf-8")


if __name__ == "__main__":
    # Open the MOCK port
    pa = PortAccessor(port="MOCK")
    pa.open()

    # Register the monitor
    handle = register_monitor(pa, fs=1000, title="Throughput Monitor", plot_out=False)

    # Create a stream of data and write it to the port
    source = stream()
    for i in range(5000):
        payload = next(source)
        pa.write(payload)
        time.sleep(0.05)

    # Stop the monitor and close the port
    handle.stop()
    pa.close()
