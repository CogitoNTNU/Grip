from rpi.src.serial.port_accessor import PortAccessor, PortEvent
from data_collection.utils.serial_monitor import register_monitor

import time
import random
import argparse
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


def run_test(port: str):
    """
    Run the serial test with specified port.

    Args:
        port: The port to connect to (e.g., "MOCK", "/dev/ttyUSB0", "/dev/ttyAMA0")
    """
    # Open the port
    pa = PortAccessor(port=port)
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


def main():
    parser = argparse.ArgumentParser(description="Test serial communication and monitoring")

    port_group = parser.add_mutually_exclusive_group(required=True)
    port_group.add_argument("--mock", action="store_true", help="Use mock port for testing")
    port_group.add_argument("--port", type=str, help="Serial port to connect to (e.g., /dev/ttyUSB0, /dev/ttyAMA0)")

    args = parser.parse_args()

    port = "MOCK" if args.mock else args.port

    print(f"Running serial test on port: {port}")

    run_test(port=port)


if __name__ == "__main__":
    main()
