from rpi.src.serial.port_accessor import PortAccessor, PortEvent
from data_collection.utils.serial_monitor import register_monitor

import random
import csv
from pathlib import Path
from datetime import datetime
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


def create_data_directory() -> Path:
    """Create and return the data directory path."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir


def create_csv_file(data_dir: Path) -> Path:
    """Create a timestamped CSV file in the data directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = data_dir / f"data_collection_{timestamp}.csv"
    return csv_file


def write_csv_header(writer: csv.writer) -> None:
    """Write the CSV header row."""
    header = [
        "timestamp",
        "iteration",
        "env0",
        "raw0",
        "env1",
        "raw1",
        "env2",
        "raw2",
        "env3",
        "raw3",
    ]
    writer.writerow(header)


def parse_payload(payload: bytes) -> list[str]:
    """Parse the payload bytes and return a list of values."""
    data_str = payload.decode("utf-8").strip()
    return data_str.split(",")


def parse_port_event(event: PortEvent) -> list[str]:
    """Parse the PortEvent data and return a list of values."""
    data_str = event.data.decode("utf-8").strip()
    return data_str.split(",")


def collect_data(
    port: str = "MOCK", num_iterations: int = 5000, sleep_time: float = 0.05
) -> None:
    """
    Collect data from the port and write it to a CSV file.

    Args:
        port: The port to connect to (default: "MOCK")
        num_iterations: Number of data points to collect (default: 5000)
        sleep_time: Time to sleep between iterations in seconds (default: 0.05)
    """
    data_dir = create_data_directory()
    csv_file = create_csv_file(data_dir)

    pa: PortAccessor = PortAccessor(port=port)
    pa.open()

    subscription = pa.subscribe(max_queue=100)

    handle = register_monitor(pa, fs=1000, title="Throughput Monitor", plot_out=False)

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        write_csv_header(writer)

        for i in range(num_iterations):
            payload = subscription.queue.get()

            # Parse and write to CSV
            values = parse_port_event(payload)
            row = [datetime.now().isoformat(), i] + values
            writer.writerow(row)

            # time.sleep(sleep_time)

    handle.stop()
    pa.close()

    print(f"Data saved to {csv_file}")


if __name__ == "__main__":
    collect_data()
