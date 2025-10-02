from src.serial.mock_port import MockPort
from src.serial.port_accessor import PortAccessor


def open_port(name: str, **kwargs):
    if name == "MOCK":
        return MockPort(**kwargs)

    kwargs.pop("stream_fn", None)
    return PortAccessor(name, **kwargs)
