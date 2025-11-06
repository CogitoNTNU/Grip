"""
Arduino data parser for Grip sensor system.

Parses Arduino serial output format: S4:raw,env;S3:raw,env;S2:raw,env;S1:raw,env
Converts to: [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
"""

from typing import Optional, List


def parse_arduino_format(data: bytes) -> Optional[List[int]]:
    """Parse Arduino sensor data format.

    Expected format: S4:raw,env;S3:raw,env;S2:raw,env;S1:raw,env

    Example: S4:512,489;S3:523,501;S2:534,512;S1:545,523

    Output order: [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
    Where env0,raw0 = Sensor 4, env1,raw1 = Sensor 3, etc.

    Args:
        data: Raw bytes from Arduino serial port

    Returns:
        List of 8 integers [env0, raw0, env1, raw1, ...] or None if parse fails
    """
    try:
        # Decode bytes to string and strip whitespace
        line = data.decode('utf-8', errors='ignore').strip()

        # Skip comment lines or empty lines
        if not line or line.startswith('#'):
            return None

        # Split by semicolon to get individual sensor readings
        # Format: S4:raw,env;S3:raw,env;S2:raw,env;S1:raw,env
        sensors = line.split(';')

        if len(sensors) != 4:
            return None

        # Parse each sensor (in order: S4, S3, S2, S1)
        result = []
        for sensor_str in sensors:
            # Split by colon: "S4:raw,env" -> ["S4", "raw,env"]
            parts = sensor_str.split(':')
            if len(parts) != 2:
                return None

            # Split values by comma: "raw,env" -> ["raw", "env"]
            values = parts[1].split(',')
            if len(values) != 2:
                return None

            raw = int(values[0])
            env = int(values[1])

            # Append as env, raw (in that order for compatibility)
            result.append(env)
            result.append(raw)

        # Should have exactly 8 values
        if len(result) != 8:
            return None

        return result

    except (ValueError, UnicodeDecodeError, AttributeError):
        return None


def parse_arduino_csv(data: bytes) -> Optional[List[str]]:
    """Parse simple CSV format (legacy).

    Expected format: env0,raw0,env1,raw1,env2,raw2,env3,raw3

    Args:
        data: Raw bytes from Arduino serial port

    Returns:
        List of string values or None if parse fails
    """
    try:
        line = data.decode('utf-8', errors='ignore').strip()

        if not line or line.startswith('#'):
            return None

        values = line.split(',')

        if len(values) != 8:
            return None

        return values

    except (UnicodeDecodeError, AttributeError):
        return None
