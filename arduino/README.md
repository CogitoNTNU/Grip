# Arduino

Arduino firmware for Grip sensor data collection.

## Setup for Data Collection

1. Install Arduino IDE
2. Open `sensor_reader/sensor_reader.ino`
3. Set baud rate to **115200**
4. Upload to Arduino Uno/Nano
5. Connect sensors to analog pins

## Pin Configuration

| Arduino Pin | Signal Type | Sensor | Description |
|-------------|-------------|--------|-------------|
| **A0** | Raw | Sensor 4 | Raw analog signal |
| **A1** | Processed | Sensor 4 | Filtered/envelope signal |
| **A2** | Raw | Sensor 3 | Raw analog signal |
| **A3** | Processed | Sensor 3 | Filtered/envelope signal |
| **A4** | Raw | Sensor 2 | Raw analog signal |
| **A5** | Processed | Sensor 2 | Filtered/envelope signal |
| **A6** | Raw | Sensor 1 | Raw analog signal |
| **A7** | Processed | Sensor 1 | Filtered/envelope signal |

## Serial Output Format

**Baud Rate:** 115200
**Format:** CSV (Comma-Separated Values)
**Order:** `env0,raw0,env1,raw1,env2,raw2,env3,raw3`

Where:
- `env0, raw0` = Sensor 4 (processed, raw)
- `env1, raw1` = Sensor 3 (processed, raw)
- `env2, raw2` = Sensor 2 (processed, raw)
- `env3, raw3` = Sensor 1 (processed, raw)

## Viewing Data in Arduino IDE

1. Upload `sensor_reader.ino` to Arduino
2. Open **Tools → Serial Monitor**
3. Set baud rate to **115200**
4. You'll see live sensor data streaming at ~100Hz

Example output:
```
# Grip Sensor System - 4 Sensors, 8 Channels
# Sensor Order: 4, 3, 2, 1
# Format: env0,raw0,env1,raw1,env2,raw2,env3,raw3
# Sampling at ~100Hz
#
512,489,523,501,534,512,545,523
511,490,522,500,535,513,544,522
...
```

## Contents

- `sensor_reader/` - **Main data collection sketch (USE THIS)**
- `sensor_pwm/` - PWM control sketches
- `examples/` - Example sketches
