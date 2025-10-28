# Data Collection

Tools for collecting training data from sensors and computer vision.

## Installation

```bash
pip install -r requirements.txt
```

## Components

- `collectors/` - Data collection scripts
- `calibration/` - Hand tracking calibration system
- `vision/` - MediaPipe hand tracking
- `utils/` - Utilities (serial monitor, etc.)

## Usage

### Calibrate Hand Tracking

```bash
python vision/hand_tracking.py
```

Press 'C' to start calibration.

### Collect Integrated Data

```bash
python collectors/integrated_collector.py
```
