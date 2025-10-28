# Integrated Data Collection System

This module combines hand tracking with sensor data collection, providing a unified interface for collecting synchronized hand gesture and sensor data.

## Features

- **Calibration Phase**: Calibrate hand tracking before data collection
- **Synchronized Collection**: Collects both sensor data and finger tracking data simultaneously
- **Visual Feedback**: Real-time display of hand tracking, sensor values, and collection status
- **Modular Design**: Clean separation of concerns across multiple modules

## Usage

### Basic Usage

```python
from src.integrated_data_collector import collect_integrated_data

collect_integrated_data(
    port="MOCK",  # Serial port (use "MOCK" for testing)
    num_iterations=1000,  # Number of samples to collect
    sleep_time=0.05,  # Delay between samples (seconds)
)
```

### Running from Command Line

```bash
python -m src.integrated_data_collector
```

## Workflow

### Phase 1: Calibration

1. The system starts in calibration mode
1. Press 'C' to begin hand calibration
1. Follow on-screen instructions to calibrate each finger
1. Press 'Q' to skip calibration (or continue after calibration complete)

### Phase 2: Data Collection

1. Press 'SPACE' to start/pause data collection
1. Hand tracking and sensor data are displayed in real-time
1. Collection progress is shown at the bottom of the screen
1. Press 'Q' to stop and save data

## Output Format

Data is saved to `data/integrated_data_YYYYMMDD_HHMMSS.csv` with the following columns:

- `timestamp`: ISO format timestamp
- `iteration`: Sample number
- `env0-env3`: Envelope values from sensors
- `raw0-raw3`: Raw values from sensors
- `thumb_tip`: Thumb tip tracking value
- `thumb_base`: Thumb base tracking value
- `index`: Index finger tracking value
- `middle`: Middle finger tracking value
- `ring`: Ring finger tracking value
- `pinky`: Pinky finger tracking value
- `hand_label`: Detected hand (Left/Right/None)

## Module Structure

- `integrated_data_collector.py`: Main data collection logic
- `calibration_helpers.py`: Calibration workflow helpers
- `ui_utils.py`: UI rendering functions
- `hand_movement_YOLO.py`: Hand detection and tracking
- `calibration_workflow.py`: Calibration state machine
- `calibration_manager.py`: Calibration data management

## Controls

### Calibration Phase

- `C`: Start calibration
- `Q`: Skip/continue

### Collection Phase

- `SPACE`: Start/pause collection
- `Q`: Stop and save data
