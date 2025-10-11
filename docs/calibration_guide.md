# Hand Tracking Calibration System

This system provides precise hand tracking with personalized calibration for each user and hand.

## Features

- **Per-finger calibration**: Each finger is calibrated individually for maximum accuracy
- **Hand-specific calibration**: Separate calibrations for left and right hands
- **Persistent storage**: Calibration data is saved to CSV files for reuse
- **Visual feedback**: Clear on-screen instructions guide you through calibration
- **Linear interpolation**: Smooth 0-1 values based on your actual range of motion

## How to Use

### 1. Start the Program

```bash
python -m src.hand_movement_YOLO
```

### 2. Keyboard Controls

- **C** - Start calibration process
- **S** - Save calibration to file (after completion)
- **L** - Load existing calibration from file
- **R** - Reset calibration
- **Q** - Quit program

### 3. Calibration Process

When you press **C**, the calibration workflow begins:

1. **For each finger (Thumb → Index → Middle → Ring → Pinky):**

   a. **EXTEND Phase** (3-4 seconds)

   - Fully extend/open the finger
   - Hold steady while data is collected
   - Screen shows GREEN color

   b. **FLEX Phase** (3-4 seconds)

   - Fully close/curl the finger
   - Hold steady while data is collected
   - Screen shows RED color

1. **After all 5 fingers:** Calibration is complete!

   - Press **S** to save the calibration
   - Calibration is saved to `vision_calibration/calibration_[hand]_hand.csv`

### 4. Using Calibration

Once calibrated:

- Finger values are automatically normalized from 0.0 (fully closed) to 1.0 (fully open)
- Values are based on YOUR actual range of motion
- Works accurately regardless of:
  - Distance from camera
  - Hand orientation
  - Camera angle

## File Structure

```
vision_calibration/
├── calibration_left_hand.csv    # Left hand calibration data
└── calibration_right_hand.csv   # Right hand calibration data
```

## Calibration File Format

CSV files contain:

- Timestamp
- Hand label (Left/Right)
- For each finger:
  - Finger ID (0-4)
  - Finger name
  - Extended value (raw distance ratio)
  - Flexed value (raw distance ratio)

## Tips for Best Results

1. **Position yourself properly**: Sit comfortably with good lighting
1. **Keep hand steady**: Minimize movement during data collection phases
1. **Full range of motion**: Extend fingers as much as possible, flex completely
1. **Recalibrate if needed**: If accuracy decreases, run calibration again
1. **Separate calibrations**: Calibrate each hand separately for best results

## Technical Details

### Raw Value Calculation

For each finger:

```
raw_value = (tip_to_pip_distance + tip_to_mcp_distance) / hand_size
```

Where:

- `tip_to_pip_distance`: Distance from fingertip to PIP joint
- `tip_to_mcp_distance`: Distance from fingertip to MCP joint (knuckle)
- `hand_size`: Distance from wrist to middle finger MCP (normalization factor)

### Linear Interpolation (Lerp)

```
normalized_value = (raw_value - flexed_value) / (extended_value - flexed_value)
```

This maps your personal range:

- `flexed_value` → 0.0 (closed)
- `extended_value` → 1.0 (open)

Values are clamped to [0.0, 1.0] range.

## Troubleshooting

**Problem**: Calibration values seem inaccurate

- **Solution**: Recalibrate with more deliberate movements
- Ensure you're fully extending and flexing each finger

**Problem**: Cannot save calibration

- **Solution**: Make sure the `vision_calibration` folder exists and is writable

**Problem**: Calibration file not loading

- **Solution**: Check that the file exists and matches the hand you're using

**Problem**: Values jumping around

- **Solution**: Ensure good lighting and keep hand within camera frame

## Advanced Usage

### Manual Calibration Adjustment

You can manually edit the CSV files to adjust calibration values:

```csv
timestamp,2025-10-09T15:30:00
hand,Right
finger_id,finger_name,extended_value,flexed_value
0,Thumb,0.850,0.320
1,Index,0.920,0.380
...
```

### Integration with Other Code

```python
from src.hand_movement_YOLO import HandDetector

# Create detector with calibration
detector = HandDetector(use_calibration=True)

# Load existing calibration
detector.loadCalibration("Right")

# Get normalized finger values
fingers = detector.fingersUp(handNo=0, normalize=True)
# Returns [0.0-1.0, 0.0-1.0, 0.0-1.0, 0.0-1.0, 0.0-1.0]
```

## Future Enhancements

- [ ] Multi-user calibration profiles
- [ ] Auto-calibration based on usage patterns
- [ ] Gesture-specific calibration
- [ ] Export/import calibration between devices
- [ ] Calibration quality metrics and validation
