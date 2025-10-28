# Calibration Quick Reference

## Quick Start

1. **Run the program:**

   ```bash
   python -m src.hand_movement_YOLO
   ```

1. **Show your hand to the camera**

1. **Press 'C' to start calibration**

1. **Follow on-screen instructions:**

   - Extend each finger fully when prompted (GREEN)
   - Flex each finger fully when prompted (RED)
   - Hold steady for 2-3 seconds each time

1. **Press 'S' to save** when complete

## Controls

| Key   | Action            |
| ----- | ----------------- |
| **C** | Start calibration |
| **S** | Save calibration  |
| **L** | Load calibration  |
| **R** | Reset calibration |
| **Q** | Quit              |

## Calibration Sequence

```
Thumb    → EXTEND (3s) → FLEX (3s)
Index    → EXTEND (3s) → FLEX (3s)
Middle   → EXTEND (3s) → FLEX (3s)
Ring     → EXTEND (3s) → FLEX (3s)
Pinky    → EXTEND (3s) → FLEX (3s)
COMPLETE → Press 'S' to save
```

## Color Codes

- 🟢 **GREEN** = Extend your finger
- 🔴 **RED** = Flex your finger
- 🔵 **CYAN** = Get ready
- 🟣 **MAGENTA** = Complete!

## Tips

- ✅ Good lighting helps
- ✅ Keep hand in frame
- ✅ Hold steady during collection
- ✅ Fully extend/flex for best results
- ✅ Calibrate each hand separately

## Output

Calibration files are saved to:

```
vision_calibration/
├── calibration_left_hand.csv
└── calibration_right_hand.csv
```

## What Gets Calibrated

Each finger is calibrated for:

- **Extended value** (finger fully open)
- **Flexed value** (finger fully closed)

This creates a personalized range for YOUR hand.

## Using Calibrated Values

After calibration, finger values are normalized:

- `0.0` = Fully closed/flexed
- `0.5` = Half open
- `1.0` = Fully extended/open

Values are smooth and accurate across all positions!
