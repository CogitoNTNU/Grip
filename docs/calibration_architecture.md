# Calibration System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Hand Tracking Application                      │
│                     (hand_movement_YOLO.py)                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ Uses
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         HandDetector                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • findHands()           - Detect hands in frame              │  │
│  │ • findPosition()        - Get landmark positions             │  │
│  │ • getRawFingerValues()  - Calculate raw distance ratios     │  │
│  │ • fingersUp()           - Get normalized finger values       │  │
│  │ • loadCalibration()     - Load saved calibration             │  │
│  │ • saveCalibration()     - Save calibration data              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                    │                           │
                    │ Uses                      │ Uses
                    ▼                           ▼
┌──────────────────────────────────┐  ┌──────────────────────────────┐
│    CalibrationManager            │  │   CalibrationWorkflow        │
│  (calibration_manager.py)        │  │  (calibration_workflow.py)   │
├──────────────────────────────────┤  ├──────────────────────────────┤
│ • Stores calibration data        │  │ • Guides calibration process │
│ • Performs linear interpolation  │  │ • State machine for workflow │
│ • Saves/loads CSV files          │  │ • Collects calibration data  │
│ • get_normalized_finger_values() │  │ • Provides UI instructions   │
│ • lerp(value, min, max)          │  │ • Visual feedback            │
└──────────────────────────────────┘  └──────────────────────────────┘
                    │                           │
                    │                           │
                    ▼                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    File System (CSV Storage)                      │
│  vision_calibration/                                             │
│  ├── calibration_left_hand.csv                                   │
│  └── calibration_right_hand.csv                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

### During Calibration

```
User Input (Hand Gestures)
         │
         ▼
   Camera Frame
         │
         ▼
HandDetector.findHands()
         │
         ▼
HandDetector.findPosition()
         │
         ▼
HandDetector.getRawFingerValues()  ──────────┐
         │                                    │
         │                                    │
         ▼                                    │
CalibrationWorkflow.update()                 │
         │                                    │
         │ (collect samples)                 │
         ▼                                    │
CalibrationWorkflow.get_results()            │
         │                                    │
         ▼                                    │
CalibrationManager.set_finger_calibration()  │
         │                                    │
         ▼                                    │
CalibrationManager.save_calibration()        │
         │                                    │
         ▼                                    │
    CSV File                                  │
                                             │
                                             │
### During Normal Operation                  │
                                             │
User Input (Hand Gestures)                   │
         │                                    │
         ▼                                    │
   Camera Frame                              │
         │                                    │
         ▼                                    │
HandDetector.findHands()                     │
         │                                    │
         ▼                                    │
HandDetector.findPosition()                  │
         │                                    │
         ▼                                    │
HandDetector.getRawFingerValues() ◄──────────┘
         │
         ▼
CalibrationManager.get_normalized_finger_values()
         │
         │ (apply lerp)
         ▼
    Normalized Values [0.0 - 1.0]
         │
         ▼
   Application Logic
```

## State Machine (CalibrationWorkflow)

```
        START
          │
          ▼
    ┌──────────┐
    │ WAITING  │ ◄────────────────────┐
    └──────────┘                      │
          │                           │
          │ (1 second delay)          │
          ▼                           │
    ┌──────────┐                      │
    │  EXTEND  │                      │
    └──────────┘                      │
          │                           │
          │ (show instruction)        │
          ▼                           │
    ┌────────────────────┐            │
    │ EXTEND_COLLECTING  │            │
    │  (2.5 seconds)     │            │
    │  [samples++]       │            │
    └────────────────────┘            │
          │                           │
          │ (calculate avg)           │
          ▼                           │
    ┌──────────┐                      │
    │   FLEX   │                      │
    └──────────┘                      │
          │                           │
          │ (show instruction)        │
          ▼                           │
    ┌────────────────────┐            │
    │  FLEX_COLLECTING   │            │
    │   (2.5 seconds)    │            │
    │   [samples++]      │            │
    └────────────────────┘            │
          │                           │
          │ (calculate avg)           │
          ▼                           │
    ┌──────────┐                      │
    │   NEXT   │ ─── (more fingers) ──┘
    │  FINGER  │
    └──────────┘
          │
          │ (all 5 done)
          ▼
    ┌──────────┐
    │ COMPLETE │
    └──────────┘
          │
          ▼
         END
```

## Linear Interpolation (Lerp) Visualization

```
Raw Value Scale:
  0.3        0.5        0.7        0.9
   │──────────│──────────│──────────│
   ▲                                ▲
   │                                │
 Flexed                          Extended
(calibrated min)            (calibrated max)

After Lerp Normalization:
  0.0        0.5        1.0
   │──────────│──────────│
   ▲          ▲          ▲
   │          │          │
 Closed     Half       Open

Example:
  Raw = 0.5, Flexed = 0.3, Extended = 0.9
  Normalized = (0.5 - 0.3) / (0.9 - 0.3) = 0.2 / 0.6 = 0.33

This means the finger is 33% of the way from closed to open.
```

## File Format

```csv
┌────────────────────────────────────────────┐
│ timestamp,2025-10-09T15:30:00             │ ◄── Metadata
│ hand,Right                                 │
├────────────────────────────────────────────┤
│ finger_id,finger_name,extended,flexed     │ ◄── Header
├────────────────────────────────────────────┤
│ 0,Thumb,0.850,0.320                       │ ◄── Data
│ 1,Index,0.920,0.380                       │     (per finger)
│ 2,Middle,0.950,0.420                      │
│ 3,Ring,0.900,0.410                        │
│ 4,Pinky,0.820,0.360                       │
└────────────────────────────────────────────┘
```

## User Interface Flow

```
┌─────────────────────────────────────────────┐
│          Normal Operation Mode              │
│                                             │
│  Press 'C' to start calibration             │
│  Press 'L' to load existing calibration     │
│  Press 'Q' to quit                          │
└─────────────────────────────────────────────┘
                    │
                    │ (Press 'C')
                    ▼
┌─────────────────────────────────────────────┐
│         Calibration Mode Active             │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  Calibrating Thumb: EXTEND finger!  │   │ ◄── Green text
│  │  Finger 1/5 (Thumb)                 │   │
│  │  Hand: Right                        │   │
│  └─────────────────────────────────────┘   │
│                                             │
│          ●  Pulsing indicator               │ ◄── Visual feedback
└─────────────────────────────────────────────┘
                    │
                    │ (After 2.5 seconds)
                    ▼
┌─────────────────────────────────────────────┐
│         Calibration Mode Active             │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  Calibrating Thumb: FLEX finger!    │   │ ◄── Red text
│  │  Finger 1/5 (Thumb)                 │   │
│  │  Hand: Right                        │   │
│  └─────────────────────────────────────┘   │
│                                             │
│          ●  Pulsing indicator               │
└─────────────────────────────────────────────┘
                    │
                    │ (Repeat for all 5 fingers)
                    ▼
┌─────────────────────────────────────────────┐
│      Calibration Complete!                  │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  Calibration COMPLETE!              │   │ ◄── Magenta text
│  │  Press 'S' to save                  │   │
│  │  Hand: Right                        │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    │
                    │ (Press 'S')
                    ▼
┌─────────────────────────────────────────────┐
│          Calibration Saved!                 │
│                                             │
│  File: vision_calibration/                  │
│        calibration_right_hand.csv           │
│                                             │
│  Back to normal operation mode              │
└─────────────────────────────────────────────┘
```
