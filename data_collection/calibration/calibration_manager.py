"""
Calibration Manager for Hand Tracking
Handles calibration data storage, loading, and linear interpolation for finger tracking.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class CalibrationManager:
    """Manages calibration data for hand tracking."""

    FINGER_NAMES = ["Thumb Tip", "Thumb Base", "Index", "Middle", "Ring", "Pinky"]

    def __init__(self, calibration_dir: str = "data/calibration"):
        """Initialize the calibration manager.

        Args:
            calibration_dir: Directory to store calibration files
        """
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(exist_ok=True)

        # Updated calibration data structure to include thumb tip and base
        self.calibration_data: Dict[int, Dict[str, float]] = {
            0: {"extended": 0.9, "flexed": 0.3},  # Thumb Tip
            1: {"extended": 0.9, "flexed": 0.3},  # Thumb Base
            2: {"extended": 0.9, "flexed": 0.3},  # Index
            3: {"extended": 0.9, "flexed": 0.3},  # Middle
            4: {"extended": 0.9, "flexed": 0.3},  # Ring
            5: {"extended": 0.9, "flexed": 0.3},  # Pinky
        }

        self.default_extended = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        self.default_flexed = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

        self.is_calibrated = False
        self.current_hand_label = None  # 'Left' or 'Right'

    def get_calibration_filepath(self, hand_label: str) -> Path:
        """Get the calibration file path for a specific hand.

        Args:
            hand_label: 'Left' or 'Right'

        Returns:
            Path to the calibration CSV file
        """
        return self.calibration_dir / f"calibration_{hand_label.lower()}_hand.csv"

    def save_calibration(self, hand_label: str) -> None:
        """Save calibration data to CSV file.

        Args:
            hand_label: 'Left' or 'Right'
        """
        filepath = self.get_calibration_filepath(hand_label)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["timestamp", datetime.now().isoformat()])
            writer.writerow(["hand", hand_label])
            writer.writerow(
                ["finger_id", "finger_name", "extended_value", "flexed_value"]
            )

            for finger_id in range(6):  # Updated to include thumb tip and base
                if finger_id in self.calibration_data:
                    data = self.calibration_data[finger_id]
                    writer.writerow(
                        [
                            finger_id,
                            self.FINGER_NAMES[finger_id],
                            data["extended"],
                            data["flexed"],
                        ]
                    )

        print(f"Calibration saved to: {filepath}")

    def load_calibration(self, hand_label: str) -> bool:
        """Load calibration data from CSV file.

        Args:
            hand_label: 'Left' or 'Right'

        Returns:
            True if calibration loaded successfully, False otherwise
        """
        filepath = self.get_calibration_filepath(hand_label)

        if not filepath.exists():
            print(f"No calibration file found for {hand_label} hand at: {filepath}")
            return False

        try:
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

                # Skip header rows (first 3 rows)
                for row in rows[3:]:
                    if len(row) >= 4:
                        finger_id = int(row[0])
                        extended = float(row[2])
                        flexed = float(row[3])

                        self.calibration_data[finger_id] = {
                            "extended": extended,
                            "flexed": flexed,
                        }

            self.is_calibrated = True
            self.current_hand_label = hand_label
            print(f"Calibration loaded for {hand_label} hand from: {filepath}")
            return True

        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def set_finger_calibration(
        self, finger_id: int, extended_value: float, flexed_value: float
    ) -> None:
        """Set calibration values for a specific finger.

        Args:
            finger_id: Finger index (0-4)
            extended_value: Raw distance ratio when finger is fully extended
            flexed_value: Raw distance ratio when finger is fully flexed
        """
        self.calibration_data[finger_id] = {
            "extended": extended_value,
            "flexed": flexed_value,
        }

    def lerp(self, value: float, min_val: float, max_val: float) -> float:
        """Linear interpolation from [min_val, max_val] to [0, 1].

        Args:
            value: Current value to interpolate
            min_val: Minimum value (maps to 0)
            max_val: Maximum value (maps to 1)

        Returns:
            Interpolated value between 0 and 1
        """
        if max_val == min_val:
            return 0.5  # Avoid division by zero

        # Linear interpolation
        normalized = (value - min_val) / (max_val - min_val)

        # Clamp to [0, 1]
        return min(max(normalized, 0.0), 1.0)

    def get_normalized_finger_values(self, raw_values: List[float]) -> List[float]:
        """Normalize raw finger values using calibration data.

        Args:
            raw_values: List of 5 raw finger distance ratios

        Returns:
            List of 5 normalized values (0=fully flexed, 1=fully extended)
        """
        normalized = []

        for finger_id, raw_value in enumerate(raw_values):
            if finger_id in self.calibration_data and self.is_calibrated:
                # Use calibrated values
                flexed = self.calibration_data[finger_id]["flexed"]
                extended = self.calibration_data[finger_id]["extended"]
            else:
                # Use default values
                flexed = self.default_flexed[finger_id]
                extended = self.default_extended[finger_id]

            # Lerp from flexed (0) to extended (1)
            normalized_value = self.lerp(raw_value, flexed, extended)
            normalized.append(normalized_value)

        return normalized

    def is_calibration_complete(self) -> bool:
        """Check if calibration is complete for all 5 fingers.

        Returns:
            True if all fingers are calibrated, False otherwise
        """
        return (
            len(self.calibration_data) == 6
        )  # Updated to check for thumb tip and base

    def reset_calibration(self) -> None:
        """Reset calibration data."""
        self.calibration_data.clear()
        self.is_calibrated = False
        self.current_hand_label = None

    def get_calibration_summary(self) -> str:
        """Get a summary of current calibration data.

        Returns:
            String summary of calibration status
        """
        if not self.is_calibrated:
            return "No calibration loaded"

        summary = f"Calibration for {self.current_hand_label} hand:\n"
        for finger_id in range(6):  # Updated to include thumb tip and base
            if finger_id in self.calibration_data:
                data = self.calibration_data[finger_id]
                summary += f"  {self.FINGER_NAMES[finger_id]}: Extended={data['extended']:.3f}, Flexed={data['flexed']:.3f}\n"

        return summary
