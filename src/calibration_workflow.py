"""
Calibration Workflow for Hand Tracking
Guides the user through the calibration process with visual feedback.
"""

import time
from typing import List, Tuple
from enum import Enum


class CalibrationState(Enum):
    """States for the calibration process."""

    WAITING = "waiting"
    EXTEND = "extend"
    EXTEND_COLLECTING = "extend_collecting"
    FLEX = "flex"
    FLEX_COLLECTING = "flex_collecting"
    COMPLETE = "complete"


class CalibrationWorkflow:
    """Manages the calibration workflow with user instructions."""

    FINGER_NAMES = ["Thumb Tip", "Thumb Base", "Index", "Middle", "Ring", "Pinky"]
    COLLECTION_DURATION = 2.5  # seconds to collect data for each state
    TRANSITION_DELAY = 1.0  # seconds between states

    def __init__(self):
        """Initialize the calibration workflow."""
        self.current_finger = 0
        self.state = CalibrationState.WAITING
        self.start_time = 0
        self.collected_values_extended: List[float] = []
        self.collected_values_flexed: List[float] = []
        self.calibration_results = {}  # {finger_id: {'extended': avg, 'flexed': avg}}
        self.is_active = False

    def start(self) -> None:
        """Start the calibration workflow."""
        self.current_finger = 0
        self.state = CalibrationState.WAITING
        self.start_time = time.time()
        self.collected_values_extended.clear()
        self.collected_values_flexed.clear()
        self.calibration_results.clear()
        self.is_active = True

    def stop(self) -> None:
        """Stop the calibration workflow."""
        self.is_active = False
        self.state = CalibrationState.WAITING

    def update(self, current_raw_value: float) -> None:
        """Update the calibration workflow with current finger value.

        Args:
            current_raw_value: Current raw distance ratio for the finger being calibrated
        """
        if not self.is_active:
            return

        current_time = time.time()
        elapsed = current_time - self.start_time

        # State machine
        if self.state == CalibrationState.WAITING:
            if elapsed >= self.TRANSITION_DELAY:
                self.state = CalibrationState.EXTEND
                self.start_time = current_time
                self.collected_values_extended.clear()

        elif self.state == CalibrationState.EXTEND:
            if elapsed >= self.TRANSITION_DELAY:
                self.state = CalibrationState.EXTEND_COLLECTING
                self.start_time = current_time

        elif self.state == CalibrationState.EXTEND_COLLECTING:
            # Collect extended values
            self.collected_values_extended.append(current_raw_value)

            if elapsed >= self.COLLECTION_DURATION:
                # Calculate average
                avg_extended = sum(self.collected_values_extended) / len(
                    self.collected_values_extended
                )

                # Store temporarily
                if self.current_finger not in self.calibration_results:
                    self.calibration_results[self.current_finger] = {}
                self.calibration_results[self.current_finger]["extended"] = avg_extended

                # Move to flex state
                self.state = CalibrationState.FLEX
                self.start_time = current_time
                self.collected_values_flexed.clear()

        elif self.state == CalibrationState.FLEX:
            if elapsed >= self.TRANSITION_DELAY:
                self.state = CalibrationState.FLEX_COLLECTING
                self.start_time = current_time

        elif self.state == CalibrationState.FLEX_COLLECTING:
            # Collect flexed values
            self.collected_values_flexed.append(current_raw_value)

            if elapsed >= self.COLLECTION_DURATION:
                # Calculate average
                avg_flexed = sum(self.collected_values_flexed) / len(
                    self.collected_values_flexed
                )

                # Store result
                self.calibration_results[self.current_finger]["flexed"] = avg_flexed

                # Move to next finger or complete
                self.current_finger += 1

                if self.current_finger >= 6:  # Updated to include thumb tip and base
                    self.state = CalibrationState.COMPLETE
                else:
                    self.state = CalibrationState.WAITING
                    self.start_time = current_time

    def get_instruction_text(self) -> str:
        """Get the current instruction text for the user.

        Returns:
            Instruction string to display
        """
        if not self.is_active:
            return "Press 'C' to start calibration"

        if self.current_finger <= 1:
            if self.current_finger == 0:
                finger_name = "Thumb Tip"
            else:
                finger_name = "Thumb Base"
        else:
            finger_name = self.FINGER_NAMES[self.current_finger - 1]

        if self.state == CalibrationState.WAITING:
            return f"Calibrating {finger_name}... Get ready!"

        elif self.state == CalibrationState.EXTEND:
            return f"Calibrating {finger_name}: EXTEND finger fully!"

        elif self.state == CalibrationState.EXTEND_COLLECTING:
            elapsed = time.time() - self.start_time
            remaining = self.COLLECTION_DURATION - elapsed
            return f"Calibrating {finger_name}: Hold EXTENDED ({remaining:.1f}s)"

        elif self.state == CalibrationState.FLEX:
            return f"Calibrating {finger_name}: FLEX finger fully!"

        elif self.state == CalibrationState.FLEX_COLLECTING:
            elapsed = time.time() - self.start_time
            remaining = self.COLLECTION_DURATION - elapsed
            return f"Calibrating {finger_name}: Hold FLEXED ({remaining:.1f}s)"

        elif self.state == CalibrationState.COMPLETE:
            return "Calibration COMPLETE! Press 'S' to save."

        return ""

    def get_progress_text(self) -> str:
        """Get progress information.

        Returns:
            Progress string showing which finger and state
        """
        if not self.is_active and self.state != CalibrationState.COMPLETE:
            return ""

        # Handle completion state where current_finger might be >= 5
        if self.state == CalibrationState.COMPLETE or self.current_finger >= 6:
            return "All fingers calibrated (6/6)"

        return f"Finger {self.current_finger + 1}/6 ({self.FINGER_NAMES[self.current_finger]})"

    def is_collecting(self) -> bool:
        """Check if currently collecting data.

        Returns:
            True if in a collecting state
        """
        return self.state in [
            CalibrationState.EXTEND_COLLECTING,
            CalibrationState.FLEX_COLLECTING,
        ]

    def is_complete(self) -> bool:
        """Check if calibration is complete.

        Returns:
            True if calibration workflow is complete
        """
        return self.state == CalibrationState.COMPLETE

    def get_results(self) -> dict:
        """Get the calibration results.

        Returns:
            Dictionary of calibration results {finger_id: {'extended': value, 'flexed': value}}
        """
        return self.calibration_results.copy()

    def get_current_finger(self) -> int:
        """Get the current finger being calibrated.

        Returns:
            Finger index (0-5)
        """
        return self.current_finger

    def get_visual_indicator_color(self) -> Tuple[int, int, int]:
        """Get the color for visual feedback based on current state.

        Returns:
            BGR color tuple
        """
        if self.state == CalibrationState.WAITING:
            return (255, 255, 0)  # Cyan - waiting
        elif self.state in [
            CalibrationState.EXTEND,
            CalibrationState.EXTEND_COLLECTING,
        ]:
            return (0, 255, 0)  # Green - extend
        elif self.state in [CalibrationState.FLEX, CalibrationState.FLEX_COLLECTING]:
            return (0, 0, 255)  # Red - flex
        elif self.state == CalibrationState.COMPLETE:
            return (255, 0, 255)  # Magenta - complete

        return (255, 255, 255)  # White - default
