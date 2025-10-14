import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import math
import mediapipe as mp

from src.calibration_manager import CalibrationManager


# HandDetector class
class HandDetector:
    def __init__(
        self,
        mode=False,
        maxHands=1,
        detectionCon=0.7,
        trackCon=0.5,
        use_calibration=True,
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.dipIds = [3, 7, 11, 15, 19]
        self.pipIds = [2, 6, 10, 14, 18]
        self.mcpIds = [1, 5, 9, 13, 17]
        self.landmarks = []

        # Calibration
        self.use_calibration = use_calibration
        self.calibration_manager = CalibrationManager() if use_calibration else None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def checkLeftRight(self, handNo=0):
        """Check if a specific hand is left or right.

        Args:
            handNo: Which hand to check (0 for first detected, 1 for second)

        Returns:
            'Left' or 'Right' string, or None if not detected
        """
        if not self.results.multi_handedness:
            return None
        if handNo < len(self.results.multi_handedness):
            hand_label = self.results.multi_handedness[handNo].classification[0].label
            return hand_label  # 'Left' or 'Right'
        return None

    def findPosition(self, img, handNo=0, draw=True):
        """Find position of landmarks for a specific hand.

        Args:
            img: The image frame
            handNo: Which hand to get landmarks for (0 for first detected, 1 for second)
            draw: Whether to draw circles on landmarks

        Returns:
            List of landmarks [id, x, y] for the specified hand
        """
        self.landmarks = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmarks.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.landmarks

    def getDistance(self, id1, id2):
        return math.sqrt(
            (self.landmarks[id1][1] - self.landmarks[id2][1]) ** 2
            + (self.landmarks[id1][2] - self.landmarks[id2][2]) ** 2
        )

    def getRawFingerValues(self) -> list[float]:
        """Get raw finger distance ratios without calibration.

        Returns:
            List of 5 raw distance ratios [thumb, index, middle, ring, pinky]
        """
        raw_values = []
        if len(self.landmarks) == 0:
            return []

        hand_size = self.getDistance(0, 9)

        # thumb
        thumb_tip_ratio = self.getDistance(self.tipIds[0], self.pipIds[0]) / hand_size
        raw_values.append(thumb_tip_ratio)

        thumb_base_ratio = self.getDistance(self.pipIds[0], self.mcpIds[0]) / hand_size
        thumb_base_ratio += (
            self.getDistance(self.dipIds[0], self.mcpIds[-1]) / hand_size
        )
        raw_values.append(thumb_base_ratio)

        for id in range(1, 5):
            # Calculate distance ratio (tip to PIP + tip to MCP) normalized by hand size
            finger_ratio = (
                self.getDistance(self.tipIds[id], self.pipIds[id]) / hand_size
            )
            finger_ratio += (
                self.getDistance(self.tipIds[id], self.mcpIds[id]) / hand_size
            )
            raw_values.append(finger_ratio)

        return raw_values

    def fingersUp(self, handNo=0, normalize=True) -> list[float]:
        """Returns list [thumb, index, middle, ring, pinky] with values 0-1.

        Args:
            handNo: Which hand to check (0 for first detected, 1 for second)
            normalize: Whether to use calibration normalization (if available)

        Returns:
            List of finger extension values [thumb, index, middle, ring, pinky]
            0.0 = fully flexed/closed, 1.0 = fully extended/open
        """
        fingers = []
        if len(self.landmarks) == 0:
            return []

        # Get raw values
        raw_values = self.getRawFingerValues()

        # Apply calibration if enabled and available
        if (
            normalize
            and self.use_calibration
            and self.calibration_manager
            and self.calibration_manager.is_calibrated
        ):
            fingers = self.calibration_manager.get_normalized_finger_values(raw_values)
        else:
            # Use old method with power modifiers if no calibration
            # hand_size = self.getDistance(0, 9)
            finger_modifiers = [3.2, 3.2, 1.1, 1.7, 1.8, 1.6]

            for id, raw_value in enumerate(raw_values):
                finger_ratio = raw_value ** finger_modifiers[id]
                finger_ratio = min(max(finger_ratio, 0.0), 1.0)
                fingers.append(finger_ratio)

        return fingers

    def loadCalibration(self, hand_label: str) -> bool:
        """Load calibration data for a specific hand.

        Args:
            hand_label: 'Left' or 'Right'

        Returns:
            True if calibration loaded successfully
        """
        if self.calibration_manager:
            return self.calibration_manager.load_calibration(hand_label)
        return False

    def saveCalibration(self, hand_label: str) -> None:
        """Save current calibration data.

        Args:
            hand_label: 'Left' or 'Right'
        """
        if self.calibration_manager:
            self.calibration_manager.save_calibration(hand_label)


# --- Main program ---


def draw_calibration_ui(frame, workflow, hand_label):
    """Draw calibration UI elements on the frame.

    Args:
        frame: Video frame
        workflow: CalibrationWorkflow instance
        hand_label: Current hand label ('Left' or 'Right')
    """
    h, w, _ = frame.shape

    # Get instruction text
    instruction = workflow.get_instruction_text()
    progress = workflow.get_progress_text()

    # Get color based on state
    color = workflow.get_visual_indicator_color()

    # Draw semi-transparent overlay at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw instruction text
    cv2.putText(frame, instruction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Draw progress
    if progress:
        cv2.putText(
            frame, progress, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    # Draw hand label
    cv2.putText(
        frame,
        f"Hand: {hand_label}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    # Draw visual indicator (pulsing circle when collecting)
    if workflow.is_collecting():
        import time

        pulse = int((time.time() % 1.0) * 100) + 50
        cv2.circle(frame, (w - 80, 80), pulse, color, 3)


def main():
    """Main function to run hand tracking with calibration."""
    from src.calibration_workflow import CalibrationWorkflow
    from pathlib import Path

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(maxHands=1, use_calibration=True)
    workflow = CalibrationWorkflow()

    # Try to load existing calibration automatically
    calibration_loaded = False
    current_hand = None

    print("=" * 60)
    print("Hand Tracking with Calibration")
    print("=" * 60)

    # Auto-load calibration if available
    calibration_dir = Path("vision_calibration")
    if calibration_dir.exists():
        # Try to load calibration for both hands, prioritize Right hand
        for hand in ["Right", "Left"]:
            calibration_file = calibration_dir / f"calibration_{hand.lower()}_hand.csv"
            if calibration_file.exists():
                if detector.loadCalibration(hand):
                    current_hand = hand
                    calibration_loaded = True
                    print(f"✓ Auto-loaded calibration for {hand} hand")
                    print(detector.calibration_manager.get_calibration_summary())
                    break

    if not calibration_loaded:
        print("ℹ No existing calibration found. Press 'C' to calibrate.")

    print("=" * 60)
    print("Controls:")
    print("  C - Start calibration")
    print("  S - Save calibration (after completion)")
    print("  L - Load existing calibration")
    print("  R - Reset calibration")
    print("  Q - Quit")
    print("=" * 60)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture video.")
            break

        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)

        # Get keyboard input
        key = cv2.waitKey(1) & 0xFF

        # Check if any hands are detected
        if detector.results.multi_hand_landmarks:
            lmList = detector.findPosition(frame, handNo=0, draw=False)
            hand_label = detector.checkLeftRight(handNo=0)

            if len(lmList) != 0:
                # Get raw finger values for calibration
                raw_values = detector.getRawFingerValues()

                if workflow.is_active:
                    current_finger = workflow.get_current_finger()
                    if current_finger < len(raw_values):
                        workflow.update(raw_values[current_finger])

                    draw_calibration_ui(frame, workflow, hand_label)

                    if workflow.is_complete():
                        results = workflow.get_results()
                        for finger_id, values in results.items():
                            detector.calibration_manager.set_finger_calibration(
                                finger_id, values["extended"], values["flexed"]
                            )
                        detector.calibration_manager.is_calibrated = True
                        detector.calibration_manager.current_hand_label = hand_label
                        current_hand = hand_label

                        # Auto-save calibration
                        detector.saveCalibration(hand_label)
                        print("\nCalibration complete and saved automatically!")
                        print(detector.calibration_manager.get_calibration_summary())
                        print("You can press 'S' to save again if needed.")

                        workflow.stop()

                else:
                    fingers = detector.fingersUp(handNo=0)
                    fingers_display = [round(f, 1) for f in fingers]

                    # Display hand info
                    text = f"{hand_label} Hand: {fingers_display}"
                    cv2.putText(
                        frame,
                        text,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # Display calibration status
                    if detector.calibration_manager.is_calibrated:
                        status_text = f"Calibrated ({detector.calibration_manager.current_hand_label})"
                        cv2.putText(
                            frame,
                            status_text,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )
                    else:
                        status_text = "Not calibrated (Press 'C')"
                        cv2.putText(
                            frame,
                            status_text,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

        # Handle keyboard commands
        if key == ord("q"):
            break
        elif key == ord("c") or key == ord("C"):
            if detector.results.multi_hand_landmarks:
                hand_label = detector.checkLeftRight(handNo=0)
                print(f"\nStarting calibration for {hand_label} hand...")
                workflow.start()
                current_hand = hand_label
            else:
                print("\nNo hand detected! Please show your hand to the camera.")

        elif key == ord("s") or key == ord("S"):
            if workflow.is_complete() or detector.calibration_manager.is_calibrated:
                if current_hand:
                    detector.saveCalibration(current_hand)
                    print(f"Calibration saved for {current_hand} hand.")
                else:
                    print("No hand information available. Cannot save.")
            else:
                print("No calibration to save. Complete calibration first (Press 'C').")

        elif key == ord("l") or key == ord("L"):
            hand_input = (
                input("\nEnter hand to load calibration (Left/Right): ")
                .strip()
                .capitalize()
            )
            if hand_input in ["Left", "Right"]:
                if detector.loadCalibration(hand_input):
                    current_hand = hand_input
                    print(detector.calibration_manager.get_calibration_summary())
                else:
                    print(f"Failed to load calibration for {hand_input} hand.")
            else:
                print("Invalid input. Please enter 'Left' or 'Right'.")

        elif key == ord("r") or key == ord("R"):
            detector.calibration_manager.reset_calibration()
            workflow.stop()
            current_hand = None
            print("\nCalibration reset.")

        cv2.imshow("Hand Gesture Detection", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
