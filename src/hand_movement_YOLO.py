import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import math
import mediapipe as mp


# HandDetector class
class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5):
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

    def fingersUp(self, handNo=0) -> list[int]:
        """Returns list [thumb, index, middle, ring, pinky] (1=up, 0=down).

        Args:
            handNo: Which hand to check (0 for first detected, 1 for second)

        Returns:
            List of finger states [thumb, index, middle, ring, pinky]
        """
        fingers = []
        if len(self.landmarks) == 0:
            return []

        hand_size = self.getDistance(0, 9)

        hand_label = self.checkLeftRight(handNo)
        thumb_distance = self.getDistance(self.tipIds[0], self.mcpIds[0])
        thumb_ratio = thumb_distance / hand_size
        if hand_label is None:
            hand_label = "Right"

        # Thumb: check x position (different logic for left vs right hand)
        if hand_label == "Right":
            # For right hand, thumb is up when tip x < dip x
            # if self.landmarks[self.tipIds[0]][1] < self.landmarks[self.dipIds[0]][1]:
            #     fingers.append(1)
            if thumb_ratio > 0.6:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            # For left hand, thumb is up when tip x > dip x
            if self.landmarks[self.tipIds[0]][1] > self.landmarks[self.dipIds[0]][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Other fingers: tip higher than pip joint (same for both hands)
        for id in range(1, 5):
            if self.landmarks[self.tipIds[id]][2] < self.landmarks[self.pipIds[id]][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


# --- Main program ---

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)  # Only detect one hand at a time
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture video.")
        break
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)

    # Check if any hands are detected
    if detector.results.multi_hand_landmarks:
        lmList = detector.findPosition(frame, handNo=0, draw=False)
        if len(lmList) != 0:
            fingers = detector.fingersUp(handNo=0)
            hand_label = detector.checkLeftRight(handNo=0)
            text = f"{hand_label} Hand: {fingers}"
            cv2.putText(
                frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
