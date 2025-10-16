import cv2
import time


def draw_calibration_ui(frame, workflow, hand_label):
    h, w, _ = frame.shape

    instruction = workflow.get_instruction_text()
    progress = workflow.get_progress_text()
    color = workflow.get_visual_indicator_color()

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, instruction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if progress:
        cv2.putText(
            frame, progress, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    cv2.putText(
        frame,
        f"Hand: {hand_label}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    if workflow.is_collecting():
        pulse = int((time.time() % 1.0) * 100) + 50
        cv2.circle(frame, (w - 80, 80), pulse, color, 3)


def draw_hand_info(frame, hand_label, fingers):
    fingers_display = [round(f, 1) for f in fingers]
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


def draw_calibration_status(frame, is_calibrated, current_hand_label):
    if is_calibrated:
        status_text = f"Calibrated ({current_hand_label})"
        color = (0, 255, 255)
    else:
        status_text = "Not calibrated (Press 'C')"
        color = (0, 0, 255)

    cv2.putText(
        frame,
        status_text,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def print_startup_banner():
    print("=" * 60)
    print("Hand Tracking with Calibration")
    print("=" * 60)


def print_controls():
    print("=" * 60)
    print("Controls:")
    print("  C - Start calibration")
    print("  S - Save calibration (after completion)")
    print("  L - Load existing calibration")
    print("  R - Reset calibration")
    print("  Q - Quit")
    print("=" * 60)


def print_calibration_loaded(hand):
    print(f"✓ Auto-loaded calibration for {hand} hand")


def print_no_calibration():
    print("ℹ No existing calibration found. Press 'C' to calibrate.")
