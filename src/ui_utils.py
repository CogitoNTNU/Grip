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


def draw_sensor_data_panel(frame, sensor_values, x_offset=None):
    h, w, _ = frame.shape

    if x_offset is None:
        x_offset = w - 250

    y_offset = 150
    cv2.putText(
        frame,
        "Sensor Data:",
        (x_offset, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )

    for i, val in enumerate(sensor_values):
        y_pos = y_offset + 30 + (i * 25)
        text = f"Ch{i}: {val}"
        cv2.putText(
            frame,
            text,
            (x_offset, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


def draw_collection_status_bar(frame, is_collecting, sample_count, target_samples):
    cv2.putText(
        frame,
        f"Samples: {sample_count}/{target_samples}",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if is_collecting else (100, 100, 100),
        2,
    )

    if is_collecting:
        cv2.putText(
            frame,
            "COLLECTING",
            (10, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )


def draw_phase_instruction(frame, text, color=(255, 255, 0)):
    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def print_no_calibration():
    print("ℹ No existing calibration found. Press 'C' to calibrate.")
