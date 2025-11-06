import cv2
from pathlib import Path
from data_collection.calibration.ui_utils import (
    draw_calibration_ui,
    draw_hand_info,
    draw_calibration_status,
    draw_phase_instruction,
    print_calibration_loaded,
    print_no_calibration,
)


def auto_load_calibration(detector, calibration_dir=None):
    current_hand = None
    if calibration_dir is None:
        calibration_dir = Path("data/calibration")
    else:
        calibration_dir = Path(calibration_dir)

    if calibration_dir.exists():
        for hand in ["Right", "Left"]:
            calibration_file = calibration_dir / f"calibration_{hand.lower()}_hand.csv"
            if calibration_file.exists():
                if detector.loadCalibration(hand):
                    current_hand = hand
                    print_calibration_loaded(hand)
                    print(detector.calibration_manager.get_calibration_summary())
                    return current_hand

    print_no_calibration()
    return None


def handle_calibration_frame(frame, detector, workflow, hand_label, raw_values):
    current_finger = workflow.get_current_finger()
    if current_finger < len(raw_values):
        workflow.update(raw_values[current_finger])

    draw_calibration_ui(frame, workflow, hand_label)

    return workflow.is_complete()


def handle_normal_frame(frame, detector, hand_label, fingers):
    draw_hand_info(frame, hand_label, fingers)
    draw_calibration_status(
        frame,
        detector.calibration_manager.is_calibrated,
        detector.calibration_manager.current_hand_label,
    )


def complete_calibration(detector, workflow, hand_label):
    results = workflow.get_results()
    for finger_id, values in results.items():
        detector.calibration_manager.set_finger_calibration(
            finger_id, values["extended"], values["flexed"]
        )
    detector.calibration_manager.is_calibrated = True
    detector.calibration_manager.current_hand_label = hand_label

    detector.saveCalibration(hand_label)
    print("\nCalibration complete and saved!")
    print(detector.calibration_manager.get_calibration_summary())

    workflow.stop()
    return hand_label


def run_calibration_loop(
    cap, detector, workflow, window_name="Integrated Data Collection", calibration_dir=None
):
    print("\n" + "=" * 60)
    print("CALIBRATION PHASE")
    print("=" * 60)
    print("Press 'C' to start calibration")
    print("Press 'Q' to skip calibration")
    print("=" * 60)

    current_hand = auto_load_calibration(detector, calibration_dir)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)

        key = cv2.waitKey(1) & 0xFF

        if detector.results.multi_hand_landmarks:
            lmList = detector.findPosition(frame, handNo=0, draw=False)
            hand_label = detector.checkLeftRight(handNo=0)

            if len(lmList) != 0:
                raw_values = detector.getRawFingerValues()

                if workflow.is_active:
                    is_complete = handle_calibration_frame(
                        frame, detector, workflow, hand_label, raw_values
                    )

                    if is_complete:
                        current_hand = complete_calibration(
                            detector, workflow, hand_label
                        )
                        return current_hand
                else:
                    fingers = detector.fingersUp(handNo=0)
                    handle_normal_frame(frame, detector, hand_label, fingers)

        draw_phase_instruction(
            frame, "Calibration Phase - Press 'C' to calibrate, 'Q' to continue"
        )

        cv2.imshow(window_name, frame)

        if key == ord("q") or key == ord("Q"):
            if detector.calibration_manager.is_calibrated:
                return current_hand
            else:
                print("\nWarning: Proceeding without calibration!")
                return None
        elif key == ord("c") or key == ord("C"):
            if detector.results.multi_hand_landmarks:
                hand_label = detector.checkLeftRight(handNo=0)
                print(f"\nStarting calibration for {hand_label} hand...")
                workflow.start()
                current_hand = hand_label
            else:
                print("\nNo hand detected! Please show your hand to the camera.")
