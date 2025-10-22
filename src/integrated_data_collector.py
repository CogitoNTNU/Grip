import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import csv
import time
from datetime import datetime
from src.hand_movement_YOLO import HandDetector
from src.calibration_workflow import CalibrationWorkflow
from src.calibration_helpers import run_calibration_loop
from src.serial.port_accessor import PortAccessor
from src.serial.serial_monitor import register_monitor
from src.data_collector import (
    create_data_directory,
    parse_port_event,
)
from src.ui_utils import (
    draw_hand_info,
    draw_calibration_status,
    draw_sensor_data_panel,
    draw_collection_status_bar,
    print_startup_banner,
)
def create_csv_file(data_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = data_dir / f"integrated_data_{timestamp}.csv"
    return csv_file
def write_csv_header(writer: csv.writer) -> None:
    header = [
        "timestamp",
        "iteration",
        "env0",
        "raw0",
        "env1",
        "raw1",
        "env2",
        "raw2",
        "env3",
        "raw3",
        "thumb_tip",
        "thumb_base",
        "index",
        "middle",
        "ring",
        "pinky",
        "hand_label",
    ]
    writer.writerow(header)
def get_hand_data(detector, frame):
    hand_label = "None"
    finger_values = [0.0] * 6
    if not detector.results.multi_hand_landmarks:
        return hand_label, finger_values
    lmList = detector.findPosition(frame, handNo=0, draw=False)
    hand_label = detector.checkLeftRight(handNo=0)
    if len(lmList) == 0:
        return hand_label if hand_label else "None", finger_values
    raw_values = detector.getRawFingerValues()
    if len(raw_values) >= 6:
        finger_values = raw_values[:6]
    return hand_label, finger_values
def draw_hand_ui(detector, frame, hand_label):
    if not detector.results.multi_hand_landmarks:
        return
    lmList = detector.findPosition(frame, handNo=0, draw=False)
    if len(lmList) != 0:
        fingers = detector.fingersUp(handNo=0)
        draw_hand_info(frame, hand_label, fingers)
        draw_calibration_status(
            frame,
            detector.calibration_manager.is_calibrated,
            detector.calibration_manager.current_hand_label,
        )
def collect_integrated_data(
    port: str = "MOCK",
    num_iterations: int = 1000,
    sleep_time: float = 0.05,
    batch_size: int = 100,
) -> None:
    data_dir = create_data_directory()
    csv_file = create_csv_file(data_dir)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(maxHands=1, use_calibration=True)
    workflow = CalibrationWorkflow()
    print_startup_banner()
    run_calibration_loop(cap, detector, workflow, "Integrated Data Collection")
    print("\n" + "=" * 60)
    print("DATA COLLECTION PHASE")
    print("=" * 60)
    print(f"Target samples: {num_iterations}")
    print(f"Batch size: {batch_size}")
    print("Press 'SPACE' to start/pause collection")
    print("Press 'Q' to quit and save")
    print("=" * 60)
    pa = PortAccessor(port=port)
    pa.open()
    subscription = pa.subscribe(max_queue=100)
    handle = register_monitor(pa, fs=1000, title="Sensor Monitor", plot_out=False)
    is_collecting = False
    sample_count = 0
    latest_sensor_values = ["0"] * 8
    batch_buffer = []
    should_exit = False
    try:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            write_csv_header(writer)
            while sample_count < num_iterations and not should_exit:
                success, frame = cap.read()
                if not success:
                    print("Failed to capture video.")
                    break
                frame = cv2.flip(frame, 1)
                frame = detector.findHands(frame)
                key = cv2.waitKey(1) & 0xFF
                try:
                    while not subscription.queue.empty():
                        payload = subscription.queue.get_nowait()
                        latest_sensor_values = parse_port_event(payload)
                except Exception as e:
                    print(f"Error parsing sensor data: {e}")
                    pass
                hand_label, finger_values = get_hand_data(detector, frame)
                draw_hand_ui(detector, frame, hand_label)
                draw_sensor_data_panel(frame, latest_sensor_values[:8])
                draw_collection_status_bar(
                    frame, is_collecting, sample_count, num_iterations
                )
                if is_collecting:
                    row = (
                        [datetime.now().isoformat(), sample_count]
                        + latest_sensor_values[:8]
                        + [f"{v:.4f}" for v in finger_values]
                        + [hand_label]
                    )
                    batch_buffer.append(row)
                    sample_count += 1
                    if len(batch_buffer) >= batch_size:
                        writer.writerows(batch_buffer)
                        f.flush()
                        batch_buffer.clear()
                        print(
                            f"\nBatch written: {sample_count}/{num_iterations} samples"
                        )
                    if sample_count >= num_iterations:
                        if batch_buffer:
                            writer.writerows(batch_buffer)
                            f.flush()
                            batch_buffer.clear()
                        print(
                            f"\nCollection complete! {sample_count} samples collected."
                        )
                        break
                    time.sleep(sleep_time)
                cv2.imshow("Integrated Data Collection", frame)
                if key == ord(" "):
                    is_collecting = not is_collecting
                    status = "started" if is_collecting else "paused"
                    print(f"\nData collection {status}")
                elif key == ord("q") or key == ord("Q"):
                    should_exit = True
                    print("\nShutting down gracefully...")
                    if batch_buffer:
                        print(f"Saving remaining {len(batch_buffer)} samples...")
                        writer.writerows(batch_buffer)
                        f.flush()
                        batch_buffer.clear()
                    print(f"Total samples collected: {sample_count}")
                    break
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        if batch_buffer:
            print(f"Saving remaining {len(batch_buffer)} samples...")
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(batch_buffer)
                batch_buffer.clear()
        print(f"Total samples collected: {sample_count}")
    except Exception as e:
        print(f"\nError during collection: {e}")
        if batch_buffer:
            print(f"Attempting to save remaining {len(batch_buffer)} samples...")
            try:
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_buffer)
                    batch_buffer.clear()
                print("Data saved successfully")
            except Exception as save_error:
                print(f"Failed to save remaining data: {save_error}")
    finally:
        print("\nCleaning up resources...")
        try:
            handle.stop()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            pa.unsubscribe(subscription)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            pa.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            cap.release()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        print(f"\nData saved to {csv_file}")
        print(f"Final sample count: {sample_count}")
        print("Shutdown complete.")
if __name__ == "__main__":
    collect_integrated_data(
        port="MOCK", num_iterations=1000, sleep_time=0.05, batch_size=100
    )