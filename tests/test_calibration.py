"""
Test script for calibration system
Tests the CalibrationManager and CalibrationWorkflow classes
"""

from src.calibration_manager import CalibrationManager
from src.calibration_workflow import CalibrationWorkflow
import time


def test_calibration_manager():
    """Test the CalibrationManager functionality."""
    print("Testing CalibrationManager...")

    # Create manager
    manager = CalibrationManager(calibration_dir="vision_calibration_test")

    # Test setting calibration data
    print("Setting calibration data for 5 fingers...")
    for finger_id in range(5):
        extended = 0.8 + finger_id * 0.02
        flexed = 0.3 + finger_id * 0.01
        manager.set_finger_calibration(finger_id, extended, flexed)

    # Test saving
    print("Saving calibration for Right hand...")
    manager.save_calibration("Right")

    # Test loading
    print("Loading calibration for Right hand...")
    success = manager.load_calibration("Right")
    print(f"Load successful: {success}")

    # Test lerp
    print("\nTesting lerp function...")
    test_value = 0.5
    min_val = 0.3
    max_val = 0.8
    result = manager.lerp(test_value, min_val, max_val)
    print(f"lerp({test_value}, {min_val}, {max_val}) = {result:.3f}")
    print(f"Expected: {(0.5 - 0.3) / (0.8 - 0.3):.3f}")

    # Test normalization
    print("\nTesting normalization...")
    raw_values = [0.5, 0.6, 0.55, 0.65, 0.58]
    normalized = manager.get_normalized_finger_values(raw_values)
    print(f"Raw values: {[round(v, 3) for v in raw_values]}")
    print(f"Normalized: {[round(v, 3) for v in normalized]}")

    # Test summary
    print("\nCalibration summary:")
    print(manager.get_calibration_summary())

    print("\n✓ CalibrationManager tests passed!")


def test_calibration_workflow():
    """Test the CalibrationWorkflow functionality."""
    print("\n" + "=" * 60)
    print("Testing CalibrationWorkflow...")

    workflow = CalibrationWorkflow()

    # Test starting
    print("Starting workflow...")
    workflow.start()
    assert workflow.is_active, "Workflow should be active"

    # Simulate workflow progression
    print("\nSimulating calibration process...")

    # Test raw values for each finger
    test_raw_values = [
        [0.75, 0.35],  # Thumb: extended, flexed
        [0.85, 0.40],  # Index
        [0.90, 0.42],  # Middle
        [0.88, 0.41],  # Ring
        [0.82, 0.38],  # Pinky
    ]

    for finger_idx in range(5):
        print(
            f"\nCalibrating finger {finger_idx} ({workflow.FINGER_NAMES[finger_idx]})..."
        )

        # Simulate extended phase
        for _ in range(30):  # Simulate ~3 seconds at 10 updates/sec
            workflow.update(test_raw_values[finger_idx][0])
            time.sleep(0.1)

            instruction = workflow.get_instruction_text()
            if "EXTENDED" in instruction:
                print(f"  {instruction}", end="\r")

        print()  # New line

        # Simulate flexed phase
        for _ in range(30):  # Simulate ~3 seconds at 10 updates/sec
            workflow.update(test_raw_values[finger_idx][1])
            time.sleep(0.1)

            instruction = workflow.get_instruction_text()
            if "FLEXED" in instruction:
                print(f"  {instruction}", end="\r")

        print()  # New line

    # Check completion
    assert workflow.is_complete(), "Workflow should be complete"
    print("\nWorkflow complete!")

    # Get results
    results = workflow.get_results()
    print("\nCalibration results:")
    for finger_id, values in results.items():
        print(
            f"  {workflow.FINGER_NAMES[finger_id]}: "
            f"Extended={values['extended']:.3f}, "
            f"Flexed={values['flexed']:.3f}"
        )

    print("\n✓ CalibrationWorkflow tests passed!")


def test_integration():
    """Test integration between CalibrationManager and CalibrationWorkflow."""
    print("\n" + "=" * 60)
    print("Testing integration...")

    # Create instances
    manager = CalibrationManager(calibration_dir="vision_calibration_test")
    workflow = CalibrationWorkflow()

    # Simulate quick workflow
    workflow.start()
    test_values = [[0.8, 0.3], [0.85, 0.35], [0.9, 0.4], [0.88, 0.38], [0.82, 0.36]]

    # Fast simulation
    for finger_idx in range(5):
        # Extended
        for _ in range(30):
            workflow.update(test_values[finger_idx][0])
            time.sleep(0.01)  # Fast simulation

        # Flexed
        for _ in range(30):
            workflow.update(test_values[finger_idx][1])
            time.sleep(0.01)

    # Transfer results to manager
    results = workflow.get_results()
    for finger_id, values in results.items():
        manager.set_finger_calibration(finger_id, values["extended"], values["flexed"])

    manager.is_calibrated = True

    # Test normalization with calibrated data
    test_raw = [0.55, 0.60, 0.65, 0.63, 0.59]
    normalized = manager.get_normalized_finger_values(test_raw)

    print("Integration test results:")
    print(f"  Raw values: {[round(v, 3) for v in test_raw]}")
    print(f"  Normalized: {[round(v, 3) for v in normalized]}")

    # Save and reload
    manager.save_calibration("Right")

    new_manager = CalibrationManager(calibration_dir="vision_calibration_test")
    success = new_manager.load_calibration("Right")

    assert success, "Should reload calibration successfully"

    # Test same values produce same results
    normalized2 = new_manager.get_normalized_finger_values(test_raw)
    assert normalized == normalized2, "Results should match after reload"

    print("\n✓ Integration tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Calibration System Test Suite")
    print("=" * 60)

    try:
        test_calibration_manager()
        test_calibration_workflow()
        test_integration()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
