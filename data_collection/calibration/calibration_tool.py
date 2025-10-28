"""
Calibration Management Utility
Command-line tool for managing hand tracking calibration files
"""

import argparse
import sys
from pathlib import Path
from data_collection.calibration.calibration_manager import CalibrationManager


def list_calibrations(calibration_dir: str = "data/calibration"):
    """List all available calibration files."""
    path = Path(calibration_dir)

    if not path.exists():
        print(f"Calibration directory does not exist: {path}")
        return

    files = list(path.glob("calibration_*.csv"))

    if not files:
        print(f"No calibration files found in {path}")
        return

    print(f"\nCalibration files in {path}:")
    print("=" * 60)

    for file in sorted(files):
        print(f"  • {file.name}")

        # Try to load and show info
        try:
            manager = CalibrationManager(calibration_dir)
            hand = "Right" if "right" in file.name else "Left"
            if manager.load_calibration(hand):
                print(
                    f"    {manager.get_calibration_summary().replace(chr(10), chr(10) + '    ')}"
                )
        except Exception as e:
            print(f"    Error loading: {e}")

    print()


def view_calibration(hand: str, calibration_dir: str = "vision_calibration"):
    """View detailed calibration data for a specific hand."""
    manager = CalibrationManager(calibration_dir)

    if not manager.load_calibration(hand):
        print(f"Failed to load calibration for {hand} hand")
        return

    print(f"\nCalibration Data for {hand} Hand")
    print("=" * 60)
    print(manager.get_calibration_summary())

    # Show example normalized values
    print("\nExample normalized values:")
    print("-" * 60)

    test_values = [
        [0.3, "Fully flexed (should be ~0.0)"],
        [0.5, "Half open (should be ~0.5)"],
        [0.7, "Mostly open (should be ~0.7-0.8)"],
        [0.9, "Fully extended (should be ~1.0)"],
    ]

    for test_val, description in test_values:
        raw = [test_val] * 5
        normalized = manager.get_normalized_finger_values(raw)
        print(f"  Raw {test_val}: {[round(v, 2) for v in normalized]} - {description}")

    print()


def delete_calibration(hand: str, calibration_dir: str = "vision_calibration"):
    """Delete calibration file for a specific hand."""
    manager = CalibrationManager(calibration_dir)
    filepath = manager.get_calibration_filepath(hand)

    if not filepath.exists():
        print(f"No calibration file found for {hand} hand")
        return

    confirm = input(f"Delete calibration for {hand} hand? (yes/no): ").strip().lower()

    if confirm == "yes":
        filepath.unlink()
        print(f"Deleted: {filepath}")
    else:
        print("Cancelled")


def export_calibration(
    hand: str, output_file: str, calibration_dir: str = "vision_calibration"
):
    """Export calibration to a different location."""
    manager = CalibrationManager(calibration_dir)
    source = manager.get_calibration_filepath(hand)

    if not source.exists():
        print(f"No calibration file found for {hand} hand")
        return

    import shutil

    dest = Path(output_file)

    shutil.copy2(source, dest)
    print(f"Exported calibration to: {dest}")


def import_calibration(
    input_file: str, hand: str, calibration_dir: str = "vision_calibration"
):
    """Import calibration from a file."""
    import shutil

    source = Path(input_file)

    if not source.exists():
        print(f"Source file not found: {source}")
        return

    manager = CalibrationManager(calibration_dir)
    dest = manager.get_calibration_filepath(hand)

    # Create directory if it doesn't exist
    dest.parent.mkdir(exist_ok=True)

    shutil.copy2(source, dest)
    print(f"Imported calibration for {hand} hand from: {source}")

    # Verify
    if manager.load_calibration(hand):
        print("✓ Calibration imported and verified successfully")
        print(manager.get_calibration_summary())
    else:
        print("✗ Failed to verify imported calibration")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Hand Tracking Calibration Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                          # List all calibrations
  %(prog)s view -h Right                 # View right hand calibration
  %(prog)s delete -h Left                # Delete left hand calibration
  %(prog)s export -h Right -o backup.csv # Export calibration
  %(prog)s import -i backup.csv -h Right # Import calibration
        """,
    )

    parser.add_argument(
        "action",
        choices=["list", "view", "delete", "export", "import"],
        help="Action to perform",
    )

    parser.add_argument(
        "-h",
        "--hand",
        choices=["Left", "Right"],
        help="Hand to operate on (Left or Right)",
    )

    parser.add_argument(
        "-d",
        "--dir",
        default="vision_calibration",
        help="Calibration directory (default: vision_calibration)",
    )

    parser.add_argument("-o", "--output", help="Output file for export")

    parser.add_argument("-i", "--input", help="Input file for import")

    args = parser.parse_args()

    try:
        if args.action == "list":
            list_calibrations(args.dir)

        elif args.action == "view":
            if not args.hand:
                print("Error: --hand is required for 'view' action")
                sys.exit(1)
            view_calibration(args.hand, args.dir)

        elif args.action == "delete":
            if not args.hand:
                print("Error: --hand is required for 'delete' action")
                sys.exit(1)
            delete_calibration(args.hand, args.dir)

        elif args.action == "export":
            if not args.hand or not args.output:
                print("Error: --hand and --output are required for 'export' action")
                sys.exit(1)
            export_calibration(args.hand, args.output, args.dir)

        elif args.action == "import":
            if not args.input or not args.hand:
                print("Error: --input and --hand are required for 'import' action")
                sys.exit(1)
            import_calibration(args.input, args.hand, args.dir)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
