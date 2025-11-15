"""
Manual Control for Robotic Hand

This script allows you to manually control the robotic hand servos by typing in values.
Useful for testing, calibration, and maintenance.

Usage:
    python rpi/src/manual_control.py

Commands:
    - Type finger number (0-5) and position (0.0-1.0): "0 0.5"
    - Type "all <value>" to set all fingers to same position: "all 0.5"
    - Type "preset <name>" to load a preset: "preset open", "preset close", "preset rest"
    - Type "status" to see current positions
    - Type "help" for command list
    - Type "quit" or "exit" to close
"""

import serial
import sys
import time

# Serial port configuration
if sys.platform == "darwin":
    PORT = "/dev/tty.usbmodem11301"  # Mac
elif sys.platform == "win32":
    PORT = "COM3"  # Windows
else:
    PORT = "/dev/ttyAMA0"  # Raspberry Pi

BAUDRATE = 115200

# Finger names for reference
FINGER_NAMES = ["thumb_tip", "thumb_base", "index", "middle", "ring", "pinky"]

# Preset positions (normalized 0.0-1.0)
PRESETS = {
    "open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "close": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "rest": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "fist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "point": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    "peace": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    "thumbsup": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
}


class ManualController:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.current_positions = [0.0] * 6  # Normalized positions [0.0, 1.0]

    def connect(self):
        """Connect to the Arduino serial port."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"✓ Connected to {self.port}")
            time.sleep(2)  # Wait for Arduino to initialize
            return True
        except Exception as e:
            print(f"✗ Failed to connect to {self.port}: {e}")
            print("\nMake sure:")
            print(f"  1. Arduino is connected to {self.port}")
            print("  2. No other program is using the port")
            print("  3. Port name is correct (check Device Manager on Windows)")
            return False

    def disconnect(self):
        """Disconnect from the serial port."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✓ Disconnected from serial port")

    def send_positions(self):
        """Send current positions to the Arduino as servo commands."""
        if not self.ser or not self.ser.is_open:
            print("Error: Not connected to serial port")
            return False

        # Convert normalized positions [0.0, 1.0] to servo values [0, 1023]
        servo_values = [int(pos * 1023) for pos in self.current_positions]
        servo_values = [max(0, min(1023, val)) for val in servo_values]  # Clamp

        # Send as comma-separated values
        msg = ",".join(map(str, servo_values)) + "\n"
        self.ser.write(msg.encode())

        return True

    def set_finger(self, finger_idx, position):
        """Set a single finger position."""
        if finger_idx < 0 or finger_idx >= 6:
            print(f"Error: Finger index must be 0-5 (got {finger_idx})")
            return False

        if position < 0.0 or position > 1.0:
            print(f"Error: Position must be 0.0-1.0 (got {position})")
            return False

        self.current_positions[finger_idx] = position
        self.send_positions()

        print(
            f"✓ Set {FINGER_NAMES[finger_idx]} to {position:.2f} (servo: {int(position * 1023)})"
        )
        return True

    def set_all(self, position):
        """Set all fingers to the same position."""
        if position < 0.0 or position > 1.0:
            print(f"Error: Position must be 0.0-1.0 (got {position})")
            return False

        self.current_positions = [position] * 6
        self.send_positions()

        print(f"✓ Set all fingers to {position:.2f} (servo: {int(position * 1023)})")
        return True

    def load_preset(self, preset_name):
        """Load a preset hand configuration."""
        preset_name = preset_name.lower()

        if preset_name not in PRESETS:
            print(f"Error: Unknown preset '{preset_name}'")
            print(f"Available presets: {', '.join(PRESETS.keys())}")
            return False

        self.current_positions = PRESETS[preset_name].copy()
        self.send_positions()

        print(f"✓ Loaded preset '{preset_name}'")
        return True

    def show_status(self):
        """Display current finger positions."""
        print("\n" + "=" * 60)
        print("CURRENT FINGER POSITIONS")
        print("=" * 60)
        print(f"{'Finger':<12} | {'Position':<8} | {'Servo':>5} | {'Bar':<20}")
        print("-" * 60)

        for i, name in enumerate(FINGER_NAMES):
            pos = self.current_positions[i]
            servo = int(pos * 1023)
            bar = "█" * int(pos * 20)
            print(f"{name:<12} | {pos:.2f}     | {servo:4d} | {bar}")

        print("=" * 60 + "\n")

    def show_help(self):
        """Display help information."""
        print("\n" + "=" * 60)
        print("MANUAL CONTROL - HELP")
        print("=" * 60)
        print("Commands:")
        print("  <finger> <position>   - Set single finger")
        print("                          Example: 0 0.5")
        print("                          Fingers: 0=thumb_tip, 1=thumb_base,")
        print("                                   2=index, 3=middle, 4=ring, 5=pinky")
        print("")
        print("  all <position>        - Set all fingers to same position")
        print("                          Example: all 0.5")
        print("")
        print("  preset <name>         - Load a preset configuration")
        print("                          Available: " + ", ".join(PRESETS.keys()))
        print("                          Example: preset close")
        print("")
        print("  status                - Show current positions")
        print("  help                  - Show this help")
        print("  quit / exit           - Close the program")
        print("")
        print("Position range: 0.0 (open) to 1.0 (closed)")
        print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("MANUAL HAND CONTROL")
    print("=" * 60)
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 60 + "\n")

    # Initialize controller
    controller = ManualController(PORT, BAUDRATE)

    # Connect to Arduino
    if not controller.connect():
        return

    # Set to rest position initially
    print("\nInitializing to rest position...")
    controller.load_preset("rest")
    controller.show_status()

    # Main control loop
    try:
        while True:
            try:
                # Get user input
                user_input = input(">>> ").strip().lower()

                if not user_input:
                    continue

                # Parse command
                parts = user_input.split()
                command = parts[0]

                if command in ["quit", "exit", "q"]:
                    print("\nExiting...")
                    break

                elif command == "help" or command == "h":
                    controller.show_help()

                elif command == "status" or command == "s":
                    controller.show_status()

                elif command == "all":
                    if len(parts) < 2:
                        print("Error: Usage: all <position>")
                        print("Example: all 0.5")
                        continue
                    try:
                        position = float(parts[1])
                        controller.set_all(position)
                    except ValueError:
                        print(
                            f"Error: Invalid position '{parts[1]}'. Must be a number 0.0-1.0"
                        )

                elif command == "preset" or command == "p":
                    if len(parts) < 2:
                        print("Error: Usage: preset <name>")
                        print(f"Available: {', '.join(PRESETS.keys())}")
                        continue
                    controller.load_preset(parts[1])

                else:
                    # Assume it's "<finger> <position>"
                    if len(parts) < 2:
                        print("Error: Unknown command. Type 'help' for command list")
                        continue

                    try:
                        finger_idx = int(parts[0])
                        position = float(parts[1])
                        controller.set_finger(finger_idx, position)
                    except ValueError:
                        print("Error: Invalid input. Expected '<finger> <position>'")
                        print("Example: 0 0.5")
                        print("Type 'help' for more info")

            except KeyboardInterrupt:
                print("\n\nCtrl+C pressed. Type 'quit' to exit or continue...")
                continue

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Return to rest position before closing
        print("\nReturning to rest position...")
        controller.load_preset("rest")
        time.sleep(0.5)

        # Disconnect
        controller.disconnect()
        print("Goodbye!\n")


if __name__ == "__main__":
    main()
