"""
User-specific path management for data collection and calibration.
Supports multiple users with separate data directories.
"""

from pathlib import Path
from typing import Tuple


def list_existing_users(data_root: str = "data") -> list[str]:
    """List all existing users by scanning the data directory.

    Args:
        data_root: Root data directory (default: "data")

    Returns:
        List of usernames found in the data directory
    """
    data_path = Path(data_root)
    if not data_path.exists():
        return []

    users = []
    for item in data_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            users.append(item.name)

    return sorted(users)


def get_user_input(data_root: str = "data") -> str:
    """Prompt user for username and return it.

    Shows list of existing users and prompts for input.
    Returns 'default' if user presses Enter without input.

    Args:
        data_root: Root data directory (default: "data")

    Returns:
        Username string (never empty, defaults to 'default')
    """
    existing_users = list_existing_users(data_root)

    print("\n" + "=" * 60)
    print("USER SELECTION")
    print("=" * 60)

    if existing_users:
        print(f"Existing users: {', '.join(existing_users)}")
    else:
        print("No existing users found. Starting fresh!")

    print("-" * 60)
    username = input("Enter username (press Enter for 'default'): ").strip()

    if not username:
        username = "default"

    print(f"Selected user: {username}")
    print("=" * 60)

    return username


def get_user_paths(username: str = "default", data_root: str = "data") -> Tuple[Path, Path, Path]:
    """Get user-specific paths for data storage.

    Args:
        username: Username for data separation (default: "default")
        data_root: Root data directory (default: "data")

    Returns:
        Tuple of (calibration_dir, raw_data_dir, processed_data_dir)
    """
    user_base = Path(data_root) / username

    calibration_dir = user_base / "calibration"
    raw_data_dir = user_base / "raw"
    processed_data_dir = user_base / "processed"

    # Create directories if they don't exist
    calibration_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    return calibration_dir, raw_data_dir, processed_data_dir


def print_user_paths(username: str, calibration_dir: Path, raw_data_dir: Path):
    """Print user-specific paths in a formatted way.

    Args:
        username: Username
        calibration_dir: Path to calibration directory
        raw_data_dir: Path to raw data directory
    """
    print("\n" + "=" * 60)
    print(f"USER: {username}")
    print("=" * 60)
    print(f"  Calibration: {calibration_dir}")
    print(f"  Raw data:    {raw_data_dir}")
    print("=" * 60 + "\n")
