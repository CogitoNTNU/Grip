"""
Filter Strategy Pattern for EMG Signal Processing

This module implements the Strategy pattern for different filtering approaches.
All filters can be used in both training (batch mode) and inference (streaming mode).

Available Strategies:
1. HighPassFilter - Removes low-frequency drift (DC bias, slow movements)
2. LowPassFilter - Smoothing filter, removes high-frequency noise
3. BandPassFilter - Combines high-pass and low-pass
4. NoFilter - Pass-through (no filtering)

Usage in Training (Batch Mode):
    filter_strategy = HighPassFilterStrategy(fs=4.2144, cutoff=0.5, order=4)
    filtered_signal = filter_strategy.apply_batch(signal)

Usage in Inference (Streaming Mode):
    filter_strategy = HighPassFilterStrategy(fs=4.2144, cutoff=0.5, order=4)
    for sample in samples:
        filtered_value = filter_strategy.apply_streaming(sample, channel=0)
"""

from abc import ABC, abstractmethod
from scipy.signal import butter, lfilter
import numpy as np
from typing import Dict, Optional


class FilterStrategy(ABC):
    """Abstract base class for filter strategies."""

    def __init__(self, fs: float):
        """
        Args:
            fs: Sampling rate in Hz
        """
        self.fs = fs
        self.streaming_state: Dict[int, np.ndarray] = {}  # Channel -> filter state

    @abstractmethod
    def apply_batch(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply filter to entire signal at once (training/batch mode).

        Args:
            signal: 1D array of signal values

        Returns:
            Filtered signal (same shape as input)
        """
        pass

    @abstractmethod
    def apply_streaming(self, value: float, channel: int = 0) -> float:
        """
        Apply filter to single sample (streaming/inference mode).
        Maintains state across calls for causal filtering.

        Args:
            value: Single sample value
            channel: Channel index (for multi-channel signals)

        Returns:
            Filtered value
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return human-readable description of filter configuration."""
        pass

    def reset_streaming_state(self, channel: Optional[int] = None):
        """
        Reset streaming filter state.

        Args:
            channel: If provided, reset only that channel. Otherwise reset all.
        """
        if channel is not None:
            if channel in self.streaming_state:
                del self.streaming_state[channel]
        else:
            self.streaming_state.clear()


class NoFilterStrategy(FilterStrategy):
    """No filtering - pass through original signal."""

    def __init__(self, fs: float):
        super().__init__(fs)

    def apply_batch(self, signal: np.ndarray) -> np.ndarray:
        return signal.copy()

    def apply_streaming(self, value: float, channel: int = 0) -> float:
        return value

    def get_description(self) -> str:
        return "No Filter (pass-through)"


class HighPassFilterStrategy(FilterStrategy):
    """
    High-pass Butterworth filter.
    Removes low-frequency components (DC bias, drift, slow movements).
    Good for removing baseline wander in EMG signals.
    """

    def __init__(self, fs: float, cutoff: float = 0.5, order: int = 4):
        """
        Args:
            fs: Sampling rate in Hz
            cutoff: Cutoff frequency in Hz (frequencies below this are attenuated)
            order: Filter order (higher = sharper cutoff, but more ringing)
        """
        super().__init__(fs)
        self.cutoff = cutoff
        self.order = order

        # Design Butterworth filter
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype="high", analog=False)

    def apply_batch(self, signal: np.ndarray) -> np.ndarray:
        """Apply causal high-pass filter to entire signal (uses lfilter)."""
        return lfilter(self.b, self.a, signal)

    def apply_streaming(self, value: float, channel: int = 0) -> float:
        """Apply causal high-pass filter to single sample."""
        # Initialize filter state for this channel if needed
        if channel not in self.streaming_state:
            self.streaming_state[channel] = np.zeros(max(len(self.a), len(self.b)) - 1)

        # Apply filter to single sample
        filtered_value, self.streaming_state[channel] = lfilter(
            self.b, self.a, [value], zi=self.streaming_state[channel]
        )

        return filtered_value[0]

    def get_description(self) -> str:
        return f"High-Pass Filter (cutoff={self.cutoff}Hz, order={self.order}, fs={self.fs:.4f}Hz)"


class LowPassFilterStrategy(FilterStrategy):
    """
    Low-pass Butterworth filter (smoothing filter).
    Removes high-frequency components (noise, artifacts).
    Good for smoothing noisy EMG signals.
    """

    def __init__(self, fs: float, cutoff: float = 5.0, order: int = 4):
        """
        Args:
            fs: Sampling rate in Hz
            cutoff: Cutoff frequency in Hz (frequencies above this are attenuated)
            order: Filter order (higher = sharper cutoff, but more ringing)
        """
        super().__init__(fs)
        self.cutoff = cutoff
        self.order = order

        # Design Butterworth filter
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Validate cutoff frequency
        if normal_cutoff >= 1.0:
            raise ValueError(
                f"Cutoff frequency {cutoff}Hz is too high for sampling rate {fs}Hz. "
                f"Must be less than Nyquist frequency ({nyq}Hz)."
            )

        self.b, self.a = butter(order, normal_cutoff, btype="low", analog=False)

    def apply_batch(self, signal: np.ndarray) -> np.ndarray:
        """Apply causal low-pass filter to entire signal (uses lfilter)."""
        return lfilter(self.b, self.a, signal)

    def apply_streaming(self, value: float, channel: int = 0) -> float:
        """Apply causal low-pass filter to single sample."""
        # Initialize filter state for this channel if needed
        if channel not in self.streaming_state:
            self.streaming_state[channel] = np.zeros(max(len(self.a), len(self.b)) - 1)

        # Apply filter to single sample
        filtered_value, self.streaming_state[channel] = lfilter(
            self.b, self.a, [value], zi=self.streaming_state[channel]
        )

        return filtered_value[0]

    def get_description(self) -> str:
        return f"Low-Pass Filter (cutoff={self.cutoff}Hz, order={self.order}, fs={self.fs:.4f}Hz)"


class BandPassFilterStrategy(FilterStrategy):
    """
    Band-pass Butterworth filter.
    Keeps frequencies within a specified range, removes everything else.
    Combines high-pass and low-pass filtering.
    """

    def __init__(
        self,
        fs: float,
        low_cutoff: float = 0.5,
        high_cutoff: float = 5.0,
        order: int = 4,
    ):
        """
        Args:
            fs: Sampling rate in Hz
            low_cutoff: Low cutoff frequency in Hz (high-pass component)
            high_cutoff: High cutoff frequency in Hz (low-pass component)
            order: Filter order (higher = sharper cutoff, but more ringing)
        """
        super().__init__(fs)
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order

        # Design Butterworth filter
        nyq = 0.5 * fs
        low_normal = low_cutoff / nyq
        high_normal = high_cutoff / nyq

        # Validate cutoff frequencies
        if high_normal >= 1.0:
            raise ValueError(
                f"High cutoff frequency {high_cutoff}Hz is too high for sampling rate {fs}Hz. "
                f"Must be less than Nyquist frequency ({nyq}Hz)."
            )
        if low_normal >= high_normal:
            raise ValueError(
                f"Low cutoff ({low_cutoff}Hz) must be less than high cutoff ({high_cutoff}Hz)."
            )

        self.b, self.a = butter(
            order, [low_normal, high_normal], btype="band", analog=False
        )

    def apply_batch(self, signal: np.ndarray) -> np.ndarray:
        """Apply causal band-pass filter to entire signal (uses lfilter)."""
        return lfilter(self.b, self.a, signal)

    def apply_streaming(self, value: float, channel: int = 0) -> float:
        """Apply causal band-pass filter to single sample."""
        # Initialize filter state for this channel if needed
        if channel not in self.streaming_state:
            self.streaming_state[channel] = np.zeros(max(len(self.a), len(self.b)) - 1)

        # Apply filter to single sample
        filtered_value, self.streaming_state[channel] = lfilter(
            self.b, self.a, [value], zi=self.streaming_state[channel]
        )

        return filtered_value[0]

    def get_description(self) -> str:
        return f"Band-Pass Filter (low={self.low_cutoff}Hz, high={self.high_cutoff}Hz, order={self.order}, fs={self.fs:.4f}Hz)"


class MovingAverageFilterStrategy(FilterStrategy):
    """
    Simple moving average filter (FIR filter).
    Very simple smoothing - averages last N samples.
    No ringing, but less effective than Butterworth for sharp cutoffs.
    """

    def __init__(self, fs: float, window_size: int = 5):
        """
        Args:
            fs: Sampling rate in Hz (for consistency, not used in computation)
            window_size: Number of samples to average
        """
        super().__init__(fs)
        self.window_size = window_size

        # FIR coefficients: uniform weights
        self.b = np.ones(window_size) / window_size
        self.a = np.array([1.0])  # No feedback (FIR filter)

    def apply_batch(self, signal: np.ndarray) -> np.ndarray:
        """Apply moving average filter to entire signal."""
        return lfilter(self.b, self.a, signal)

    def apply_streaming(self, value: float, channel: int = 0) -> float:
        """Apply moving average filter to single sample."""
        # Initialize filter state for this channel if needed
        if channel not in self.streaming_state:
            self.streaming_state[channel] = np.zeros(self.window_size - 1)

        # Apply filter to single sample
        filtered_value, self.streaming_state[channel] = lfilter(
            self.b, self.a, [value], zi=self.streaming_state[channel]
        )

        return filtered_value[0]

    def get_description(self) -> str:
        return f"Moving Average Filter (window={self.window_size} samples, fs={self.fs:.4f}Hz)"


# Factory function for easy filter creation
def create_filter(filter_type: str, fs: float, **kwargs) -> FilterStrategy:
    """
    Factory function to create filter strategies.

    Args:
        filter_type: One of 'none', 'highpass', 'lowpass', 'bandpass', 'moving_average'
        fs: Sampling rate in Hz
        **kwargs: Additional arguments passed to filter constructor

    Returns:
        FilterStrategy instance

    Examples:
        >>> # No filtering
        >>> filter = create_filter('none', fs=4.2144)

        >>> # High-pass filter (remove DC bias)
        >>> filter = create_filter('highpass', fs=4.2144, cutoff=0.5, order=4)

        >>> # Low-pass filter (smoothing)
        >>> filter = create_filter('lowpass', fs=4.2144, cutoff=5.0, order=4)

        >>> # Band-pass filter
        >>> filter = create_filter('bandpass', fs=4.2144, low_cutoff=0.5, high_cutoff=5.0, order=4)

        >>> # Moving average (simple smoothing)
        >>> filter = create_filter('moving_average', fs=4.2144, window_size=5)
    """
    filter_type = filter_type.lower()

    if filter_type == "none":
        return NoFilterStrategy(fs)
    elif filter_type == "highpass":
        return HighPassFilterStrategy(fs, **kwargs)
    elif filter_type == "lowpass":
        return LowPassFilterStrategy(fs, **kwargs)
    elif filter_type == "bandpass":
        return BandPassFilterStrategy(fs, **kwargs)
    elif filter_type == "moving_average":
        return MovingAverageFilterStrategy(fs, **kwargs)
    else:
        raise ValueError(
            f"Unknown filter type: {filter_type}. "
            f"Choose from: 'none', 'highpass', 'lowpass', 'bandpass', 'moving_average'"
        )


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("FILTER STRATEGY EXAMPLES")
    print("=" * 80)

    fs = 4.2144  # Measured sample rate

    # Test different filters
    filters = [
        create_filter("none", fs),
        create_filter("highpass", fs, cutoff=0.5, order=4),
        create_filter("lowpass", fs, cutoff=2.0, order=4),
        create_filter("bandpass", fs, low_cutoff=0.5, high_cutoff=2.0, order=4),
        create_filter("moving_average", fs, window_size=5),
    ]

    # Test signal: DC + sine wave + noise
    t = np.linspace(0, 10, int(fs * 10))
    signal = 5.0 + np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))

    print("\nApplying filters to test signal...")
    for filt in filters:
        print(f"\n{filt.get_description()}")

        # Batch mode
        filtered_batch = filt.apply_batch(signal)
        print(
            f"  Batch:     mean={filtered_batch.mean():.4f}, std={filtered_batch.std():.4f}"
        )

        # Streaming mode
        filt.reset_streaming_state()
        filtered_streaming = np.array(
            [filt.apply_streaming(val, channel=0) for val in signal]
        )
        print(
            f"  Streaming: mean={filtered_streaming.mean():.4f}, std={filtered_streaming.std():.4f}"
        )

        # Check if batch and streaming match
        diff = np.abs(filtered_batch - filtered_streaming).max()
        print(
            f"  Max difference: {diff:.6f} {'✅ MATCH' if diff < 1e-10 else '⚠️ DIFFER'}"
        )
