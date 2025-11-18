"""
Real-Time Streaming Inference for EMG-to-Finger Position Prediction

This script implements TRUE REAL-TIME inference by reading directly from the sensor
using the same PortAccessor interface as data collection:
- Reads sensor data directly from Arduino/serial port
- Processes each sample as it arrives (true streaming)
- High-pass filtering uses causal lfilter (StreamingHighPassFilter)
- No lookahead or future information is used
- Displays predictions in real-time

Key Components:
1. PortAccessor: Serial port interface (same as data collection)
2. StreamingHighPassFilter: Causal IIR filter with state preservation
3. LSTMModel: 3-layer LSTM with skip connections
4. RealtimeInference: Online processing with sliding window buffer
5. Real-time visualization of predictions

NOTE: Model should be trained with causal filtering for best results.
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import time
from collections import deque
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rpi.src.serial_config.port_accessor import PortAccessor
from data_collection.collectors.data_collector import parse_port_event
from training.filter_strategies import create_filter


class StreamingStandardScaler:
    """
    Online/streaming version of StandardScaler that adapts to incoming data.

    Uses Welford's algorithm for numerically stable online computation of mean and variance.
    """

    def __init__(
        self, n_features, warmup_samples=100, adaptation_rate=0.01, use_pretrained=True
    ):
        """
        Args:
            n_features: Number of features to scale
            warmup_samples: Number of samples to collect before starting to scale
            adaptation_rate: How fast to adapt to new data (0 = no adaptation, 1 = only use new data)
            use_pretrained: If True, initialize with pretrained scaler stats and adapt slowly
        """
        self.n_features = n_features
        self.warmup_samples = warmup_samples
        self.adaptation_rate = adaptation_rate
        self.use_pretrained = use_pretrained

        # Statistics
        self.mean_ = np.zeros(n_features)
        self.var_ = np.ones(n_features)
        self.scale_ = np.ones(n_features)

        # Online statistics tracking (Welford's algorithm)
        self.n_samples_seen_ = 0
        self.M2_ = np.zeros(n_features)

        # Warmup buffer
        self.warmup_buffer = []
        self.is_warmed_up = False

    def initialize_from_pretrained(self, pretrained_scaler):
        """Initialize statistics from a pretrained StandardScaler."""
        if hasattr(pretrained_scaler, "mean_"):
            self.mean_ = pretrained_scaler.mean_.copy()
        if hasattr(pretrained_scaler, "var_"):
            self.var_ = pretrained_scaler.var_.copy()
        if hasattr(pretrained_scaler, "scale_"):
            self.scale_ = pretrained_scaler.scale_.copy()

        if self.use_pretrained:
            self.is_warmed_up = True
            print(
                f"  Initialized with pretrained statistics (mean range: [{self.mean_.min():.3f}, {self.mean_.max():.3f}])"
            )
            print(
                f"  Adaptation rate: {self.adaptation_rate} (will slowly adapt to new data)"
            )

    def partial_fit(self, X):
        """Update statistics with a new sample using Welford's algorithm."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        for sample in X:
            # Warmup phase
            if not self.is_warmed_up:
                self.warmup_buffer.append(sample)

                if len(self.warmup_buffer) >= self.warmup_samples:
                    warmup_data = np.array(self.warmup_buffer)

                    if not self.use_pretrained:
                        self.mean_ = warmup_data.mean(axis=0)
                        self.var_ = warmup_data.var(axis=0)
                        self.scale_ = np.sqrt(self.var_)
                        self.n_samples_seen_ = len(warmup_data)
                        self.M2_ = self.var_ * (self.n_samples_seen_ - 1)
                        print(f"  Warmup complete with {len(warmup_data)} samples")
                    else:
                        self.n_samples_seen_ = len(warmup_data)
                        self.M2_ = self.var_ * max(1, self.n_samples_seen_ - 1)
                        print(
                            f"  Warmup complete with {len(warmup_data)} samples (using pretrained stats)"
                        )

                    self.is_warmed_up = True
                    self.warmup_buffer = []

                continue

            # Online update with adaptation rate
            self.n_samples_seen_ += 1
            delta = sample - self.mean_

            if self.adaptation_rate > 0:
                effective_n = min(1.0 / self.adaptation_rate, self.n_samples_seen_)
                self.mean_ = self.mean_ + delta / effective_n
                delta2 = sample - self.mean_
                self.M2_ = self.M2_ + delta * delta2

                if self.n_samples_seen_ > 1:
                    new_var = self.M2_ / (effective_n - 1)
                    self.var_ = (
                        1 - self.adaptation_rate
                    ) * self.var_ + self.adaptation_rate * new_var
                    self.scale_ = np.sqrt(np.maximum(self.var_, 1e-8))

    def transform(self, X):
        """Scale features using current mean and std. Also updates statistics."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Update statistics
        self.partial_fit(X)

        # Scale using current statistics
        if self.is_warmed_up:
            X_scaled = (X - self.mean_) / np.maximum(self.scale_, 1e-8)
        else:
            if self.use_pretrained and np.any(self.scale_ != 1.0):
                X_scaled = (X - self.mean_) / np.maximum(self.scale_, 1e-8)
            else:
                X_scaled = X

        return X_scaled

    def get_stats(self):
        """Return current statistics for debugging/monitoring."""
        return {
            "mean": self.mean_.copy(),
            "std": self.scale_.copy(),
            "var": self.var_.copy(),
            "n_samples": self.n_samples_seen_,
            "is_warmed_up": self.is_warmed_up,
        }


class LSTMModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = n_inputs if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
            )

        if num_layers > 1:
            self.dropout = nn.Dropout(dropout)

        self.input_projection = (
            nn.Linear(n_inputs, hidden_size) if n_inputs != hidden_size else None
        )
        self.fc = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        skip_connections = []
        out, _ = self.lstm_layers[0](x)

        if self.input_projection is not None:
            x_projected = self.input_projection(x)
        else:
            x_projected = x

        skip_connections.append(x_projected)

        for i in range(1, self.num_layers):
            if self.num_layers > 1:
                out = self.dropout(out)
            out = out + skip_connections[-1]
            out, _ = self.lstm_layers[i](out)
            skip_connections.append(out)

        last_output = out[:, -1, :]
        output = self.fc(last_output)
        return output


class RealtimeInference:
    def __init__(
        self,
        model_path,
        scaler_path,
        window_size=30,
        device="cpu",
        feature_window=10,
        use_online_scaling=False,
        online_adaptation_rate=0.001,  # Very slow adaptation (0.1% per sample)
        online_warmup_samples=500,  # More samples for stable initial statistics
    ):
        """
        Args:
            model_path: Path to model checkpoint
            scaler_path: Path to pretrained scaler
            window_size: Number of samples in sliding window
            device: 'cpu' or 'cuda'
            feature_window: Window size for EMG features
            use_online_scaling: If True, scaler adapts to incoming data
            online_adaptation_rate: How fast to adapt (0.001-0.1, lower=more stable)
            online_warmup_samples: Samples to collect before adapting
        """
        self.window_size = window_size
        self.device = torch.device(device)
        self.feature_window = feature_window
        self.use_online_scaling = use_online_scaling
        self.online_adaptation_rate = online_adaptation_rate
        self.online_warmup_samples = online_warmup_samples

        # Load checkpoint to determine model configuration
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Extract state dict and hyperparameters from checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            hyperparams = checkpoint.get("hyperparameters", {})
        else:
            state_dict = checkpoint
            hyperparams = {}

        # Load configuration from checkpoint (matching training settings)
        self.filter_type = hyperparams.get("filter_type", "none")
        self.filter_config = hyperparams.get("filter_config", {})
        self.use_scaling = hyperparams.get("use_scaling", False)
        fs = hyperparams.get(
            "sampling_rate", 30.0
        )  # Default to 30 Hz if not in checkpoint

        print(f"\n{'=' * 70}")
        print("LOADED TRAINING CONFIGURATION FROM CHECKPOINT")
        print(f"{'=' * 70}")
        print(f"  Filter type: {self.filter_type}")
        print(f"  Filter config: {self.filter_config}")
        print(f"  Sampling rate: {fs:.4f} Hz")
        print(f"  Scaling: {'ENABLED' if self.use_scaling else 'DISABLED'}")
        print(f"{'=' * 70}\n")

        # Initialize filter using strategy pattern
        self.filter = create_filter(self.filter_type, fs, **self.filter_config)
        print(f"Filter strategy: {self.filter.get_description()}")

        # Detect number of input features from model weights
        # The first LSTM layer's input weight shape is (4*hidden_size, n_inputs)
        first_lstm_weight = state_dict["lstm_layers.0.weight_ih_l0"]
        n_inputs = first_lstm_weight.shape[1]  # Extract n_inputs from weight dimensions
        n_outputs = 6

        print(f"\nDetected model with {n_inputs} input features")

        # Determine if EMG features are used
        self.use_emg_features = n_inputs == 32  # 8 base + 8 spatial + 16 EMG

        if self.use_emg_features:
            print("EMG FEATURES ENABLED (32 features: 8 base + 8 spatial + 16 EMG)")
            # Initialize buffers for EMG feature computation
            self.raw_buffers = {i: deque(maxlen=feature_window) for i in range(4)}
        else:
            print("EMG features disabled (16 features: 8 base + 8 spatial)")

        # Load pretrained scaler
        pretrained_scaler = joblib.load(scaler_path)

        # Initialize scaler (online or fixed) - only if scaling is enabled
        if self.use_scaling:
            if use_online_scaling:
                print(f"\n{'=' * 70}")
                print("ONLINE ADAPTIVE SCALING ENABLED")
                print(f"{'=' * 70}")
                self.scaler = StreamingStandardScaler(
                    n_features=n_inputs,
                    warmup_samples=online_warmup_samples,
                    adaptation_rate=online_adaptation_rate,
                    use_pretrained=True,  # Start with pretrained stats
                )
                self.scaler.initialize_from_pretrained(pretrained_scaler)
                print(f"  Warmup samples: {online_warmup_samples}")
                print(f"  Adaptation rate: {online_adaptation_rate}")
                print("  -> Scaler will adapt to distribution shifts in real-time")
                print(f"{'=' * 70}\n")
            else:
                print(f"\n{'=' * 70}")
                print("FIXED PRETRAINED SCALING")
                print(f"{'=' * 70}")
                self.scaler = pretrained_scaler
                print("  Using fixed pretrained scaler (no adaptation)")
                print(f"{'=' * 70}\n")
        else:
            print(f"\n{'=' * 70}")
            print("SCALING DISABLED")
            print(f"{'=' * 70}")
            print("  Using raw feature values (no scaling)")
            print("  Model was trained without scaling")
            print(f"{'=' * 70}\n")
            self.scaler = None

        # Initialize model with detected n_inputs
        self.model = LSTMModel(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Initialize sliding window buffer
        self.window_buffer = deque(maxlen=window_size)

        # Define neighbor relationships for spatial features (MUST match training notebook!)
        # Training uses: {1: [2], 2: [1, 4], 3: [4], 4: [2, 3]}
        self.neighbors = {1: [2], 2: [1, 4], 3: [4], 4: [2, 3]}

        # Finger names for display
        self.finger_names = [
            "thumb_tip",
            "thumb_base",
            "index",
            "middle",
            "ring",
            "pinky",
        ]

        print(f"OK - Model loaded on {self.device}")
        if self.use_scaling:
            # StreamingStandardScaler uses n_features, pretrained scaler uses n_features_in_
            n_features = getattr(
                self.scaler,
                "n_features",
                getattr(self.scaler, "n_features_in_", n_inputs),
            )
            print(f"OK - Scaler loaded with {n_features} features (WILL BE APPLIED)")
        else:
            print("OK - Scaler NOT applied (model trained without scaling)")
        print(f"OK - Window size: {window_size}")

    def compute_spatial_features(self, raw_values, env_values):
        """Compute spatial features exactly as in the notebook."""
        raw_diffs = []
        env_diffs = []

        for base in range(1, 5):
            if not self.neighbors[base]:
                raw_diffs.append(0.0)
                env_diffs.append(0.0)
            else:
                raw_diff_sum = sum(
                    raw_values[base - 1] - raw_values[n - 1]
                    for n in self.neighbors[base]
                )
                env_diff_sum = sum(
                    env_values[base - 1] - env_values[n - 1]
                    for n in self.neighbors[base]
                )
                raw_diffs.append(raw_diff_sum / len(self.neighbors[base]))
                env_diffs.append(env_diff_sum / len(self.neighbors[base]))

        return np.array(raw_diffs), np.array(env_diffs)

    def compute_emg_features(self, signal_buffer):
        """
        Compute EMG features from a signal buffer (for streaming inference).

        Features computed:
        - RMS (Root Mean Square): Signal power/intensity
        - MAV (Mean Absolute Value): Average amplitude
        - ZC (Zero Crossings): Frequency content indicator
        - WL (Waveform Length): Signal complexity

        Args:
            signal_buffer: Deque containing recent signal samples

        Returns:
            Array of 4 EMG features [RMS, MAV, ZC, WL]
        """
        if len(signal_buffer) < 2:
            # Not enough data yet - return zeros
            return np.zeros(4)

        signal = np.array(signal_buffer)

        # RMS: Root Mean Square (signal power)
        rms = np.sqrt(np.mean(signal**2))

        # MAV: Mean Absolute Value (average amplitude)
        mav = np.mean(np.abs(signal))

        # ZC: Zero Crossings (frequency content indicator)
        # Count sign changes with small threshold to avoid noise
        threshold = 0.01
        zc = 0
        for i in range(len(signal) - 1):
            if abs(signal[i]) > threshold or abs(signal[i + 1]) > threshold:
                if signal[i] * signal[i + 1] < 0:  # Sign change
                    zc += 1

        # WL: Waveform Length (signal complexity)
        wl = np.sum(np.abs(np.diff(signal)))

        return np.array([rms, mav, zc, wl])

    def process_sample(self, raw_sample):
        """
        Process a single sensor sample and add it to the sliding window.

        TRUE STREAMING BEHAVIOR:
        - Applies filter to env channels (if filter_type != 'none')
        - Computes spatial features from current sample only
        - Optionally computes EMG features if model uses them
        - Applies scaling only if use_scaling is True
        - No lookahead or future information used

        Args:
            raw_sample: Array of 8 sensor values [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
        """
        # Extract raw env and raw values
        env_values_raw = raw_sample[::2]  # [env0, env1, env2, env3]
        raw_values = raw_sample[1::2]  # [raw0, raw1, raw2, raw3]

        # Apply filter to env channels ONLY if filter_type != 'none'
        # This matches the training configuration
        env_values_filtered = np.array(
            [
                self.filter.apply_streaming(env_values_raw[i], channel=i)
                for i in range(4)
            ]
        )

        # Compute spatial features
        raw_diffs, env_diffs = self.compute_spatial_features(
            raw_values, env_values_filtered
        )

        # Create feature vector in same order as notebook
        interleaved_sensors = []
        for i in range(4):
            interleaved_sensors.append(env_values_filtered[i])
            interleaved_sensors.append(raw_values[i])

        # Base features: env0, raw0, env1, raw1, env2, raw2, env3, raw3
        # Spatial features: raw_diff1-4, env_diff1-4
        features = np.concatenate(
            [
                interleaved_sensors,  # 8 features
                raw_diffs,  # 4 features
                env_diffs,  # 4 features
            ]
        )

        # Add EMG features if the model uses them
        if self.use_emg_features:
            # Update raw signal buffers
            for i in range(4):
                self.raw_buffers[i].append(raw_values[i])

            # Compute EMG features for each channel
            emg_features = []
            for i in range(4):
                channel_emg = self.compute_emg_features(self.raw_buffers[i])
                emg_features.extend(channel_emg)  # 4 features per channel

            # Append EMG features (16 additional features: 4 features x 4 channels)
            features = np.concatenate([features, np.array(emg_features)])

        # Apply scaling ONLY if use_scaling is True (matching training)
        if self.use_scaling:
            features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
            self.window_buffer.append(features_scaled)
        else:
            # No scaling - use raw features (as during training)
            self.window_buffer.append(features)

    def predict(self):
        """
        Make a prediction using the current sliding window buffer.

        Returns:
            Predicted finger positions (6 values) or None if buffer not full
        """
        if len(self.window_buffer) < self.window_size:
            return None

        # Create window tensor
        window_array = np.array(list(self.window_buffer))  # (window_size, n_features)
        window_tensor = (
            torch.tensor(window_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Get prediction
        with torch.no_grad():
            prediction = self.model(window_tensor).cpu().numpy()[0]

        # Clamp to valid range [0, 1]
        prediction = np.clip(prediction, 0.0, 1.0)

        return prediction

    def run_realtime(self, port="MOCK", baudrate=115200):
        """
        Run real-time inference from sensor data.

        Args:
            port: Serial port name (or "MOCK" for testing)
            baudrate: Serial port baud rate
        """
        print("\n" + "=" * 60)
        print("REAL-TIME STREAMING INFERENCE")
        print("=" * 60)
        print(f"Port: {port}")
        print(f"Baudrate: {baudrate}")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Open serial port
        pa = PortAccessor(port=port, baudrate=baudrate)
        pa.open()
        subscription = pa.subscribe(max_queue=20)

        start_time = time.time()
        last_display_time = start_time

        # Diagnostics
        samples_received = 0
        predictions_made = 0

        try:
            while True:
                # Read sensor data from queue
                try:
                    if not subscription.queue.empty():
                        payload = subscription.queue.get_nowait()
                        sensor_values = parse_port_event(payload)

                        samples_received += 1

                        # Convert to numeric
                        if len(sensor_values) == 8:
                            sensor_array = np.array(
                                [float(v) for v in sensor_values], dtype=np.float32
                            )

                            # Process sample
                            self.process_sample(sensor_array)

                            # Make prediction (only when buffer is full)
                            prediction = self.predict()

                            if prediction is not None:
                                predictions_made += 1

                                current_time = time.time()
                                time_since_last = current_time - last_display_time

                                # Display prediction every 0.5 seconds instead of every 10 samples
                                if time_since_last >= 0.5:
                                    elapsed = current_time - start_time
                                    sample_fps = (
                                        samples_received / elapsed if elapsed > 0 else 0
                                    )
                                    pred_fps = (
                                        predictions_made / elapsed if elapsed > 0 else 0
                                    )

                                    print(
                                        f"\n[Samples: {samples_received} | Predictions: {predictions_made}]"
                                    )
                                    print(
                                        f"Sample FPS: {sample_fps:.1f} | Prediction FPS: {pred_fps:.1f}"
                                    )
                                    print(f"Queue size: {subscription.queue.qsize()}")
                                    print("Predictions:")
                                    for i, name in enumerate(self.finger_names):
                                        bar = "█" * int(prediction[i] * 20)
                                        print(
                                            f"  {name:<12}: {prediction[i]:.3f} {bar}"
                                        )

                                    last_display_time = current_time

                except Exception as e:
                    print(f"Error processing sample: {e}")
                    import traceback

                    traceback.print_exc()

                # No sleep - process as fast as data arrives!

        except KeyboardInterrupt:
            print("\n\nStopping real-time inference...")
            print(f"Total samples received: {samples_received}")
            print(f"Total predictions made: {predictions_made}")
            elapsed = time.time() - start_time
            if elapsed > 0:
                print(f"Average sample rate: {samples_received / elapsed:.1f} Hz")
                print(f"Average prediction rate: {predictions_made / elapsed:.1f} Hz")

        finally:
            pa.close()
            print("✓ Port closed")


def main():
    # Configuration
    model_path = "training/notebooks/best_lstm_model.pth"
    scaler_path = "training/notebooks/scaler_inputs_lstm.pkl"
    port = "COM3"  # Change to actual port like "COM3" or "/dev/ttyUSB0"
    baudrate = 115200

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return

    # Initialize real-time inference
    print("Initializing real-time inference engine...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_engine = RealtimeInference(
        model_path=model_path, scaler_path=scaler_path, window_size=30, device=device
    )

    # Run real-time inference
    inference_engine.run_realtime(port=port, baudrate=baudrate)


if __name__ == "__main__":
    main()
