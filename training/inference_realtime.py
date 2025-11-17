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
from scipy.signal import butter, lfilter
from collections import deque
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rpi.src.serial_config.port_accessor import PortAccessor
from data_collection.collectors.data_collector import parse_port_event


class StreamingHighPassFilter:
    def __init__(self, fs, cutoff=0.5, order=4):
        self.fs = fs
        self.cutoff = cutoff
        self.order = order
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype="high", analog=False)
        self.zi = {i: np.zeros(max(len(self.a), len(self.b)) - 1) for i in range(4)}

    def filter(self, value, channel):
        filtered_value, self.zi[channel] = lfilter(
            self.b, self.a, [value], zi=self.zi[channel]
        )
        return filtered_value[0]


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
    def __init__(self, model_path, scaler_path, window_size=30, device="cpu"):
        self.window_size = window_size
        self.device = torch.device(device)

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Initialize model
        sensor_columns = [
            "env0",
            "raw0",
            "env1",
            "raw1",
            "env2",
            "raw2",
            "env3",
            "raw3",
            "raw_diff1",
            "raw_diff2",
            "raw_diff3",
            "raw_diff4",
            "env_diff1",
            "env_diff2",
            "env_diff3",
            "env_diff4",
        ]
        n_inputs = len(sensor_columns)
        n_outputs = 6

        self.model = LSTMModel(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
        ).to(self.device)

        # Load model weights
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Initialize sliding window buffer
        self.window_buffer = deque(maxlen=window_size)

        # Initialize streaming high-pass filter for env channels
        # ⚠️ CRITICAL: This MUST match the actual training data sample rate!
        # Run the training notebook's sampling rate analysis cell to get the measured value
        fs = 30.0  # Hz - Updated to match Arduino (33ms delay) and data collection (30 Hz target)
        print(f"✓ Using sampling rate: {fs} Hz for high-pass filter")
        self.highpass_filter = StreamingHighPassFilter(fs, cutoff=0.5, order=4)

        # Define neighbor relationships for spatial features (matching notebook)
        self.neighbors = {1: [], 2: [3], 3: [2, 4], 4: [3]}

        # Finger names for display
        self.finger_names = [
            "thumb_tip",
            "thumb_base",
            "index",
            "middle",
            "ring",
            "pinky",
        ]

        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Scaler loaded with {self.scaler.n_features_in_} features")
        print(f"✓ Window size: {window_size}")

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

    def process_sample(self, raw_sample):
        """
        Process a single sensor sample and add it to the sliding window.

        TRUE STREAMING BEHAVIOR:
        - Applies causal high-pass filter to env channels sample-by-sample
        - Computes spatial features from current sample only
        - No lookahead or future information used

        Args:
            raw_sample: Array of 8 sensor values [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
        """
        # Extract raw env and raw values
        env_values_raw = raw_sample[::2]  # [env0, env1, env2, env3]
        raw_values = raw_sample[1::2]  # [raw0, raw1, raw2, raw3]

        # Apply streaming high-pass filter to env channels
        env_values_filtered = np.array(
            [self.highpass_filter.filter(env_values_raw[i], i) for i in range(4)]
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

        features = np.concatenate(
            [
                interleaved_sensors,  # [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
                raw_diffs,  # [raw_diff1, raw_diff2, raw_diff3, raw_diff4]
                env_diffs,  # [env_diff1, env_diff2, env_diff3, env_diff4]
            ]
        )

        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))[0]

        # Add to sliding window
        self.window_buffer.append(features_scaled)

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
