"""
Real-Time Streaming Inference for EMG-to-Finger Position Prediction

This script implements TRUE REAL-TIME inference by reading directly from the sensor:
- Reads sensor data directly from Arduino/serial port using pySerial
- Processes each sample as it arrives (true streaming)
- High-pass filtering uses causal lfilter (StreamingHighPassFilter)
- No lookahead or future information is used
- Sends servo commands back to Arduino

Matches main.py's simple serial communication approach.
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import time
import serial
from scipy.signal import butter, lfilter
from collections import deque
import sys
import os


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
    def __init__(
        self, model_path, scaler_path, window_size=30, device="cpu", smoothing_alpha=1.0
    ):
        self.window_size = window_size
        self.device = torch.device(device)

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Initialize model (matching inference_streaming.py exactly)
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
        # The old hardcoded value (1.0/0.03446 = 29.03 Hz) may not match your actual data
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

        # FIX: Model outputs predictions in wrong order
        # Model outputs: [thumb_tip, index, middle, ring, pinky, thumb_base]
        # Hardware needs: [thumb_tip, thumb_base, index, middle, ring, pinky]
        # Reorder indices: [0, 5, 1, 2, 3, 4]
        self.prediction_reorder = [0, 5, 1, 2, 3, 4]
        print(
            f"✓ Prediction reordering: model order [0,1,2,3,4,5] → hardware order {self.prediction_reorder}"
        )

        # Output smoothing (Exponential Moving Average)
        self.smoothing_alpha = 1.0  # 0.0 = max smoothing, 1.0 = no smoothing
        self.smoothed_output = None  # Will be initialized on first prediction

        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Scaler loaded with {self.scaler.n_features_in_} features")
        print(f"✓ Window size: {window_size}")
        print(f"✓ Output smoothing alpha: {smoothing_alpha} (lower = smoother)")

    def compute_spatial_features(self, raw_values, env_values):
        """Compute spatial features exactly as in inference_streaming.py"""
        raw_diffs = []
        env_diffs = []

        for base in range(1, 5):
            if not self.neighbors[base]:
                raw_diffs.append(0.0)
                env_diffs.append(0.0)
            else:
                raw_diff_list = [
                    raw_values[base - 1] - raw_values[n - 1]
                    for n in self.neighbors[base]
                ]
                env_diff_list = [
                    env_values[base - 1] - env_values[n - 1]
                    for n in self.neighbors[base]
                ]
                raw_diffs.append(sum(raw_diff_list) / len(raw_diff_list))
                env_diffs.append(sum(env_diff_list) / len(env_diff_list))

        return raw_diffs, env_diffs

    def process_sample(self, raw_sample):
        """
        Process a single incoming sample with true streaming behavior.
        EXACTLY matches inference_streaming.py logic.

        Args:
            raw_sample: Array of [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
                       All values are RAW (unfiltered) as they would come from sensors

        Returns:
            Scaled feature vector ready for windowing
        """
        # Extract raw sensor values
        env_values_raw = raw_sample[
            ::2
        ].copy()  # env0, env1, env2, env3 (raw from sensor)
        raw_values = raw_sample[1::2].copy()  # raw0, raw1, raw2, raw3 (unfiltered)

        # Apply streaming high-pass filter to env values (causal, online filtering)
        env_values_filtered = np.array(
            [self.highpass_filter.filter(env_values_raw[i], i) for i in range(4)]
        )

        # Compute spatial features using FILTERED env and UNFILTERED raw
        raw_diffs, env_diffs = self.compute_spatial_features(
            raw_values, env_values_filtered
        )

        # Match the notebook feature order: env0, raw0, env1, raw1, env2, raw2, env3, raw3, then diffs
        interleaved_sensors = []
        for i in range(4):
            interleaved_sensors.append(env_values_filtered[i])
            interleaved_sensors.append(raw_values[i])

        features = np.concatenate([interleaved_sensors, raw_diffs, env_diffs])

        features_scaled = self.scaler.transform(features.reshape(1, -1))[0]

        self.window_buffer.append(features_scaled)

    def predict(self):
        """
        Make a prediction using the current sliding window buffer with smoothing.

        Returns:
            Smoothed predicted finger positions (6 values) or None if buffer not full
        """
        if len(self.window_buffer) < self.window_size:
            return None

        # Create window tensor
        window_array = np.array(list(self.window_buffer))
        window_tensor = (
            torch.tensor(window_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Get prediction
        with torch.no_grad:
            prediction = self.model(window_tensor)

        # Clamp predictions to valid range [0, 1] (matching inference_streaming.py)
        prediction = torch.clamp(prediction, 0.0, 1.0)
        prediction_raw = prediction.cpu().numpy()[0]

        # FIX: Reorder predictions to match hardware servo mapping
        # Model outputs: [thumb_tip, index, middle, ring, pinky, thumb_base]
        # Reorder to:    [thumb_tip, thumb_base, index, middle, ring, pinky]
        prediction_raw = prediction_raw[self.prediction_reorder]

        # Apply exponential moving average smoothing for fluid movements
        if self.smoothed_output is None:
            # Initialize smoothed output with first prediction
            self.smoothed_output = prediction_raw.copy()
        else:
            # EMA: smoothed = alpha * new + (1 - alpha) * smoothed
            self.smoothed_output = (
                self.smoothing_alpha * prediction_raw
                + (1 - self.smoothing_alpha) * self.smoothed_output
            )

        return self.smoothed_output

    def run_realtime(self, port, baudrate=115200):
        """
        Run real-time inference from sensor data using simple serial communication.
        Matches main.py's approach with RATE CONTROL for consistent sampling.

        Args:
            port: Serial port name (e.g., "COM3", "/dev/ttyAMA0")
            baudrate: Serial port baud rate
        """
        print("\n" + "=" * 60)
        print("REAL-TIME STREAMING INFERENCE WITH RATE CONTROL")
        print("=" * 60)
        print(f"Port: {port}")
        print(f"Baudrate: {baudrate}")

        # CRITICAL: Match the training data sample rate
        target_sample_rate = 30.0  # Hz (matches Arduino 33ms delay and data collection)
        sample_interval = 1.0 / target_sample_rate

        print(
            f"🎯 Target sample rate: {target_sample_rate} Hz ({sample_interval * 1000:.1f}ms interval)"
        )
        print("   This MUST match your training data sample rate!")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Open serial port (same as main.py)
        ser = serial.Serial(
            port, baudrate, timeout=0.01
        )  # Shorter timeout for responsiveness

        start_time = time.time()
        last_display_time = start_time
        last_sample_time = start_time

        # Diagnostics
        samples_received = 0
        predictions_made = 0
        samples_processed = 0

        # Buffer for latest sensor reading
        latest_sensor_reading = None

        try:
            while True:
                current_time = time.time()

                # CONTINUOUSLY read from serial (non-blocking, keep data fresh)
                try:
                    line = ser.readline().decode().strip()

                    if line:
                        # Parse Arduino format: S4:raw,env;S3:raw,env;S2:raw,env;S1:raw,env
                        parts = line.split(";")
                        if len(parts) == 4:
                            sensor_data = {}
                            for part in parts:
                                if ":" not in part:
                                    continue
                                sensor_id, values = part.split(":")
                                raw, env = values.split(",")
                                sensor_data[sensor_id] = (float(raw), float(env))

                            # Check we got all sensors
                            if all(s in sensor_data for s in ["S1", "S2", "S3", "S4"]):
                                # Map to expected format: [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
                                # S4 = sensor 0, S3 = sensor 1, S2 = sensor 2, S1 = sensor 3
                                latest_sensor_reading = np.array(
                                    [
                                        sensor_data["S4"][1],  # env0
                                        sensor_data["S4"][0],  # raw0
                                        sensor_data["S3"][1],  # env1
                                        sensor_data["S3"][0],  # raw1
                                        sensor_data["S2"][1],  # env2
                                        sensor_data["S2"][0],  # raw2
                                        sensor_data["S1"][1],  # env3
                                        sensor_data["S1"][0],  # raw3
                                    ],
                                    dtype=np.float32,
                                )
                                samples_received += 1
                except Exception:
                    pass  # Ignore parse errors, keep trying

                # PROCESS at controlled rate (matching training data)
                time_since_last_sample = current_time - last_sample_time

                if time_since_last_sample >= sample_interval:
                    last_sample_time = current_time

                    if latest_sensor_reading is not None:
                        samples_processed += 1

                        # Process sample
                        self.process_sample(latest_sensor_reading)

                        # Make prediction (only when buffer is full)
                        prediction = self.predict()

                        if prediction is not None:
                            predictions_made += 1

                            # Convert predictions [0,1] to servo values [0,1023]
                            servo_values = (prediction * 1023).astype(int)
                            servo_values = np.clip(servo_values, 0, 1023)

                            # Send servo commands to Arduino (same as main.py)
                            msg = ",".join(map(str, servo_values)) + "\n"
                            ser.write(msg.encode())

                        time_since_last = current_time - last_display_time

                        # Display prediction every 0.5 seconds
                        if time_since_last >= 0.5:
                            elapsed = current_time - start_time
                            sample_fps = (
                                samples_received / elapsed if elapsed > 0 else 0
                            )
                            process_fps = (
                                samples_processed / elapsed if elapsed > 0 else 0
                            )
                            pred_fps = predictions_made / elapsed if elapsed > 0 else 0

                            # Calculate actual processing rate
                            # actual_rate = 1.0 / time_since_last_sample if time_since_last_sample > 0 else 0

                            print(
                                f"\n[Received: {samples_received} | Processed: {samples_processed} | Predictions: {predictions_made}]"
                            )
                            print(
                                f"Receive rate: {sample_fps:.1f} Hz | Process rate: {process_fps:.1f} Hz (target: {target_sample_rate:.1f} Hz) | Pred rate: {pred_fps:.1f} Hz"
                            )
                            print("Predictions:")
                            for i, name in enumerate(self.finger_names):
                                bar = "█" * int(prediction[i] * 20)
                                servo_val = servo_values[i]
                                print(
                                    f"  {name:<12}: {prediction[i]:.3f} {bar} (servo: {servo_val})"
                                )

                            last_display_time = current_time

        except KeyboardInterrupt:
            print("\n\nStopping real-time inference...")
            print(f"Total samples received: {samples_received}")
            print(f"Total samples processed: {samples_processed}")
            print(f"Total predictions made: {predictions_made}")
            elapsed = time.time() - start_time
            if elapsed > 0:
                print(f"Average sample rate: {samples_received / elapsed:.1f} Hz")
                print(f"Average prediction rate: {predictions_made / elapsed:.1f} Hz")

        finally:
            ser.close()
            print("✓ Serial port closed")


def main():
    # Configuration
    # model_path = "training/notebooks/best_lstm_model.pth"
    # scaler_path = "training/notebooks/scaler_inputs_lstm.pkl"
    model_path = "data/martin6/best_lstm_model.pth"
    scaler_path = "data/martin6/scaler_inputs_lstm.pkl"

    # Hardware configuration
    if sys.platform == "darwin":
        port = "/dev/tty.usbmodem11301"  # Mac
    elif sys.platform == "win32":
        port = "COM3"  # Windows
    else:
        port = "/dev/ttyAMA0"  # Raspberry Pi

    baudrate = 115200

    # Output Smoothing (for fluid hand movements)
    # Lower values = smoother movements (less jitter, slower response)
    # Higher values = more responsive (faster response, more jitter)
    # Recommended: 0.2-0.4 for smooth hand movements

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
    print(f"✓ Model loaded successfully on {device}")
    print(f"✓ Model num_layers: {inference_engine.model.num_layers}")
    print(f"✓ Model hidden_size: {inference_engine.model.hidden_size}")

    # Run real-time inference
    inference_engine.run_realtime(port=port, baudrate=baudrate)


if __name__ == "__main__":
    main()
