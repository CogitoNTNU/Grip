import torch
import numpy as np
import pandas as pd
import joblib
import time
import glob
import os
from scipy.signal import butter, lfilter
from collections import deque
import sys
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting
import serial as pyserial  # For hardware communication (renamed to avoid conflicts)


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


class LSTMModel(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create individual LSTM layers for skip connections
        self.lstm_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            input_size = n_inputs if i == 0 else hidden_size
            self.lstm_layers.append(
                torch.nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
            )

        # Dropout layers between LSTM layers
        if num_layers > 1:
            self.dropout = torch.nn.Dropout(dropout)

        # Projection layer to match dimensions for skip connections
        self.input_projection = (
            torch.nn.Linear(n_inputs, hidden_size) if n_inputs != hidden_size else None
        )

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # Store skip connections
        skip_connections = []

        # First LSTM layer
        out, _ = self.lstm_layers[0](x)

        # Project input to match hidden_size for skip connection
        if self.input_projection is not None:
            x_projected = self.input_projection(x)
        else:
            x_projected = x

        skip_connections.append(x_projected)

        # Remaining LSTM layers with skip connections
        for i in range(1, self.num_layers):
            if self.num_layers > 1:
                out = self.dropout(out)

            # Add skip connection from previous layer
            out = out + skip_connections[-1]

            # Pass through LSTM layer
            out, _ = self.lstm_layers[i](out)

            # Store current output for next skip connection
            skip_connections.append(out)

        # Take the last time step output
        last_output = out[:, -1, :]

        # Pass through fully connected layer
        output = self.fc(last_output)
        return output


class DebugInference:
    def __init__(
        self, model_path, scaler_path, window_size=30, device="cpu", smoothing_alpha=0.7
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

        return features_scaled

    def predict(self):
        """Make a prediction using the current sliding window buffer with smoothing."""
        if len(self.window_buffer) < self.window_size:
            return None

        # Create window tensor
        window_array = np.array(list(self.window_buffer))
        window_tensor = (
            torch.tensor(window_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Get prediction
        with torch.no_grad():
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
            # Lower alpha = more smoothing (slower response, less jitter)
            # Higher alpha = less smoothing (faster response, more jitter)
            self.smoothed_output = (
                self.smoothing_alpha * prediction_raw
                + (1 - self.smoothing_alpha) * self.smoothed_output
            )

        return self.smoothed_output

    def load_data(self, data_dir):
        """
        Load test data WITHOUT pre-filtering for true streaming simulation.
        EXACTLY matches inference_streaming.py load_test_data() function.

        The data is loaded in its raw form as it would come from sensors.
        Filtering will be applied sample-by-sample during streaming using
        the causal StreamingHighPassFilter.
        """
        print(f"\nLoading data from {data_dir}...")

        csv_files = glob.glob(os.path.join(data_dir, "integrated_data_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        print(f"Found {len(csv_files)} CSV files")

        df_list = []
        for f in csv_files:
            df_list.append(pd.read_csv(f))

        df = pd.concat(df_list, ignore_index=True)

        # Convert to numeric (matching inference_streaming.py)
        numeric_columns = [
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
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df_clean = df.dropna(subset=numeric_columns)

        # NO FILTERING APPLIED HERE - data remains raw as from sensors
        # Filtering happens online in process_sample() using causal lfilter

        print(f"✓ Loaded {len(df_clean)} samples")

        # DEBUG: Check data ranges (matching inference_streaming.py)
        sensor_columns = [
            "env0",
            "raw0",
            "env1",
            "raw1",
            "env2",
            "raw2",
            "env3",
            "raw3",
        ]
        print("\nData ranges after loading:")
        for col in sensor_columns:
            vals = df_clean[col].values
            print(
                f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}"
            )

        return df_clean

    def simulate_realtime_with_hardware(
        self,
        df,
        port,
        baudrate=115200,
        start_idx=1000,
        num_samples=500,
        playback_speed=1.0,
        show_plot=True,
    ):
        """
        Simulate real-time inference from CSV data AND send commands to actual hardware.
        This is the BEST debugging mode: use recorded data but control the real hand.

        Args:
            df: DataFrame with sensor and finger data
            port: Serial port for Arduino (e.g., "COM3")
            baudrate: Serial port baud rate
            start_idx: Starting sample index
            num_samples: Number of samples to process
            playback_speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed)
            show_plot: Whether to show real-time plot
        """
        print("\n" + "=" * 80)
        print("CSV REPLAY + HARDWARE CONTROL MODE")
        print("=" * 80)
        print("Data source: martin6/test")
        print(f"Hardware port: {port}")
        print(f"Starting at sample: {start_idx}")
        print(f"Processing {num_samples} samples")
        print(f"Playback speed: {playback_speed}x")
        print(f"Live plot: {'ENABLED' if show_plot else 'DISABLED'}")
        print("Press Ctrl+C to stop")
        print("=" * 80 + "\n")

        # Open serial port to control hardware
        try:
            ser = pyserial.Serial(port, baudrate, timeout=0.1)
            print(f"✓ Connected to {port}")
        except Exception as e:
            print(f"✗ Failed to connect to {port}: {e}")
            print("\nMake sure:")
            print(f"  1. Arduino is connected to {port}")
            print("  2. No other program is using the port")
            print("  3. Port name is correct")
            return

        sensor_cols = ["env0", "raw0", "env1", "raw1", "env2", "raw2", "env3", "raw3"]
        finger_cols = ["thumb_tip", "thumb_base", "index", "middle", "ring", "pinky"]

        end_idx = min(start_idx + num_samples, len(df))

        # Storage
        servo_commands = []
        ground_truth = []
        predictions = []

        start_time = time.time()
        last_display_time = start_time
        samples_processed = 0
        predictions_made = 0

        # Simulate sensor sample rate (approximately 29 Hz based on training data)
        sample_interval = (1.0 / 29.0) / playback_speed  # seconds per sample

        # Setup real-time plotting
        if show_plot:
            plt.ion()  # Turn on interactive mode
            fig, axes = plt.subplots(6, 1, figsize=(12, 10))
            fig.suptitle(
                "CSV Replay + Hardware Control - Debug Monitor",
                fontsize=14,
                fontweight="bold",
            )

            # Rolling window for plot (show last 100 samples)
            plot_window_size = 100
            plot_data = {
                "ground_truth": deque(maxlen=plot_window_size),
                "predictions": deque(maxlen=plot_window_size),
                "sample_indices": deque(maxlen=plot_window_size),
            }

            lines_true = []
            lines_pred = []

            for i, name in enumerate(self.finger_names):
                ax = axes[i]
                (line_true,) = ax.plot([], [], "b-", label="Ground Truth", linewidth=2)
                (line_pred,) = ax.plot(
                    [], [], "r--", label="Predicted (Sent to Hand)", linewidth=2
                )
                lines_true.append(line_true)
                lines_pred.append(line_pred)

                ax.set_ylabel(name, fontsize=10)
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)

                if i == len(self.finger_names) - 1:
                    ax.set_xlabel("Sample Index", fontsize=10)

            plt.tight_layout()
            plt.show(block=False)

        try:
            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]

                # Get sensor values from CSV (simulating sensor input)
                sensor_values = row[sensor_cols].values.astype(np.float32)

                # Get ground truth finger positions
                true_fingers = row[finger_cols].values.astype(np.float32)

                # Process sample (same as real-time)
                self.process_sample(sensor_values)
                samples_processed += 1

                # Make prediction
                prediction = self.predict()

                if prediction is not None:
                    predictions_made += 1

                    # Convert to servo values [0, 1023]
                    servo_values = ((prediction) * 1023).astype(int)
                    servo_values = np.clip(servo_values, 0, 1023)

                    if predictions_made == 1:
                        print("\n" + "=" * 80)
                        print("PREDICTION REORDERING APPLIED")
                        print("=" * 80)
                        print(
                            "Model output order: [thumb_tip, index, middle, ring, pinky, thumb_base]"
                        )
                        print(
                            "Reordered to:       [thumb_tip, thumb_base, index, middle, ring, pinky]"
                        )
                        print("\nServo mapping (after reordering):")
                        for i, name in enumerate(self.finger_names):
                            print(f"  Servo {i} ({name:<12}) = {servo_values[i]}")
                        print("=" * 80 + "\n")

                    # *** SEND TO ACTUAL HARDWARE ***
                    msg = ",".join(map(str, servo_values)) + "\n"
                    ser.write(msg.encode())

                    # Store for analysis
                    servo_commands.append(servo_values)
                    ground_truth.append(true_fingers)
                    predictions.append(prediction)

                    # Update plot
                    if show_plot:
                        plot_data["ground_truth"].append(true_fingers)
                        plot_data["predictions"].append(prediction)
                        plot_data["sample_indices"].append(idx)

                        if len(plot_data["sample_indices"]) > 1:
                            indices = list(plot_data["sample_indices"])
                            true_array = np.array(plot_data["ground_truth"])
                            pred_array = np.array(plot_data["predictions"])

                            for i in range(6):
                                lines_true[i].set_data(indices, true_array[:, i])
                                lines_pred[i].set_data(indices, pred_array[:, i])
                                axes[i].set_xlim(indices[0], indices[-1])

                            plt.pause(0.001)  # Update plot

                    # Display every 0.5 seconds
                    current_time = time.time()
                    time_since_last = current_time - last_display_time

                    if time_since_last >= 0.5:
                        elapsed = current_time - start_time
                        sample_fps = samples_processed / elapsed if elapsed > 0 else 0
                        pred_fps = predictions_made / elapsed if elapsed > 0 else 0

                        print(f"\n{'=' * 80}")
                        print(
                            f"[Samples: {samples_processed}/{num_samples} | Predictions: {predictions_made}]"
                        )
                        print(
                            f"Sample FPS: {sample_fps:.1f} | Prediction FPS: {pred_fps:.1f}"
                        )
                        print("\nGround Truth vs Predicted (Sent to Hardware):")
                        print("-" * 80)
                        print(
                            f"{'Finger':<12} | {'True':<6} {'Bar':<15} | {'Pred':<6} {'Bar':<15} | {'Servo':>5} | {'Error':>6}"
                        )
                        print("-" * 80)

                        for i, name in enumerate(self.finger_names):
                            true_val = true_fingers[i]
                            pred_val = prediction[i]
                            servo_val = servo_values[i]
                            error = abs(true_val - pred_val)

                            true_bar = "█" * int(true_val * 15)
                            pred_bar = "▓" * int(pred_val * 15)

                            print(
                                f"{name:<12} | {true_val:6.3f} {true_bar:<15} | {pred_val:6.3f} {pred_bar:<15} | {servo_val:4d} | {error:.3f}"
                            )

                        last_display_time = current_time

                # Sleep to simulate real-time data arrival
                time.sleep(sample_interval)

        except KeyboardInterrupt:
            print("\n\nStopping simulation...")

        finally:
            # Close serial port
            ser.close()
            print(f"\n✓ Serial port {port} closed")

            # Close plot if it was shown
            if show_plot:
                plt.ioff()
                print("Plot window will remain open. Close it manually to continue.")

            # Calculate overall statistics
            if predictions:
                predictions_array = np.array(predictions)
                ground_truth_array = np.array(ground_truth)

                mae = np.mean(np.abs(predictions_array - ground_truth_array), axis=0)
                overall_mae = np.mean(mae)

                print("\n" + "=" * 80)
                print("SIMULATION SUMMARY")
                print("=" * 80)
                print(f"Total samples processed: {samples_processed}")
                print(f"Total predictions made: {predictions_made}")
                print("\nMean Absolute Error (MAE) per finger:")
                for i, name in enumerate(self.finger_names):
                    print(f"  {name:<12}: {mae[i]:.4f}")
                print(f"\nOverall MAE: {overall_mae:.4f}")
                print("=" * 80)

    def simulate_realtime(
        self, df, start_idx=1000, num_samples=500, playback_speed=1.0, show_plot=True
    ):
        """
        Simulate real-time inference from recorded data.

        Args:
            df: DataFrame with sensor and finger data
            start_idx: Starting sample index
            num_samples: Number of samples to process
            playback_speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed)
            show_plot: Whether to show real-time plot
        """
        print("\n" + "=" * 80)
        print("SIMULATED REAL-TIME INFERENCE (Debug Mode)")
        print("=" * 80)
        print("Data source: martin3/raw")
        print(f"Starting at sample: {start_idx}")
        print(f"Processing {num_samples} samples")
        print(f"Playback speed: {playback_speed}x")
        print(f"Live plot: {'ENABLED' if show_plot else 'DISABLED'}")
        print("Press Ctrl+C to stop")
        print("=" * 80 + "\n")

        sensor_cols = ["env0", "raw0", "env1", "raw1", "env2", "raw2", "env3", "raw3"]
        finger_cols = ["thumb_tip", "thumb_base", "index", "middle", "ring", "pinky"]

        end_idx = min(start_idx + num_samples, len(df))

        # Simulated serial output buffer (for debugging)
        servo_commands = []
        ground_truth = []
        predictions = []

        start_time = time.time()
        last_display_time = start_time
        samples_processed = 0
        predictions_made = 0

        # Simulate sensor sample rate (approximately 29 Hz based on training data)
        sample_interval = (1.0 / 29.0) / playback_speed  # seconds per sample

        # Setup real-time plotting
        if show_plot:
            plt.ion()  # Turn on interactive mode
            fig, axes = plt.subplots(6, 1, figsize=(12, 10))
            fig.suptitle("Real-Time Inference Monitor", fontsize=14, fontweight="bold")

            # Rolling window for plot (show last 100 samples)
            plot_window_size = 100
            plot_data = {
                "ground_truth": deque(maxlen=plot_window_size),
                "predictions": deque(maxlen=plot_window_size),
                "sample_indices": deque(maxlen=plot_window_size),
            }

            lines_true = []
            lines_pred = []

            for i, name in enumerate(self.finger_names):
                ax = axes[i]
                (line_true,) = ax.plot([], [], "b-", label="Ground Truth", linewidth=2)
                (line_pred,) = ax.plot([], [], "r--", label="Predicted", linewidth=2)
                lines_true.append(line_true)
                lines_pred.append(line_pred)

                ax.set_ylabel(name, fontsize=10)
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)

                if i == len(self.finger_names) - 1:
                    ax.set_xlabel("Sample Index", fontsize=10)

            plt.tight_layout()
            plt.show(block=False)

        try:
            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]

                # Get sensor values (simulating Arduino output)
                sensor_values = row[sensor_cols].values.astype(np.float32)

                # Get ground truth finger positions
                true_fingers = row[finger_cols].values.astype(np.float32)

                # Process sample (same as real-time)
                self.process_sample(sensor_values)
                samples_processed += 1

                # Make prediction
                prediction = self.predict()

                if prediction is not None:
                    predictions_made += 1

                    # Convert to servo values [0, 1023]
                    servo_values = (prediction * 1023).astype(int)
                    servo_values = np.clip(servo_values, 0, 1023)

                    # Store for analysis
                    servo_commands.append(servo_values)
                    ground_truth.append(true_fingers)
                    predictions.append(prediction)

                    # Update plot
                    if show_plot:
                        plot_data["ground_truth"].append(true_fingers)
                        plot_data["predictions"].append(prediction)
                        plot_data["sample_indices"].append(idx)

                        if len(plot_data["sample_indices"]) > 1:
                            indices = list(plot_data["sample_indices"])
                            gt_array = np.array(plot_data["ground_truth"])
                            pred_array = np.array(plot_data["predictions"])

                            for i in range(6):
                                lines_true[i].set_data(indices, gt_array[:, i])
                                lines_pred[i].set_data(indices, pred_array[:, i])
                                axes[i].set_xlim(indices[0], indices[-1])

                            plt.pause(0.001)  # Update plot

                    # Display every 0.5 seconds
                    current_time = time.time()
                    time_since_last = current_time - last_display_time

                    if time_since_last >= 0.5:
                        elapsed = current_time - start_time
                        sample_fps = samples_processed / elapsed if elapsed > 0 else 0
                        pred_fps = predictions_made / elapsed if elapsed > 0 else 0

                        print(f"\n{'=' * 80}")
                        print(
                            f"[Samples: {samples_processed}/{num_samples} | Predictions: {predictions_made}]"
                        )
                        print(
                            f"Sample FPS: {sample_fps:.1f} | Prediction FPS: {pred_fps:.1f}"
                        )
                        print(
                            f"\n{'Ground Truth':<15} {'Predicted':<15} {'Servo Value':<15} {'Error':<10}"
                        )
                        print("-" * 80)

                        for i, name in enumerate(self.finger_names):
                            true_val = true_fingers[i]
                            pred_val = prediction[i]
                            servo_val = servo_values[i]
                            error = abs(true_val - pred_val)

                            # Visual comparison
                            true_bar = "█" * int(true_val * 15)
                            pred_bar = "▓" * int(pred_val * 15)

                            print(
                                f"{name:<12} | {true_val:6.3f} {true_bar:<15} | {pred_val:6.3f} {pred_bar:<15} | {servo_val:4d} | {error:.3f}"
                            )

                        last_display_time = current_time

                # Sleep to simulate real-time data arrival
                time.sleep(sample_interval)

        except KeyboardInterrupt:
            print("\n\nStopping simulation...")

        finally:
            # Close plot if it was shown
            if show_plot:
                plt.ioff()
                print("\nPlot window will remain open. Close it manually to continue.")

            # Calculate overall statistics
            if predictions:
                predictions_array = np.array(predictions)
                ground_truth_array = np.array(ground_truth)

                mae = np.mean(np.abs(predictions_array - ground_truth_array), axis=0)
                overall_mae = np.mean(mae)

                print("\n" + "=" * 80)
                print("SIMULATION SUMMARY")
                print("=" * 80)
                print(f"Total samples processed: {samples_processed}")
                print(f"Total predictions made: {predictions_made}")
                print("\nMean Absolute Error (MAE) per finger:")
                for i, name in enumerate(self.finger_names):
                    print(f"  {name:<12}: {mae[i]:.4f}")
                print(f"\nOverall MAE: {overall_mae:.4f}")
                print("=" * 80)

    def run_hardware_debug(self, port, baudrate=115200, show_plot=True):
        """
        Run real-time inference with HARDWARE while showing debug visualization.
        This combines hardware control (like inference.py) with visual debugging.

        Args:
            port: Serial port name (e.g., "COM5", "/dev/ttyAMA0")
            baudrate: Serial port baud rate
            show_plot: Whether to show real-time plot (hand movements only, no ground truth)
        """
        print("\n" + "=" * 80)
        print("HARDWARE DEBUG MODE - Real-Time Inference with Visualization")
        print("=" * 80)
        print(f"Port: {port}")
        print(f"Baudrate: {baudrate}")
        print(f"Live plot: {'ENABLED' if show_plot else 'DISABLED'}")
        print("Press Ctrl+C to stop")
        print("=" * 80 + "\n")

        # Open serial port
        try:
            ser = pyserial.Serial(port, baudrate, timeout=0.1)
            print(f"✓ Connected to {port}")
        except Exception as e:
            print(f"✗ Failed to connect to {port}: {e}")
            print("\nMake sure:")
            print(f"  1. Arduino is connected to {port}")
            print("  2. No other program is using the port")
            print("  3. Port name is correct (check Device Manager on Windows)")
            return

        start_time = time.time()
        last_display_time = start_time

        # Diagnostics
        samples_received = 0
        predictions_made = 0

        # Storage for visualization
        prediction_history = []

        # Setup real-time plotting
        if show_plot:
            plt.ion()  # Turn on interactive mode
            fig, axes = plt.subplots(6, 1, figsize=(12, 10))
            fig.suptitle(
                "Hardware Debug - Real-Time Hand Control",
                fontsize=14,
                fontweight="bold",
            )

            # Rolling window for plot (show last 100 samples)
            plot_window_size = 100
            plot_data = {
                "predictions": deque(maxlen=plot_window_size),
                "sample_indices": deque(maxlen=plot_window_size),
            }

            lines_pred = []

            for i, name in enumerate(self.finger_names):
                ax = axes[i]
                (line_pred,) = ax.plot([], [], "r-", label="Predicted", linewidth=2)
                lines_pred.append(line_pred)

                ax.set_ylabel(name, fontsize=10)
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)

                if i == len(self.finger_names) - 1:
                    ax.set_xlabel("Sample Index", fontsize=10)

            plt.tight_layout()
            plt.show(block=False)

        try:
            sample_count = 0

            while True:
                try:
                    # Read line from serial port (same as inference.py)
                    line = ser.readline().decode().strip()

                    if not line:
                        continue

                    # Parse Arduino format: S4:raw,env;S3:raw,env;S2:raw,env;S1:raw,env
                    parts = line.split(";")
                    if len(parts) != 4:
                        continue

                    sensor_data = {}
                    for part in parts:
                        if ":" not in part:
                            continue
                        sensor_id, values = part.split(":")
                        raw, env = values.split(",")
                        sensor_data[sensor_id] = (float(raw), float(env))

                    # Check we got all sensors
                    if not all(s in sensor_data for s in ["S1", "S2", "S3", "S4"]):
                        continue

                    # Map to expected format: [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
                    # S4 = sensor 0, S3 = sensor 1, S2 = sensor 2, S1 = sensor 3
                    sensor_values = np.array(
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

                    # Process sample
                    self.process_sample(sensor_values)

                    # Make prediction (only when buffer is full)
                    prediction = self.predict()

                    if prediction is not None:
                        predictions_made += 1
                        sample_count += 1

                        # Convert predictions [0,1] to servo values [0,1023]
                        servo_values = (prediction * 1023).astype(int)
                        servo_values = np.clip(servo_values, 0, 1023)

                        # Send servo commands to Arduino (CONTROL THE HAND!)
                        msg = ",".join(map(str, servo_values)) + "\n"
                        ser.write(msg.encode())

                        # Store for visualization
                        prediction_history.append(prediction)

                        # Update plot
                        if show_plot:
                            plot_data["predictions"].append(prediction)
                            plot_data["sample_indices"].append(sample_count)

                            if len(plot_data["sample_indices"]) > 1:
                                indices = list(plot_data["sample_indices"])
                                pred_array = np.array(plot_data["predictions"])

                                for i in range(6):
                                    lines_pred[i].set_data(indices, pred_array[:, i])
                                    axes[i].set_xlim(indices[0], indices[-1])

                                plt.pause(0.001)  # Update plot

                        # Display every 0.5 seconds
                        current_time = time.time()
                        time_since_last = current_time - last_display_time

                        if time_since_last >= 0.5:
                            elapsed = current_time - start_time
                            sample_fps = (
                                samples_received / elapsed if elapsed > 0 else 0
                            )
                            pred_fps = predictions_made / elapsed if elapsed > 0 else 0

                            print(f"\n{'=' * 80}")
                            print(
                                f"[Samples: {samples_received} | Predictions: {predictions_made}]"
                            )
                            print(
                                f"Sample FPS: {sample_fps:.1f} | Prediction FPS: {pred_fps:.1f}"
                            )
                            print("\nPredictions → Servo Commands:")
                            print("-" * 80)

                            for i, name in enumerate(self.finger_names):
                                pred_val = prediction[i]
                                servo_val = servo_values[i]

                                # Visual bar
                                pred_bar = "█" * int(pred_val * 20)

                                print(
                                    f"{name:<12}: {pred_val:.3f} {pred_bar:<20} → servo: {servo_val:4d}"
                                )

                            last_display_time = current_time

                except Exception as e:
                    print(f"Error processing sample: {e}")
                    import traceback

                    traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\nStopping hardware debug...")
            print(f"Total samples received: {samples_received}")
            print(f"Total predictions made: {predictions_made}")
            elapsed = time.time() - start_time
            if elapsed > 0:
                print(f"Average sample rate: {samples_received / elapsed:.1f} Hz")
                print(f"Average prediction rate: {predictions_made / elapsed:.1f} Hz")

        finally:
            # Close serial connection
            ser.close()
            print("✓ Serial port closed")

            # Close plot if it was shown
            if show_plot:
                plt.ioff()
                print("\nPlot window will remain open. Close it manually to continue.")


def main():
    # Configuration (matching inference_streaming.py)
    # model_path = "training/notebooks/best_lstm_model.pth"
    # scaler_path = "training/notebooks/scaler_inputs_lstm.pkl"
    model_path = "data/martin6/best_lstm_model.pth"
    scaler_path = "data/martin6/scaler_inputs_lstm.pkl"

    # ==============================================================================
    # MODE SELECTION: Choose what you want to debug
    # ==============================================================================
    # "csv"          - Test predictions with recorded CSV data (no hardware)
    # "hardware"     - Control actual hand with live Arduino data (no CSV)
    # "csv+hardware" - Replay CSV data AND send predictions to actual hand (BEST FOR DEBUGGING!)
    # ==============================================================================
    MODE = "csv+hardware"  # <-- CHANGE THIS to switch modes

    # CSV Simulation Settings (for "csv" and "csv+hardware" modes)
    data_dir = "data/martin3/raw"
    start_idx = 1000  # Skip initial samples for stability
    num_samples = 1000  # Number of samples to process
    playback_speed = (
        1.0  # 1.0 = real-time, 2.0 = 2x speed (slower for watching hand move)
    )

    # Hardware Settings (for "hardware" and "csv+hardware" modes)
    if sys.platform == "darwin":
        port = "/dev/tty.usbmodem11301"  # Mac
    elif sys.platform == "win32":
        port = "COM3"  # Windows
    else:
        port = "/dev/ttyAMA0"  # Raspberry Pi
    baudrate = 115200

    # Visualization
    show_plot = True  # Enable real-time plotting

    # Output Smoothing (for fluid hand movements)
    # Lower values = smoother movements (less jitter, slower response)
    # Higher values = more responsive (faster response, more jitter)
    # Recommended: 0.2-0.4 for smooth hand movements

    # ==============================================================================

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return

    # Initialize inference engine
    print("\n" + "=" * 80)
    print("INITIALIZING DEBUG INFERENCE ENGINE")
    print("=" * 80)
    print(f"Mode: {MODE.upper()}")
    if MODE == "csv":
        print(f"Data source: {data_dir}")
        print(f"Samples: {start_idx} to {start_idx + num_samples}")
    else:
        print(f"Hardware port: {port}")
        print(f"Baudrate: {baudrate}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_engine = DebugInference(
        model_path=model_path,
        scaler_path=scaler_path,
        window_size=30,
        device=device,
        smoothing_alpha=1.0,
    )
    print(f"✓ Model loaded successfully on {device}")
    print(f"✓ Model num_layers: {inference_engine.model.num_layers}")
    print(f"✓ Model hidden_size: {inference_engine.model.hidden_size}")

    # Run based on selected mode
    if MODE == "csv":
        # CSV SIMULATION MODE - Test with recorded data only
        if not os.path.exists(data_dir):
            print(f"Error: Data directory not found at {data_dir}")
            return

        print("\n📊 Running in CSV SIMULATION mode")
        print("   Testing with recorded data (no hardware control)")

        # Load data
        df = inference_engine.load_data(data_dir)

        # Run simulation
        inference_engine.simulate_realtime(
            df=df,
            start_idx=start_idx,
            num_samples=num_samples,
            playback_speed=playback_speed,
            show_plot=show_plot,
        )

    elif MODE == "csv+hardware":
        # CSV + HARDWARE MODE - Replay CSV data but control actual hand
        if not os.path.exists(data_dir):
            print(f"Error: Data directory not found at {data_dir}")
            return

        print("\n🎯 Running in CSV + HARDWARE mode")
        print(f"   Replaying CSV data from {data_dir}")
        print(f"   Sending predictions to actual hand on {port}")
        print("   This lets you see if the hand moves correctly!")

        # Load data
        df = inference_engine.load_data(data_dir)

        # Run CSV replay with hardware control
        inference_engine.simulate_realtime_with_hardware(
            df=df,
            port=port,
            baudrate=baudrate,
            start_idx=start_idx,
            num_samples=num_samples,
            playback_speed=playback_speed,
            show_plot=show_plot,
        )

    elif MODE == "hardware":
        # HARDWARE MODE - Control actual hand with live Arduino data
        print("\n🤖 Running in HARDWARE mode")
        print(f"   Reading live sensor data from {port}")
        print("   Controlling actual hand in real-time")

        # Run hardware debug
        inference_engine.run_hardware_debug(
            port=port, baudrate=baudrate, show_plot=show_plot
        )

    else:
        print(
            f"Error: Invalid MODE '{MODE}'. Choose 'csv', 'hardware', or 'csv+hardware'"
        )


if __name__ == "__main__":
    main()
