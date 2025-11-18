#!/usr/bin/env python3
"""
🚀 Hardware Inference Script for Optimized LSTM Model

This script runs real-time inference using the optimized LSTM model from the notebook
with attention mechanism, layer normalization, and other advanced features.

Key Features:
- Optimized LSTM with attention mechanism
- BCEWithLogitsLoss compatibility (model outputs logits)
- Delta-enhanced servo smoothing for reduced latency
- Mixed precision support
- Real-time hardware control

Usage:
    python -m rpi.src.inference_optimized

Author: AI Assistant
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import time
import os
import sys
from collections import deque
import threading
from queue import Queue, Empty

# Try to import serial - make it optional for testing
try:
    import serial as pyserial

    HAS_SERIAL = True
except ImportError:
    print("⚠️ pyserial not installed. Hardware inference disabled.")
    print("   Install with: pip install pyserial")
    HAS_SERIAL = False

# Add parent directory to path for filter strategies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.filter_strategies import create_filter


class AttentionLayer(nn.Module):
    """Multi-head attention for sequence modeling - matches notebook exactly"""

    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return self.layer_norm(x + attended)


class OptimizedLSTM(nn.Module):
    """
    Optimized LSTM model matching the notebook architecture exactly.

    Features:
    - Multi-head attention mechanism
    - Layer normalization
    - Residual connections
    - Outputs raw logits (no sigmoid) for BCEWithLogitsLoss compatibility
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        hidden_size=256,
        num_layers=3,
        dropout=0.3,
        use_attention=True,
        use_layer_norm=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Input projection
        self.input_proj = nn.Linear(n_inputs, hidden_size)

        # LSTM layers with layer norm
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0
                )
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_size))

        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_size, num_heads=4)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layers - NO sigmoid here, outputs raw logits
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Project input
        x = self.input_proj(x)

        # LSTM layers with residual connections
        for i, lstm in enumerate(self.lstm_layers):
            residual = x
            x, _ = lstm(x)

            if self.layer_norms:
                x = self.layer_norms[i](x)

            x = x + residual  # Residual connection
            x = self.dropout(x)

        # Attention
        if self.use_attention:
            x = self.attention(x)

        # Take last timestep
        x = x[:, -1, :]

        # Output layers - returns logits (unbounded)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # Return raw logits, apply sigmoid during inference


class OptimizedHardwareInference:
    """
    🚀 Ultra-fast hardware inference for the optimized LSTM model.

    Features:
    - Optimized LSTM with attention mechanism
    - Delta-enhanced servo smoothing
    - Threading for minimal latency
    - Pre-allocated memory buffers
    - JIT compilation support
    """

    def __init__(
        self,
        model_path="../../data/tobias/optimized_lstm_complete.pth",
        scaler_path="../../data/tobias/scaler_optimized.pkl",
        window_size=50,  # Match notebook window size
        device="cpu",
        delta_weight=0.4,  # Higher responsiveness for optimized model
        delta_threshold=0.03,  # Lower threshold for faster response
        servo_smoothing_alpha=0.7,  # Less aggressive smoothing
        prediction_smoothing_alpha=0.8,  # Light prediction smoothing
    ):
        """
        Initialize the optimized hardware inference system.

        Args:
            model_path: Path to optimized LSTM checkpoint
            scaler_path: Path to feature scaler
            window_size: Sliding window size (should match training)
            device: 'cpu' or 'cuda'
            delta_weight: How much to weight prediction changes (0.0-1.0)
            delta_threshold: Minimum change to respond to
            servo_smoothing_alpha: Servo command smoothing (higher = less smooth)
            prediction_smoothing_alpha: Prediction smoothing (higher = less smooth)
        """
        self.window_size = window_size
        self.device = torch.device(device)
        self.delta_weight = delta_weight
        self.delta_threshold = delta_threshold
        self.servo_smoothing_alpha = servo_smoothing_alpha
        self.prediction_smoothing_alpha = prediction_smoothing_alpha

        print(f"\n{'=' * 80}")
        print("🚀 OPTIMIZED LSTM HARDWARE INFERENCE")
        print(f"{'=' * 80}")
        print(f"Device: {device}")
        print(f"Window size: {window_size}")
        print(f"Delta weight: {delta_weight}")
        print(f"Delta threshold: {delta_threshold}")
        print(f"Servo smoothing: {servo_smoothing_alpha}")
        print(f"Prediction smoothing: {prediction_smoothing_alpha}")
        print(f"{'=' * 80}\n")

        # Load model checkpoint
        print("Loading optimized LSTM model...")
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            hyperparams = checkpoint.get("hyperparameters", {})

            # Extract training configuration
            self.filter_type = hyperparams.get("filter_type", "none")
            self.filter_config = hyperparams.get("filter_config", {})
            self.use_scaling = hyperparams.get("use_scaling", True)
            fs = hyperparams.get("sampling_rate", 30.0)
            self.use_attention = hyperparams.get("use_attention", True)
            self.use_layer_norm = hyperparams.get("use_layer_norm", True)

            print("✓ Loaded training configuration:")
            print(f"  Filter: {self.filter_type}")
            print(f"  Scaling: {self.use_scaling}")
            print(f"  Attention: {self.use_attention}")
            print(f"  Layer norm: {self.use_layer_norm}")
            print(f"  Sampling rate: {fs:.2f} Hz")
        else:
            # Fallback for older checkpoints
            state_dict = checkpoint
            self.filter_type = "none"
            self.filter_config = {}
            self.use_scaling = True
            fs = 30.0
            self.use_attention = True
            self.use_layer_norm = True
            print("⚠️ Using default configuration (checkpoint format not recognized)")

        # Initialize filter
        self.filter = create_filter(self.filter_type, fs, **self.filter_config)

        # Detect model dimensions from state dict
        # The input_proj layer tells us the actual input feature size
        input_proj_weight = state_dict["input_proj.weight"]
        n_inputs = input_proj_weight.shape[1]  # Input features to the projection layer
        n_outputs = 6  # Always 6 fingers

        print(f"✓ Model dimensions: {n_inputs} inputs → {n_outputs} outputs")

        # Initialize optimized LSTM model
        self.model = OptimizedLSTM(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_size=256,  # Match notebook
            num_layers=3,  # Match notebook
            dropout=0.3,  # Match notebook
            use_attention=self.use_attention,
            use_layer_norm=self.use_layer_norm,
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 🚀 JIT compilation for maximum speed
        try:
            dummy_input = torch.randn(1, window_size, n_inputs).to(self.device)
            self.model = torch.jit.trace(self.model, dummy_input)
            print("✓ Model JIT compiled for maximum inference speed")
        except Exception as e:
            print(f"⚠️ JIT compilation failed: {e}")

        # Load scaler
        if self.use_scaling:
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded ({self.scaler.n_features_in_} features)")
        else:
            self.scaler = None
            print("✓ No scaling (raw features)")

        # Initialize spatial feature computation
        self.neighbors = {1: [2], 2: [1, 4], 3: [4], 4: [2, 3]}

        # Pre-allocate memory buffers for zero-copy operations
        self.sensor_buffer = np.zeros(8, dtype=np.float32)
        self.feature_buffer = np.zeros(n_inputs, dtype=np.float32)
        self.window_array = np.zeros((window_size, n_inputs), dtype=np.float32)
        self.prediction_buffer = np.zeros(6, dtype=np.float32)
        self.servo_buffer = np.zeros(6, dtype=np.int32)

        # Initialize sliding window and tracking
        self.window_buffer = deque(maxlen=window_size)
        self.previous_prediction = None
        self.previous_servo_values = None
        self.smoothed_prediction = None

        # Finger names and reordering (match hardware expectations)
        self.finger_names = [
            "thumb_tip",
            "thumb_base",
            "index",
            "middle",
            "ring",
            "pinky",
        ]
        # Model outputs: [thumb_tip, index, middle, ring, pinky, thumb_base]
        # Hardware needs: [thumb_tip, thumb_base, index, middle, ring, pinky]
        self.prediction_reorder = [0, 5, 1, 2, 3, 4]

        print("✓ Memory buffers pre-allocated")
        print("✓ Optimized LSTM hardware inference ready!")

    def compute_spatial_features(self, raw_values, env_values):
        """Compute spatial difference features between neighboring sensors."""
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

    def process_sensor_sample(self, raw_sample):
        """
        🚀 Process a single sensor sample with maximum speed.

        Args:
            raw_sample: [env0, raw0, env1, raw1, env2, raw2, env3, raw3]

        Returns:
            Processed feature vector
        """
        # Copy to pre-allocated buffer
        np.copyto(self.sensor_buffer, raw_sample)

        # Extract sensor values
        env_values_raw = self.sensor_buffer[::2]  # env0, env1, env2, env3
        raw_values = self.sensor_buffer[1::2]  # raw0, raw1, raw2, raw3

        # Apply filtering to env values only
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

        # Build feature vector: [env0, raw0, env1, raw1, env2, raw2, env3, raw3, raw_diff1-4, env_diff1-4]
        idx = 0
        # Interleaved sensors
        for i in range(4):
            self.feature_buffer[idx] = env_values_filtered[i]
            self.feature_buffer[idx + 1] = raw_values[i]
            idx += 2

        # Spatial differences
        for i in range(4):
            self.feature_buffer[idx] = raw_diffs[i]
            idx += 1
        for i in range(4):
            self.feature_buffer[idx] = env_diffs[i]
            idx += 1

        # Apply scaling if enabled
        if self.use_scaling:
            features_scaled = self.scaler.transform(self.feature_buffer.reshape(1, -1))[
                0
            ]
            self.window_buffer.append(features_scaled.copy())
            return features_scaled
        else:
            self.window_buffer.append(self.feature_buffer.copy())
            return self.feature_buffer.copy()

    def predict_with_delta_enhancement(self):
        """
        🚀 Make prediction with delta enhancement for ultra-responsive control.

        Returns:
            Enhanced servo values [0, 1023] or None if buffer not full
        """
        if len(self.window_buffer) < self.window_size:
            return None

        # Copy window to pre-allocated array
        for i, features in enumerate(self.window_buffer):
            np.copyto(self.window_array[i], features)

        # Create tensor and predict
        window_tensor = (
            torch.from_numpy(self.window_array)
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )

        with torch.no_grad():
            logits = self.model(window_tensor)
            # Convert logits to probabilities using sigmoid
            prediction = torch.sigmoid(logits)

        # Copy to numpy buffer
        np.copyto(self.prediction_buffer, prediction.cpu().numpy().flatten())

        # Reorder predictions to match hardware
        prediction_reordered = self.prediction_buffer[self.prediction_reorder]

        # Apply prediction-level smoothing
        if self.smoothed_prediction is None:
            self.smoothed_prediction = prediction_reordered.copy()
        else:
            self.smoothed_prediction = (
                self.prediction_smoothing_alpha * prediction_reordered
                + (1 - self.prediction_smoothing_alpha) * self.smoothed_prediction
            )

        # 🚀 Delta enhancement
        if self.previous_prediction is None:
            # First prediction
            self.previous_prediction = self.smoothed_prediction.copy()
            enhanced_prediction = self.smoothed_prediction
        else:
            # Calculate delta (change from previous)
            delta = self.smoothed_prediction - self.previous_prediction

            # Apply threshold to filter noise
            delta_filtered = np.where(np.abs(delta) > self.delta_threshold, delta, 0.0)

            # Enhance prediction with weighted delta
            enhanced_prediction = self.smoothed_prediction + (
                self.delta_weight * delta_filtered
            )
            enhanced_prediction = np.clip(enhanced_prediction, 0.0, 1.0)

            # Update previous prediction
            self.previous_prediction = self.smoothed_prediction.copy()

        # Convert to servo values [0, 1023] with inversion
        raw_servo_values = ((1.0 - enhanced_prediction) * 1023).astype(int)
        raw_servo_values = np.clip(raw_servo_values, 0, 1023)

        # Apply servo-level smoothing
        if self.previous_servo_values is None:
            smoothed_servo_values = raw_servo_values
            self.previous_servo_values = raw_servo_values.copy()
        else:
            smoothed_servo_values = (
                self.servo_smoothing_alpha * raw_servo_values
                + (1 - self.servo_smoothing_alpha) * self.previous_servo_values
            ).astype(int)
            smoothed_servo_values = np.clip(smoothed_servo_values, 0, 1023)
            self.previous_servo_values = smoothed_servo_values.copy()

        return smoothed_servo_values

    def run_hardware_inference(self, port, baudrate=115200, show_stats=True):
        """
        🚀 Run ultra-low latency hardware inference with threading.

        Args:
            port: Serial port (e.g., "COM3")
            baudrate: Serial baud rate
            show_stats: Whether to show performance statistics
        """
        if not HAS_SERIAL:
            print("❌ pyserial not available. Cannot run hardware inference.")
            print("   Install with: pip install pyserial")
            return

        print(f"\n{'=' * 80}")
        print("🚀 STARTING OPTIMIZED HARDWARE INFERENCE")
        print(f"{'=' * 80}")
        print(f"Port: {port}")
        print(f"Baudrate: {baudrate}")
        print("Model: Optimized LSTM with Attention")
        print("Features: Delta enhancement, JIT compilation, threading")
        print("Press Ctrl+C to stop")
        print(f"{'=' * 80}\n")

        # Open serial port with optimized settings
        try:
            ser = pyserial.Serial(
                port,
                baudrate,
                timeout=0.001,
                write_timeout=0.001,
                inter_byte_timeout=None,
            )
            ser.set_buffer_size(rx_size=4096, tx_size=4096)
            print(f"✓ Connected to {port}")
        except Exception as e:
            print(f"✗ Failed to connect to {port}: {e}")
            return

        # Threading setup
        sensor_queue = Queue(maxsize=5)
        servo_queue = Queue(maxsize=5)
        running = threading.Event()
        running.set()

        # Statistics
        stats = {
            "samples_received": 0,
            "predictions_made": 0,
            "servo_commands_sent": 0,
            "start_time": time.time(),
        }

        def serial_reader_thread():
            """Read sensor data from Arduino."""
            while running.is_set():
                try:
                    line = ser.readline().decode().strip()
                    if not line:
                        continue

                    # Parse different sensor data formats
                    # Format 1: S4:raw,env;S3:raw,env;S2:raw,env;S1:raw,env
                    # Format 2: env0,raw0,env1,raw1,env2,raw2,env3,raw3 (comma-separated)

                    if ";" in line and ":" in line:
                        # Format 1: S4:raw,env;S3:raw,env;S2:raw,env;S1:raw,env
                        parts = line.split(";")
                        if len(parts) != 4:
                            continue

                        sensor_data = {}
                        valid = True

                        for part in parts:
                            if ":" not in part or "," not in part:
                                valid = False
                                break
                            sensor_id, values = part.split(":", 1)
                            raw_str, env_str = values.split(",", 1)
                            try:
                                sensor_data[sensor_id] = (
                                    float(raw_str),
                                    float(env_str),
                                )
                            except ValueError:
                                valid = False
                                break

                        if not valid or len(sensor_data) != 4:
                            continue

                        # Map to expected format
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

                    elif "," in line:
                        # Format 2: env0,raw0,env1,raw1,env2,raw2,env3,raw3
                        parts = line.split(",")
                        if len(parts) != 8:
                            continue

                        try:
                            sensor_values = np.array(
                                [float(x) for x in parts], dtype=np.float32
                            )
                        except ValueError:
                            continue
                    else:
                        continue

                    try:
                        sensor_queue.put_nowait(sensor_values)
                        stats["samples_received"] += 1
                    except Exception:
                        pass  # Queue full, drop sample

                except Exception as e:
                    if running.is_set():
                        print(f"Serial read error: {e}")
                        if hasattr(e, "args") and len(e.args) > 0:
                            print(f"  Raw line: '{line[:50]}...' (first 50 chars)")

        def inference_thread():
            """Process sensors and make predictions."""
            while running.is_set():
                try:
                    try:
                        sensor_values = sensor_queue.get(timeout=0.1)
                    except Empty:
                        continue

                    # Process sample
                    self.process_sensor_sample(sensor_values)

                    # Make prediction with delta enhancement
                    servo_values = self.predict_with_delta_enhancement()

                    if servo_values is not None:
                        stats["predictions_made"] += 1
                        try:
                            servo_queue.put_nowait(servo_values.copy())
                        except Exception:
                            pass  # Queue full

                except Exception as e:
                    if running.is_set():
                        print(f"Inference error: {e}")

        def servo_writer_thread():
            """Send servo commands to hardware."""
            while running.is_set():
                try:
                    try:
                        servo_values = servo_queue.get(timeout=0.1)
                    except Empty:
                        continue

                    # Send immediately
                    msg = ",".join(map(str, servo_values)) + "\n"
                    ser.write(msg.encode())
                    ser.flush()
                    stats["servo_commands_sent"] += 1

                except Exception as e:
                    if running.is_set():
                        print(f"Servo write error: {e}")

        def stats_thread():
            """Display performance statistics."""
            while running.is_set():
                try:
                    time.sleep(1.0)

                    elapsed = time.time() - stats["start_time"]
                    if elapsed > 0:
                        sample_fps = stats["samples_received"] / elapsed
                        pred_fps = stats["predictions_made"] / elapsed
                        servo_fps = stats["servo_commands_sent"] / elapsed

                        queue_depth = sensor_queue.qsize() + servo_queue.qsize()

                        print(
                            f"\r🚀 FPS: Sensors={sample_fps:.1f} | Predictions={pred_fps:.1f} | Servos={servo_fps:.1f} | Queue={queue_depth}",
                            end="",
                            flush=True,
                        )

                except Exception:
                    pass

        # Start all threads
        threads = [
            threading.Thread(target=serial_reader_thread, daemon=True),
            threading.Thread(target=inference_thread, daemon=True),
            threading.Thread(target=servo_writer_thread, daemon=True),
        ]

        if show_stats:
            threads.append(threading.Thread(target=stats_thread, daemon=True))

        for thread in threads:
            thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n Stopping optimized inference...")
            running.clear()

            for thread in threads:
                thread.join(timeout=1.0)

            # Final statistics
            elapsed = time.time() - stats["start_time"]
            print("\n📊 Final Performance:")
            print(f"   Runtime: {elapsed:.2f} seconds")
            print(
                f"   Samples: {stats['samples_received']} ({stats['samples_received'] / elapsed:.1f} Hz)"
            )
            print(
                f"   Predictions: {stats['predictions_made']} ({stats['predictions_made'] / elapsed:.1f} Hz)"
            )
            print(
                f"   Servo commands: {stats['servo_commands_sent']} ({stats['servo_commands_sent'] / elapsed:.1f} Hz)"
            )

        finally:
            ser.close()
            print("✓ Serial connection closed")


def main():
    """Main entry point for optimized hardware inference."""
    # Configuration - use paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    MODEL_PATH = os.path.join(
        project_root, "data", "tobias", "optimized_lstm_complete.pth"
    )
    SCALER_PATH = os.path.join(project_root, "data", "tobias", "scaler_optimized.pkl")
    SERIAL_PORT = "COM3"  # Change this to your Arduino port
    BAUDRATE = 115200

    print("🚀 Optimized LSTM Hardware Inference")
    print("=" * 80)

    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please train the optimized model first using the notebook.")
        return

    if not os.path.exists(SCALER_PATH):
        print(f"❌ Scaler file not found: {SCALER_PATH}")
        print("Please train the optimized model first using the notebook.")
        return

    # Check if serial is available for hardware inference
    if not HAS_SERIAL:
        print("\n🧪 TESTING MODE: Serial not available, will only test model loading")
        print("   Install pyserial for hardware inference: pip install pyserial")
        print("=" * 80)

    # Initialize inference engine
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        inference_engine = OptimizedHardwareInference(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            window_size=50,  # Match notebook training
            device=device,
            delta_weight=0.4,  # Responsive delta enhancement
            delta_threshold=0.03,  # Low noise threshold
            servo_smoothing_alpha=0.7,  # Moderate servo smoothing
            prediction_smoothing_alpha=0.8,  # Light prediction smoothing
        )

        print(f"✓ Optimized inference engine ready on {device}")

        if HAS_SERIAL:
            # Run hardware inference
            print(f"\n🔌 Starting hardware inference on {SERIAL_PORT}")
            inference_engine.run_hardware_inference(
                port=SERIAL_PORT, baudrate=BAUDRATE, show_stats=True
            )
        else:
            print("\n✅ MODEL LOADING TEST SUCCESSFUL!")
            print("   Model architecture:", inference_engine.model.__class__.__name__)
            print("   Window size:", inference_engine.window_size)
            print("   Device:", device)
            print("   Features:", inference_engine.feature_buffer.shape[0])
            print("   Delta enhancement enabled")
            print("\n   Install pyserial to run hardware inference:")
            print("   pip install pyserial")

    except Exception as e:
        print(f"❌ Error initializing inference engine: {e}")
        return


if __name__ == "__main__":
    main()
