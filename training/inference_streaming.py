"""
Streaming Inference for EMG-to-Finger Position Prediction

This script implements TRUE STREAMING inference with causal filtering:
- Raw sensor data is loaded without preprocessing
- Each sample is processed sequentially (sample-by-sample)
- High-pass filtering uses causal lfilter (StreamingHighPassFilter)
- No lookahead or future information is used
- Simulates real-time sensor data arrival

Key Components:
1. StreamingHighPassFilter: Causal IIR filter with state preservation
2. LSTMModel: 3-layer LSTM with skip connections
3. StreamingInference: Online processing with sliding window buffer
4. simulate_streaming: Sequential sample processing loop

NOTE: Model was trained with non-causal filtfilt, so performance may differ
from training metrics. For best results, retrain with causal filtering.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import glob
from collections import deque
import sys

# Add parent directory to path to import filter_strategies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.filter_strategies import create_filter, FilterStrategy


class StreamingStandardScaler:
    """
    Online/streaming version of StandardScaler that adapts to incoming data.

    Uses Welford's algorithm for numerically stable online computation of mean and variance.
    This allows the scaler to adapt to new data distributions without needing to see all data upfront.

    Useful for:
    - Handling distribution shift between training and deployment
    - Adapting to different users/sessions
    - Online learning scenarios

    The scaler can be initialized with pretrained statistics and then adapt,
    or start from scratch and learn entirely from streaming data.
    """

    def __init__(
        self, n_features, warmup_samples=100, adaptation_rate=0.01, use_pretrained=True
    ):
        """
        Args:
            n_features: Number of features to scale
            warmup_samples: Number of samples to collect before starting to scale
                          (ensures stable initial statistics)
            adaptation_rate: How fast to adapt to new data (0 = no adaptation, 1 = only use new data)
                           Typical values: 0.001-0.1
            use_pretrained: If True, initialize with pretrained scaler stats and adapt slowly
                          If False, learn from scratch (no pretrained initialization)
        """
        self.n_features = n_features
        self.warmup_samples = warmup_samples
        self.adaptation_rate = adaptation_rate
        self.use_pretrained = use_pretrained

        # Statistics (will be initialized from pretrained scaler if available)
        self.mean_ = np.zeros(n_features)
        self.var_ = np.ones(n_features)
        self.scale_ = np.ones(n_features)  # std dev

        # Online statistics tracking (Welford's algorithm)
        self.n_samples_seen_ = 0
        self.M2_ = np.zeros(n_features)  # Sum of squared differences from mean

        # Warmup buffer (collect samples before scaling)
        self.warmup_buffer = []
        self.is_warmed_up = False

    def initialize_from_pretrained(self, pretrained_scaler):
        """
        Initialize statistics from a pretrained StandardScaler.
        This provides a good starting point that adapts to new data.
        """
        if hasattr(pretrained_scaler, "mean_"):
            self.mean_ = pretrained_scaler.mean_.copy()
        if hasattr(pretrained_scaler, "var_"):
            self.var_ = pretrained_scaler.var_.copy()
        if hasattr(pretrained_scaler, "scale_"):
            self.scale_ = pretrained_scaler.scale_.copy()

        # Mark as warmed up since we have pretrained stats
        if self.use_pretrained:
            self.is_warmed_up = True
            print(
                f"✓ Initialized with pretrained statistics (mean range: [{self.mean_.min():.3f}, {self.mean_.max():.3f}])"
            )
            print(
                f"  Adaptation rate: {self.adaptation_rate} (will slowly adapt to new data)"
            )

    def partial_fit(self, X):
        """
        Update statistics with a new sample using Welford's algorithm.

        Welford's algorithm for online variance:
        For each new sample x:
          n = n + 1
          delta = x - mean
          mean = mean + delta / n
          M2 = M2 + delta * (x - mean)
          variance = M2 / (n - 1)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        for sample in X:
            # Warmup phase: collect samples before computing stable statistics
            if not self.is_warmed_up:
                self.warmup_buffer.append(sample)

                if len(self.warmup_buffer) >= self.warmup_samples:
                    # Warmup complete - compute initial statistics from buffer
                    warmup_data = np.array(self.warmup_buffer)

                    if not self.use_pretrained:
                        # Learn from scratch using warmup data
                        self.mean_ = warmup_data.mean(axis=0)
                        self.var_ = warmup_data.var(axis=0)
                        self.scale_ = np.sqrt(self.var_)
                        self.n_samples_seen_ = len(warmup_data)
                        self.M2_ = self.var_ * (self.n_samples_seen_ - 1)
                        print(f"✓ Warmup complete with {len(warmup_data)} samples")
                        print(
                            f"  Learned mean range: [{self.mean_.min():.3f}, {self.mean_.max():.3f}]"
                        )
                    else:
                        # We already have pretrained stats, just update counters
                        self.n_samples_seen_ = len(warmup_data)
                        # Initialize M2 based on current variance
                        self.M2_ = self.var_ * max(1, self.n_samples_seen_ - 1)
                        print(
                            f"✓ Warmup complete with {len(warmup_data)} samples (using pretrained stats)"
                        )

                    self.is_warmed_up = True
                    self.warmup_buffer = []  # Clear buffer to save memory

                continue

            # Online update using Welford's algorithm with adaptation rate
            self.n_samples_seen_ += 1

            # Compute update with adaptation rate (exponential moving average style)
            delta = sample - self.mean_

            if self.adaptation_rate > 0:
                # Adaptive update: blend old and new statistics
                # Lower adaptation_rate = more stable (slower to adapt)
                # Higher adaptation_rate = more responsive (faster to adapt)
                effective_n = min(1.0 / self.adaptation_rate, self.n_samples_seen_)

                self.mean_ = self.mean_ + delta / effective_n
                delta2 = sample - self.mean_
                self.M2_ = self.M2_ + delta * delta2

                # Update variance and scale with smoothing
                if self.n_samples_seen_ > 1:
                    new_var = self.M2_ / (effective_n - 1)
                    # Exponential moving average for variance
                    self.var_ = (
                        1 - self.adaptation_rate
                    ) * self.var_ + self.adaptation_rate * new_var
                    self.scale_ = np.sqrt(
                        np.maximum(self.var_, 1e-8)
                    )  # Avoid division by zero
            else:
                # No adaptation - keep pretrained statistics
                pass

    def transform(self, X):
        """
        Scale features using current mean and std.
        Also updates statistics in the background (online learning).
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Update statistics with this sample (online learning)
        self.partial_fit(X)

        # Scale using current statistics
        if self.is_warmed_up:
            X_scaled = (X - self.mean_) / np.maximum(self.scale_, 1e-8)
        else:
            # During warmup, use pretrained stats if available, else pass through
            if self.use_pretrained and np.any(self.scale_ != 1.0):
                X_scaled = (X - self.mean_) / np.maximum(self.scale_, 1e-8)
            else:
                X_scaled = X  # Pass through during warmup

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


class StreamingInference:
    def __init__(
        self,
        model_path,
        scaler_path,
        window_size=50,
        device="cpu",
        use_online_scaling=True,
        online_adaptation_rate=0.1,
        online_warmup_samples=100,
        filter_type="none",
        filter_config=None,
        use_scaling=True,
        feature_window=10,
    ):
        """
        Initialize streaming inference with optional online scaling and configurable filtering.

        Args:
            model_path: Path to trained LSTM model
            scaler_path: Path to pretrained StandardScaler
            window_size: Size of sliding window for LSTM input
            device: 'cpu' or 'cuda'
            use_online_scaling: If True, use adaptive online scaler that learns from incoming data
                               If False, use fixed pretrained scaler
                               NOTE: This is only used if use_scaling=True
            online_adaptation_rate: How fast online scaler adapts (0.001-0.1)
                                   Lower = more stable, Higher = more responsive
            online_warmup_samples: Number of samples to collect before online scaling starts
            filter_type: Type of filter to use: 'none', 'highpass', 'lowpass', 'bandpass', 'moving_average'
            filter_config: Dictionary of filter parameters (depends on filter_type)
                          For 'highpass': {'cutoff': 0.5, 'order': 4}
                          For 'lowpass': {'cutoff': 5.0, 'order': 4}
                          For 'bandpass': {'low_cutoff': 0.5, 'high_cutoff': 5.0, 'order': 4}
                          For 'moving_average': {'window_size': 5}
                          For 'none': {} (no parameters needed)
            use_scaling: If True, apply scaling to features. If False, use raw features.
                        Should match the USE_SCALING setting from training.
            feature_window: Window size for computing EMG features (RMS, MAV, ZC, WL)
        """
        self.window_size = window_size
        self.device = torch.device(device)
        self.use_online_scaling = use_online_scaling
        self.use_scaling = use_scaling
        self.feature_window = feature_window

        # Load model checkpoint first to get number of features
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Determine number of input features from checkpoint
        # PRIORITY: Use actual model weights (most reliable) over hyperparameters
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            first_layer_weight = checkpoint["model_state_dict"][
                "lstm_layers.0.weight_ih_l0"
            ]
            n_inputs = first_layer_weight.shape[
                1
            ]  # input dimension from actual weights
            print(f"Loaded n_features from model weights: {n_inputs}")
        elif isinstance(checkpoint, dict) and "hyperparameters" in checkpoint:
            n_inputs = checkpoint["hyperparameters"].get("n_features", 16)
            print(f"Loaded n_features from hyperparameters: {n_inputs}")
        else:
            n_inputs = 16  # Default fallback
            print(f"Using default n_features: {n_inputs}")

        n_outputs = 6

        # Load pretrained scaler
        pretrained_scaler = joblib.load(scaler_path)

        # Initialize scaler (online or fixed) - only if scaling is enabled
        if use_scaling:
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
                print("  → Scaler will adapt to distribution shifts in real-time")
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

        self.model = LSTMModel(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
        ).to(self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.window_buffer = deque(maxlen=window_size)

        # Store model dimensions for feature processing
        self.n_inputs = n_inputs
        self.use_emg_features = n_inputs == 32  # 32 features = with EMG, 16 = without

        if self.use_emg_features:
            print(f"\n{'=' * 70}")
            print("EMG FEATURES ENABLED")
            print(f"{'=' * 70}")
            print(f"  Model expects {n_inputs} features (base + spatial + EMG)")
            print("  EMG features: RMS, MAV, ZC, WL per channel")
            print(f"{'=' * 70}\n")
        else:
            print(f"\n{'=' * 70}")
            print("EMG FEATURES DISABLED")
            print(f"{'=' * 70}")
            print(f"  Model expects {n_inputs} features (base + spatial only)")
            print(f"{'=' * 70}\n")

        # ⚠️ CRITICAL: This MUST match the training data sample rate!
        # The training notebook measured: MEASURED_SAMPLE_RATE = 4.2144 Hz
        fs = 4.2144  # Hz - MEASURED from training data timestamps

        # Initialize filter using strategy pattern
        if filter_config is None:
            filter_config = {}

        # Set default configs for different filter types
        default_configs = {
            "none": {},
            "highpass": {"cutoff": 0.5, "order": 4},
            "lowpass": {"cutoff": 2.0, "order": 4},
            "bandpass": {"low_cutoff": 0.5, "high_cutoff": 2.0, "order": 4},
            "moving_average": {"window_size": 5},
        }

        # Merge default config with user config
        final_config = default_configs.get(filter_type, {}).copy()
        final_config.update(filter_config)

        self.filter = create_filter(filter_type, fs, **final_config)

        print(f"\n{'=' * 70}")
        print("FILTER CONFIGURATION")
        print(f"{'=' * 70}")
        print(f"  {self.filter.get_description()}")
        print(f"{'=' * 70}\n")

        # Define neighbor relationships for spatial features (MUST match notebook exactly!)
        # Training notebook uses 1-indexed: {1: [], 2: [3], 3: [2, 4], 4: [3]}
        # Convert to 0-indexed for array access in inference
        self.neighbors = {1: [2], 2: [1, 4], 3: [4], 4: [2, 3]}

        # Buffers for EMG feature computation (per channel) - only if using EMG features
        # We need to keep a history of raw samples to compute features
        if self.use_emg_features:
            self.raw_buffers = {
                "raw0": deque(maxlen=feature_window),
                "raw1": deque(maxlen=feature_window),
                "raw2": deque(maxlen=feature_window),
                "raw3": deque(maxlen=feature_window),
            }
        else:
            self.raw_buffers = None

    def compute_spatial_features(self, raw_values, env_values):
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

    def compute_emg_features(self, signal_buffer):
        """
        Compute EMG features from a buffered signal (streaming mode).

        Args:
            signal_buffer: deque of recent samples (length <= feature_window)

        Returns:
            Dictionary with keys: 'rms', 'mav', 'zc', 'wl'
        """
        if len(signal_buffer) < 2:
            # Not enough samples yet
            return {"rms": 0.0, "mav": 0.0, "zc": 0.0, "wl": 0.0}

        # Convert deque to array
        signal = np.array(signal_buffer)

        # RMS (Root Mean Square) - signal power
        rms = np.sqrt(np.mean(signal**2))

        # MAV (Mean Absolute Value) - signal amplitude
        mav = np.mean(np.abs(signal))

        # ZC (Zero Crossings) - frequency content indicator
        zc = np.sum(np.diff(np.sign(signal)) != 0)

        # WL (Waveform Length) - signal complexity
        wl = np.sum(np.abs(np.diff(signal)))

        return {"rms": rms, "mav": mav, "zc": float(zc), "wl": wl}

    def process_sample(self, raw_sample):
        """
        Process a single incoming sample.

        Args:
            raw_sample: Array of [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
                       env values are ALREADY FILTERED (batch-filtered in load_test_data)
                       raw values are unfiltered

        Returns:
            Scaled feature vector ready for windowing
        """
        # Extract sensor values
        # ⚠️ env values are ALREADY BATCH-FILTERED (done in load_test_data to match training)
        env_values_filtered = raw_sample[
            ::2
        ].copy()  # env0, env1, env2, env3 (already filtered)
        raw_values = raw_sample[1::2].copy()  # raw0, raw1, raw2, raw3 (unfiltered)

        # Compute spatial features using FILTERED env and UNFILTERED raw
        raw_diffs, env_diffs = self.compute_spatial_features(
            raw_values, env_values_filtered
        )

        # Match the notebook feature order:
        # env0, raw0, env1, raw1, env2, raw2, env3, raw3, spatial_diffs, [emg_features if enabled]
        interleaved_sensors = []
        for i in range(4):
            interleaved_sensors.append(env_values_filtered[i])
            interleaved_sensors.append(raw_values[i])

        # Conditionally compute EMG features if model expects them
        if self.use_emg_features:
            # Update raw buffers for EMG feature computation
            for i, ch in enumerate(["raw0", "raw1", "raw2", "raw3"]):
                self.raw_buffers[ch].append(raw_values[i])

            # Compute EMG features for each raw channel
            emg_features = []
            for ch in ["raw0", "raw1", "raw2", "raw3"]:
                feat_dict = self.compute_emg_features(self.raw_buffers[ch])
                # Add in order: rms, mav, zc, wl
                emg_features.extend(
                    [
                        feat_dict["rms"],
                        feat_dict["mav"],
                        feat_dict["zc"],
                        feat_dict["wl"],
                    ]
                )

            features = np.concatenate(
                [interleaved_sensors, raw_diffs, env_diffs, emg_features]
            )
        else:
            # No EMG features - just base sensors + spatial features
            features = np.concatenate([interleaved_sensors, raw_diffs, env_diffs])

        # Scale features if scaling is enabled
        if self.use_scaling:
            if self.use_online_scaling:
                # Online scaler automatically updates statistics during transform
                features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
            else:
                # Fixed pretrained scaler (traditional approach)
                features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
        else:
            # No scaling - use raw features
            features_scaled = features

        self.window_buffer.append(features_scaled)

        return features_scaled

    def get_scaler_stats(self):
        """
        Get current scaler statistics for monitoring/debugging.
        Useful for tracking how the online scaler adapts over time.
        """
        if not self.use_scaling:
            return {
                "mean": None,
                "std": None,
                "var": None,
                "n_samples": "N/A (scaling disabled)",
                "is_warmed_up": None,
            }
        elif self.use_online_scaling:
            return self.scaler.get_stats()
        else:
            return {
                "mean": self.scaler.mean_,
                "std": self.scaler.scale_,
                "var": self.scaler.var_,
                "n_samples": "N/A (fixed scaler)",
                "is_warmed_up": True,
            }

    def predict(self):
        if len(self.window_buffer) < self.window_size:
            return None

        window_array = np.array(list(self.window_buffer))
        window_tensor = (
            torch.tensor(window_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            prediction = self.model(window_tensor)

        # Clamp predictions to valid range [0, 1] since finger positions are normalized
        prediction = torch.clamp(prediction, 0.0, 1.0)

        return prediction.cpu().numpy()[0]


def load_test_data(filter_strategy: FilterStrategy = None):
    """
    Load test data and optionally apply batch filtering to match training preprocessing.

    Args:
        filter_strategy: FilterStrategy instance to apply to env channels.
                        If None, no filtering is applied.

    Returns:
        DataFrame with optionally filtered env channels
    """
    dirs = ["data/tobias/test"]
    # dirs = ["data/afras/raw"]

    csv_files = []
    for d in dirs:
        csv_files.extend(glob.glob(os.path.join(d, "integrated_data_*.csv")))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified directory")

    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

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

    # Apply filtering if strategy provided
    if filter_strategy is not None:
        print(f"\n{'=' * 70}")
        print("APPLYING BATCH FILTERING TO ENV CHANNELS")
        print(f"{'=' * 70}")
        print(f"  {filter_strategy.get_description()}")
        print(f"{'=' * 70}\n")

        for col in ["env0", "env1", "env2", "env3"]:
            df_clean[col] = filter_strategy.apply_batch(df_clean[col].values)

        print("✅ Batch filtering complete - data matches training preprocessing")
    else:
        print(f"\n{'=' * 70}")
        print("NO FILTERING APPLIED")
        print(f"{'=' * 70}")
        print("  Using raw unfiltered data")
        print(f"{'=' * 70}\n")

    return df_clean


def simulate_streaming(
    inference_engine,
    df_clean,
    num_samples=1000,
    start_idx=0,
    show_scaler_stats=True,
    debug_features=False,
):
    sensor_columns = ["env0", "raw0", "env1", "raw1", "env2", "raw2", "env3", "raw3"]
    finger_columns = ["thumb_tip", "thumb_base", "index", "middle", "ring", "pinky"]

    predictions = []
    ground_truth = []

    end_idx = min(start_idx + num_samples, len(df_clean))

    print(f"\n{'=' * 70}")
    print("STREAMING INFERENCE SIMULATION")
    print(f"{'=' * 70}")
    print(f"Samples: {start_idx} to {end_idx} (total: {num_samples})")
    print(f"{'=' * 70}\n")

    # Track initial scaler stats
    if show_scaler_stats and inference_engine.use_online_scaling:
        initial_stats = inference_engine.get_scaler_stats()
        print("📊 Initial scaler statistics:")
        print(
            f"   Mean range: [{initial_stats['mean'].min():.3f}, {initial_stats['mean'].max():.3f}]"
        )
        print(
            f"   Std range:  [{initial_stats['std'].min():.3f}, {initial_stats['std'].max():.3f}]"
        )
        print(f"   Samples seen: {initial_stats['n_samples']}")
        print(f"   Warmed up: {initial_stats['is_warmed_up']}\n")

    # Debug: collect features for first few samples
    if debug_features:
        debug_raw_features = []
        debug_scaled_features = []

    for idx in range(start_idx, end_idx):
        row = df_clean.iloc[idx]

        sensor_values = row[sensor_columns].values.astype(np.float32)
        finger_values = row[finger_columns].values.astype(np.float32)

        scaled_features = inference_engine.process_sample(sensor_values)

        # Collect debug info for first 5 samples
        if debug_features and idx < start_idx + 5:
            debug_raw_features.append(sensor_values.copy())
            debug_scaled_features.append(scaled_features.copy())

        prediction = inference_engine.predict()

        if prediction is not None:
            predictions.append(prediction)
            ground_truth.append(finger_values)

        # Show progress with scaler stats updates
        if (idx - start_idx + 1) % 100 == 0:
            print(f"  Processed {idx - start_idx + 1}/{num_samples} samples...", end="")

            if show_scaler_stats and inference_engine.use_online_scaling:
                stats = inference_engine.get_scaler_stats()
                print(
                    f" [Scaler: {stats['n_samples']} samples, warmed_up={stats['is_warmed_up']}]"
                )
            else:
                print()

    # Show debug features
    if debug_features and len(debug_scaled_features) > 0:
        print(f"\n{'=' * 70}")
        print("DEBUG: First processed sample features")
        print(f"{'=' * 70}")
        print(f"Raw sensor input: {debug_raw_features[0]}")
        print(f"Scaled features (16 total): {debug_scaled_features[0]}")
        print(f"Feature shape: {debug_scaled_features[0].shape}")
        print(
            f"Feature range: [{debug_scaled_features[0].min():.3f}, {debug_scaled_features[0].max():.3f}]"
        )
        print(f"{'=' * 70}\n")

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Show final scaler statistics and adaptation
    if show_scaler_stats and inference_engine.use_online_scaling:
        final_stats = inference_engine.get_scaler_stats()
        print(f"\n{'=' * 70}")
        print(f"📊 FINAL SCALER STATISTICS (after {final_stats['n_samples']} samples)")
        print(f"{'=' * 70}")
        print(
            f"Mean range: [{final_stats['mean'].min():.3f}, {final_stats['mean'].max():.3f}]"
        )
        print(
            f"Std range:  [{final_stats['std'].min():.3f}, {final_stats['std'].max():.3f}]"
        )

        if "initial_stats" in locals():
            # Show how much the scaler adapted
            mean_change = np.abs(final_stats["mean"] - initial_stats["mean"]).mean()
            std_change = np.abs(final_stats["std"] - initial_stats["std"]).mean()
            print("\nAdaptation from initial:")
            print(f"  Mean change (avg): {mean_change:.4f}")
            print(f"  Std change (avg):  {std_change:.4f}")

            if mean_change > 0.1 or std_change > 0.1:
                print("  ✅ Scaler adapted to distribution shift in data")
            else:
                print("  → Scaler remained stable (data similar to training)")
        print(f"{'=' * 70}\n")

    # Debug: Check value ranges
    print("\nPrediction ranges:")
    print(f"  Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
    print("Ground truth ranges:")
    print(f"  Min: {ground_truth.min():.4f}, Max: {ground_truth.max():.4f}")

    return predictions, ground_truth


def compute_metrics(predictions, ground_truth):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)

    print("\n" + "=" * 60)
    print("STREAMING INFERENCE METRICS")
    print("=" * 60)
    print("Overall Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.4f}")

    finger_names = ["thumb_tip", "thumb_base", "index", "middle", "ring", "pinky"]
    print("\nPer-finger metrics:")
    for i, name in enumerate(finger_names):
        mse_i = mean_squared_error(ground_truth[:, i], predictions[:, i])
        mae_i = mean_absolute_error(ground_truth[:, i], predictions[:, i])
        r2_i = r2_score(ground_truth[:, i], predictions[:, i])
        print(f"  {name:<12} | MSE: {mse_i:.6f} | MAE: {mae_i:.6f} | R²: {r2_i:.4f}")


def visualize_predictions(predictions, ground_truth, n_samples=400):
    import matplotlib.pyplot as plt

    finger_names = ["thumb_tip", "thumb_base", "index", "middle", "ring", "pinky"]

    plt.figure(figsize=(12, len(finger_names) * 4))
    for i, name in enumerate(finger_names):
        plt.subplot(len(finger_names), 1, i + 1)
        plt.plot(ground_truth[:n_samples, i], label=f"True {name}", linewidth=2)
        plt.plot(
            predictions[:n_samples, i], label=f"Pred {name}", alpha=0.7, linewidth=2
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"Streaming Predictions vs True for {name}", fontsize=12)
        plt.ylabel("Amplitude")
    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.show()


def main():
    # model_path = "training/notebooks/best_lstm_model.pth"
    # scaler_path = "training/notebooks/scaler_inputs_lstm.pkl"
    model_path = "data/tobias/lstm_model_complete.pth"
    scaler_path = "data/tobias/scaler_inputs_lstm.pkl"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return

    print("\n" + "=" * 70)
    print("STREAMING INFERENCE")
    print("=" * 70)

    # Load model checkpoint to extract filter configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract filter configuration from checkpoint (if available)
    if isinstance(checkpoint, dict) and "hyperparameters" in checkpoint:
        hyperparams = checkpoint["hyperparameters"]
        FILTER_TYPE = hyperparams.get("filter_type", "none")
        FILTER_CONFIG = hyperparams.get("filter_config", {"cutoff": 0.5, "order": 4})
        fs = hyperparams.get("sampling_rate", 4.2144)
        USE_SCALING = hyperparams.get("use_scaling", False)
        FEATURE_WINDOW = hyperparams.get(
            "feature_window", 10
        )  # NEW: Load EMG feature window

        print("\nLoaded configuration from model checkpoint:")
        print(f"   Filter type: {FILTER_TYPE}")
        print(f"   Filter config: {FILTER_CONFIG}")
        print(f"   Sampling rate: {fs:.4f} Hz")
        print(f"   Scaling: {'ENABLED' if USE_SCALING else 'DISABLED'}")
        print(f"   Feature window: {FEATURE_WINDOW} samples")
    else:
        # Fallback to defaults if checkpoint doesn't have filter config
        print("\nNo configuration found in checkpoint, using defaults:")
        FILTER_TYPE = "highpass"
        FILTER_CONFIG = {"cutoff": 0.5, "order": 4}
        fs = 4.2144
        USE_SCALING = True
        FEATURE_WINDOW = 10
        print(f"   Filter type: {FILTER_TYPE}")
        print(f"   Filter config: {FILTER_CONFIG}")
        print(f"   Sampling rate: {fs:.4f} Hz")
        print(f"   Scaling: {'ENABLED' if USE_SCALING else 'DISABLED'}")
        print(f"   Feature window: {FEATURE_WINDOW} samples")

    # You can override the filter configuration here if needed:
    # FILTER_TYPE = 'none'
    # FILTER_CONFIG = {}
    # USE_SCALING = False
    # FEATURE_WINDOW = 10

    print("\nInitializing streaming inference engine...")

    # ⚠️ ONLINE ADAPTIVE SCALING: Use with caution!
    # For testing on TRAINING DATA: Set use_online_scaling=False (use fixed pretrained scaler)
    # For NEW sessions/users: Set use_online_scaling=True with LOW adaptation_rate (0.001-0.01)
    # NOTE: Online scaling only applies if USE_SCALING=True
    inference_engine = StreamingInference(
        model_path=model_path,
        scaler_path=scaler_path,
        window_size=30,
        device=device,
        use_online_scaling=True
        if USE_SCALING
        else False,  # ✅ Only use online scaling if scaling enabled
        online_adaptation_rate=0.001,  # Only used if use_online_scaling=True
        online_warmup_samples=100,  # Only used if use_online_scaling=True
        filter_type=FILTER_TYPE,  # ✅ Loaded from checkpoint
        filter_config=FILTER_CONFIG,  # ✅ Loaded from checkpoint
        use_scaling=USE_SCALING,  # ✅ Loaded from checkpoint
        feature_window=FEATURE_WINDOW,  # ✅ NEW: Loaded from checkpoint
    )

    print(f"✓ Model loaded successfully on {device}")
    print(f"  Model num_layers: {inference_engine.model.num_layers}")
    print(f"  Model hidden_size: {inference_engine.model.hidden_size}")

    print("\nLoading test data...")
    # Create same filter for batch preprocessing
    batch_filter = create_filter(FILTER_TYPE, fs, **FILTER_CONFIG)
    df_clean = load_test_data(filter_strategy=batch_filter)
    print(f"✓ Loaded {len(df_clean)} samples\n")

    # DEBUG: Check data ranges
    sensor_columns = ["env0", "raw0", "env1", "raw1", "env2", "raw2", "env3", "raw3"]
    print("\n" + "=" * 70)
    print("RAW DATA RANGES (after batch filtering)")
    print("=" * 70)
    for col in sensor_columns:
        vals = df_clean[col].values
        print(
            f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}, std={vals.std():.4f}"
        )
    print("=" * 70 + "\n")

    # ⚠️ IMPORTANT: Skip initial samples to allow filter to stabilize
    # The training applies lfilter to entire signal, so first ~100 samples have filter transient
    # Start from later in dataset for fair comparison
    predictions, ground_truth = simulate_streaming(
        inference_engine,
        df_clean,
        num_samples=2000,
        start_idx=5000,
        debug_features=True,
    )

    print(f"\nGenerated {len(predictions)} predictions")

    compute_metrics(predictions, ground_truth)

    visualize_predictions(predictions, ground_truth, n_samples=2000)


if __name__ == "__main__":
    main()
