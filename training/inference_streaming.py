import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import glob
from scipy.signal import butter, lfilter
from collections import deque


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


class StreamingInference:
    def __init__(self, model_path, scaler_path, window_size=30, device="cpu"):
        self.window_size = window_size
        self.device = torch.device(device)

        self.scaler = joblib.load(scaler_path)

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

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.window_buffer = deque(maxlen=window_size)

        # Define neighbor relationships for spatial features (matching notebook)
        self.neighbors = {1: [], 2: [3], 3: [2, 4], 4: [3]}

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

    def process_sample(self, raw_sample):
        # raw_sample contains: [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
        # env values are already high-pass filtered in load_test_data()
        env_values = raw_sample[::2].copy()  # env0, env1, env2, env3
        raw_values = raw_sample[1::2].copy()  # raw0, raw1, raw2, raw3

        # Compute spatial features
        raw_diffs, env_diffs = self.compute_spatial_features(raw_values, env_values)

        # Match the notebook feature order: env0, raw0, env1, raw1, env2, raw2, env3, raw3, then diffs
        interleaved_sensors = []
        for i in range(4):
            interleaved_sensors.append(env_values[i])
            interleaved_sensors.append(raw_values[i])

        features = np.concatenate([interleaved_sensors, raw_diffs, env_diffs])

        features_scaled = self.scaler.transform(features.reshape(1, -1))[0]

        self.window_buffer.append(features_scaled)

        return features_scaled

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


def load_test_data():
    from scipy.signal import butter, filtfilt

    def highpass(signal, fs, cutoff=0.5, order=4):
        b, a = butter(order, cutoff / (0.5 * fs), btype="high")
        return filtfilt(b, a, signal)

    dirs = ["data/martin2/raw"]

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

    # Apply high-pass filter to env columns (matching notebook preprocessing)
    fs = 1.0 / 0.03446
    for col in ["env0", "env1", "env2", "env3"]:
        df_clean[col] = highpass(df_clean[col].values, fs, cutoff=0.5)

    return df_clean


def simulate_streaming(inference_engine, df_clean, num_samples=1000, start_idx=0):
    sensor_columns = ["env0", "raw0", "env1", "raw1", "env2", "raw2", "env3", "raw3"]
    finger_columns = ["thumb_tip", "thumb_base", "index", "middle", "ring", "pinky"]

    predictions = []
    ground_truth = []

    end_idx = min(start_idx + num_samples, len(df_clean))

    print(f"Simulating streaming inference from sample {start_idx} to {end_idx}...")

    for idx in range(start_idx, end_idx):
        row = df_clean.iloc[idx]

        sensor_values = row[sensor_columns].values.astype(np.float32)
        finger_values = row[finger_columns].values.astype(np.float32)

        inference_engine.process_sample(sensor_values)

        prediction = inference_engine.predict()

        if prediction is not None:
            predictions.append(prediction)
            ground_truth.append(finger_values)

        if (idx - start_idx + 1) % 100 == 0:
            print(f"  Processed {idx - start_idx + 1}/{num_samples} samples...")

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

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
    model_path = "training/notebooks/best_lstm_model.pth"
    scaler_path = "training/notebooks/scaler_inputs_lstm.pkl"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return

    print("Initializing streaming inference engine...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_engine = StreamingInference(
        model_path=model_path, scaler_path=scaler_path, window_size=30, device=device
    )
    print(f"Model loaded successfully on {device}")
    print(f"Model num_layers: {inference_engine.model.num_layers}")
    print(f"Model hidden_size: {inference_engine.model.hidden_size}")

    print("\nLoading test data...")
    df_clean = load_test_data()
    print(f"Loaded {len(df_clean)} samples")

    # DEBUG: Check data ranges
    sensor_columns = ["env0", "raw0", "env1", "raw1", "env2", "raw2", "env3", "raw3"]
    print("\nData ranges after loading:")
    for col in sensor_columns:
        vals = df_clean[col].values
        print(
            f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}"
        )

    predictions, ground_truth = simulate_streaming(
        inference_engine, df_clean, num_samples=1000, start_idx=1000
    )

    print(f"\nGenerated {len(predictions)} predictions")

    compute_metrics(predictions, ground_truth)

    visualize_predictions(predictions, ground_truth, n_samples=200)


if __name__ == "__main__":
    main()
