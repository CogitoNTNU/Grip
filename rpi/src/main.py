import sys

import joblib
import numpy as np
import serial
import torch

if sys.platform == "darwin":
    PORT = "/dev/tty.usbmodem11301"
else:
    PORT = "/dev/ttyAMA0"

BAUDRATE = 115200

NUM_INPUTS = 16
NUM_OUTPUTS = 6
WINDOW = 30


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


model = LSTMModel(NUM_INPUTS, NUM_OUTPUTS)
model.load_state_dict(torch.load("models/nn/lstm_model_final.pth", map_location="cpu"))
model.eval()

scaler = joblib.load("training/notebooks/scaler_inputs_lstm.pkl")


def engineer_features(sensor_values):
    """
    Convert 8 sensor values to 16 features with neighbor differences.
    sensor_values: [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
    returns: [env0, raw0, env1, raw1, env2, raw2, env3, raw3,
              raw_diff1, raw_diff2, raw_diff3, raw_diff4,
              env_diff1, env_diff2, env_diff3, env_diff4]
    """
    env0, raw0, env1, raw1, env2, raw2, env3, raw3 = sensor_values

    # Neighbor relationships (1-indexed in training)
    # 1: no neighbors, 2: [3], 3: [2,4], 4: [3]
    raw_diff1 = 0.0
    raw_diff2 = raw1 - raw2  # sensor 2 - sensor 3
    raw_diff3 = (raw2 - raw1 + raw2 - raw3) / 2  # avg of (sensor 3 - sensor 2) and (sensor 3 - sensor 4)
    raw_diff4 = raw3 - raw2  # sensor 4 - sensor 3

    env_diff1 = 0.0
    env_diff2 = env1 - env2
    env_diff3 = (env2 - env1 + env2 - env3) / 2
    env_diff4 = env3 - env2

    return np.array([
        env0, raw0, env1, raw1, env2, raw2, env3, raw3,
        raw_diff1, raw_diff2, raw_diff3, raw_diff4,
        env_diff1, env_diff2, env_diff3, env_diff4
    ], dtype=np.float32)


buf = []
ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)

while True:
    try:
        line = ser.readline().decode().strip()

        if not line:
            continue

        parts = line.split(",")
        if len(parts) != 8:
            continue

        sensor_values = np.array([float(x) for x in parts], dtype=np.float32)
        print(f"Received: {sensor_values}")

        features = engineer_features(sensor_values)
        features_scaled = scaler.transform(features.reshape(1, -1))[0]

        buf.append(features_scaled)
        if len(buf) < WINDOW:
            continue

        if len(buf) > WINDOW:
            buf = buf[-WINDOW:]

        X = np.stack(buf)
        X = X.reshape(1, WINDOW, NUM_INPUTS)
        X = torch.from_numpy(X)

        with torch.no_grad():
            out = model(X)

        servo = out.numpy()[0] * 1023
        servo = np.clip((servo), 0, 1023).astype(int)
        print(f"Sent: {servo}")

        msg = ",".join(map(str, servo)) + "\n"
        ser.write(msg.encode())
    except Exception as e:
        print("Error:", e)
        continue
