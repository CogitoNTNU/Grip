# Data Processing Order Comparison

## NOTEBOOK PIPELINE (Correct Order)

### Cell 3: Load Data

- Load CSV files from `data/martin2/raw`
- Convert to numeric
- Drop NaN rows
- Result: `df_clean` with raw data

### Cell 5: High-Pass Filter ENV columns

```python
for col in ["env0", "env1", "env2", "env3"]:
    df_clean[col] = highpass(df_clean[col].values, fs, cutoff=0.5)
```

- **FILTERS ONLY env0-3, NOT raw0-3**
- Uses `filtfilt` (non-causal, zero-phase filter on entire dataset)

### Cell 6: OLD Scaling (8 features) - GETS OVERWRITTEN

```python
sensor_columns = ["env0", "raw0", "env1", "raw1", "env2", "raw2", "env3", "raw3"]
X = df_clean[sensor_columns].values
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)  # 8 features
```

### Cell 7: Save OLD scaler (8 features) - GETS OVERWRITTEN

```python
joblib.dump(scaler_X, "scaler_inputs_lstm.pkl")  # 8 features!
```

### Cell 9: Spatial Features + NEW Scaling (16 features)

```python
# Compute spatial features from FILTERED env and UNFILTERED raw
neighbors = {1: [], 2: [3], 3: [2, 4], 4: [3]}

for base in range(1, 5):
    raw_diffs = [
        df_clean[f"raw{base - 1}"] - df_clean[f"raw{n - 1}"] for n in neighbors[base]
    ]
    env_diffs = [
        df_clean[f"env{base - 1}"] - df_clean[f"env{n - 1}"] for n in neighbors[base]
    ]
    df_clean[f"raw_diff{base}"] = sum(raw_diffs) / len(raw_diffs)
    df_clean[f"env_diff{base}"] = sum(env_diffs) / len(env_diffs)

# Create 16-feature X
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

X = df_clean[sensor_columns].values

# Create NEW scaler for 16 features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)  # 16 features!
```

### Cell 10: Save NEW scaler (16 features) - **MUST BE RUN!**

```python
joblib.dump(scaler_X, "scaler_inputs_lstm.pkl")  # 16 features!
print(f"Scaler saved with {scaler_X.n_features_in_} features")
```

**⚠️ THIS CELL MUST BE EXECUTED to overwrite the old 8-feature scaler!**

### Cell 13: Create Windows

```python
X_win, y_win = create_windows(X_scaled, y_scaled, window_size=30, stride=5)
```

- Windows are created from ALREADY SCALED data

### Cell 19: Train Model

```python
model = LSTMModel(n_inputs=16, n_outputs=6, hidden_size=128, num_layers=3, dropout=0.2)
```

______________________________________________________________________

## INFERENCE PIPELINE (Should Match Notebook)

### 1. load_test_data()

```python
# Load CSV
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Convert to numeric, drop NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df_clean = df.dropna(subset=numeric_columns)

# High-pass filter ENV columns (matching notebook)
fs = 1.0 / 0.03446
for col in ["env0", "env1", "env2", "env3"]:
    df_clean[col] = highpass(df_clean[col].values, fs, cutoff=0.5)
```

✅ **Matches notebook Cell 5**

### 2. simulate_streaming()

```python
sensor_columns = ["env0", "raw0", "env1", "raw1", "env2", "raw2", "env3", "raw3"]
for idx in range(start_idx, end_idx):
    row = df_clean.iloc[idx]
    sensor_values = row[sensor_columns].values  # 8 values
    inference_engine.process_sample(sensor_values)
```

✅ **Extracts data in correct order**

### 3. process_sample()

```python
# Extract filtered env and unfiltered raw
env_values = raw_sample[::2]  # [env0, env1, env2, env3] (FILTERED)
raw_values = raw_sample[1::2]  # [raw0, raw1, raw2, raw3] (UNFILTERED)

# Compute spatial features (same as notebook)
raw_diffs, env_diffs = self.compute_spatial_features(raw_values, env_values)

# Create 16-feature vector in same order as notebook
interleaved_sensors = []
for i in range(4):
    interleaved_sensors.append(env_values[i])
    interleaved_sensors.append(raw_values[i])

features = np.concatenate(
    [
        interleaved_sensors,  # [env0, raw0, env1, raw1, env2, raw2, env3, raw3]
        raw_diffs,  # [raw_diff1, raw_diff2, raw_diff3, raw_diff4]
        env_diffs,  # [env_diff1, env_diff2, env_diff3, env_diff4]
    ]
)

# Scale using pre-fitted scaler
features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
```

✅ **Matches notebook Cell 9 processing**

### 4. predict()

```python
# Use sliding window buffer of 30 scaled samples
window_array = np.array(list(self.window_buffer))  # (30, 16)
window_tensor = torch.tensor(window_array).unsqueeze(0)  # (1, 30, 16)

prediction = self.model(window_tensor)
```

✅ **Matches notebook Cell 13 windowing**

______________________________________________________________________

## CHECKLIST FOR CORRECT OPERATION

- [ ] **Notebook Cell 10 has been executed** (saves 16-feature scaler)
- [ ] **Scaler file has 16 features**: Run `python -c "import joblib; s=joblib.load('training/notebooks/scaler_inputs_lstm.pkl'); print(s.n_features_in_)"`
  - Should print: `16`
- [ ] **Model has 3 LSTM layers**: Check model initialization in inference script
- [ ] **Same data source**: Both use `data/martin2/raw`
- [ ] **High-pass filter applied to env0-3 only**
- [ ] **raw0-3 remain unfiltered**
- [ ] **Spatial features computed correctly**
- [ ] **Feature order matches**: `[env0, raw0, env1, raw1, ..., raw_diff1-4, env_diff1-4]`

______________________________________________________________________

## DEBUGGING COMMANDS

```bash
# Check scaler features
python -c "import joblib; s=joblib.load('training/notebooks/scaler_inputs_lstm.pkl'); print('Scaler features:', s.n_features_in_)"

# Check model layers
python -c "import torch; m=torch.load('training/notebooks/best_lstm_model.pth', weights_only=False); layers=[k for k in m.keys() if 'lstm_layers' in k]; print('Model has', len(set([k.split('.')[1] for k in layers if 'lstm_layers.' in k])), 'LSTM layers')"

# Verify data shape
python -c "import pandas as pd; import glob; files=glob.glob('data/martin2/raw/integrated_data_*.csv'); df=pd.concat([pd.read_csv(f) for f in files]); print('Data shape:', df.shape)"
```
