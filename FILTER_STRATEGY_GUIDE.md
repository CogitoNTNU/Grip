# Filter Strategy Pattern Guide

## Overview

The EMG processing pipeline uses the **Strategy Pattern** for filtering, allowing easy experimentation with different filter types while maintaining consistency between training and inference.

## Architecture

```
FilterStrategy (Abstract Base Class)
├── NoFilterStrategy (pass-through)
├── HighPassFilterStrategy (remove DC bias, drift)
├── LowPassFilterStrategy (smoothing, noise reduction)
├── BandPassFilterStrategy (combination of high + low pass)
└── MovingAverageFilterStrategy (simple FIR smoothing)
```

All filters implement:

- `apply_batch(signal)` - For training (batch mode)
- `apply_streaming(value, channel)` - For inference (streaming mode)
- `get_description()` - Human-readable filter description

## Available Filters

### 1. No Filter (Pass-Through)

**Use when:** You want raw, unfiltered data

**Training Configuration:**

```python
FILTER_TYPE = "none"
FILTER_CONFIG = {}
```

**Inference:** Automatically loaded from model checkpoint

**Characteristics:**

- ✅ No phase delay
- ✅ No artifacts
- ❌ Includes DC bias and drift
- ❌ Includes all noise

______________________________________________________________________

### 2. High-Pass Filter (Butterworth)

**Use when:** Remove DC bias and slow baseline drift

**Training Configuration:**

```python
FILTER_TYPE = "highpass"
FILTER_CONFIG = {"cutoff": 0.5, "order": 4}
```

**Parameters:**

- `cutoff`: Frequency in Hz (frequencies below this are attenuated)
  - Typical: 0.5 Hz (removes very slow drift)
- `order`: Filter sharpness (higher = sharper cutoff, more ringing)
  - Typical: 4 (good balance)

**Characteristics:**

- ✅ Removes DC bias (centering around zero)
- ✅ Removes slow baseline drift
- ⚠️ Phase delay at cutoff frequency
- ⚠️ Initial transient (~100 samples)

**Good for:** EMG signals with baseline wander, DC offset

______________________________________________________________________

### 3. Low-Pass Filter (Butterworth)

**Use when:** Smooth signals, remove high-frequency noise

**Training Configuration:**

```python
FILTER_TYPE = "lowpass"
FILTER_CONFIG = {"cutoff": 2.0, "order": 4}
```

**Parameters:**

- `cutoff`: Frequency in Hz (frequencies above this are attenuated)
  - Typical: 2.0 Hz (smooth without losing too much detail)
  - ⚠️ Must be < Nyquist frequency (fs/2 = 2.1 Hz for fs=4.2144 Hz)
- `order`: Filter sharpness
  - Typical: 4

**Characteristics:**

- ✅ Smooths noisy signals
- ✅ Reduces high-frequency artifacts
- ⚠️ Can blur rapid changes
- ⚠️ Phase delay
- ⚠️ Initial transient

**Good for:** Very noisy signals, when smooth predictions are desired

______________________________________________________________________

### 4. Band-Pass Filter (Butterworth)

**Use when:** Keep only a specific frequency range

**Training Configuration:**

```python
FILTER_TYPE = "bandpass"
FILTER_CONFIG = {"low_cutoff": 0.5, "high_cutoff": 2.0, "order": 4}
```

**Parameters:**

- `low_cutoff`: High-pass cutoff (remove frequencies below this)
- `high_cutoff`: Low-pass cutoff (remove frequencies above this)
  - Must satisfy: `low_cutoff < high_cutoff < Nyquist`
- `order`: Filter sharpness

**Characteristics:**

- ✅ Combines benefits of high-pass and low-pass
- ✅ Removes both DC drift AND high-frequency noise
- ⚠️ More complex filter (higher order)
- ⚠️ Longer transient

**Good for:** When you need both drift removal and smoothing

______________________________________________________________________

### 5. Moving Average Filter (FIR)

**Use when:** Simple smoothing with minimal complexity

**Training Configuration:**

```python
FILTER_TYPE = "moving_average"
FILTER_CONFIG = {"window_size": 5}
```

**Parameters:**

- `window_size`: Number of samples to average
  - Typical: 3-10 samples
  - Larger = smoother but more lag

**Characteristics:**

- ✅ Very simple, no ringing
- ✅ Linear phase (predictable delay)
- ⚠️ Fixed delay = (window_size - 1) / 2 samples
- ❌ Less effective than Butterworth for sharp cutoffs
- ❌ Doesn't remove DC bias

**Good for:** Simple smoothing without complexity

______________________________________________________________________

## Usage

### In Training Notebook

1. **Configure filter at the top:**

```python
# Import filter strategies
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), ".."))
from training.filter_strategies import create_filter

# Choose your filter
FILTER_TYPE = "highpass"
FILTER_CONFIG = {"cutoff": 0.5, "order": 4}
```

2. **Apply filter to data:**

```python
# Use measured sample rate
fs = MEASURED_SAMPLE_RATE

# Create filter
filter_strategy = create_filter(FILTER_TYPE, fs, **FILTER_CONFIG)

# Apply to env channels
for col in ["env0", "env1", "env2", "env3"]:
    df_clean[col] = filter_strategy.apply_batch(df_clean[col].values)
```

3. **Save filter config with model:**

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "hyperparameters": {
            # ... other hyperparameters ...
            "filter_type": FILTER_TYPE,
            "filter_config": FILTER_CONFIG,
            "sampling_rate": MEASURED_SAMPLE_RATE,
        },
    },
    "model.pth",
)
```

### In Inference Script

The inference script **automatically loads** filter configuration from the model checkpoint:

```python
# Filter config is loaded from checkpoint automatically
checkpoint = torch.load(model_path)
FILTER_TYPE = checkpoint["hyperparameters"]["filter_type"]
FILTER_CONFIG = checkpoint["hyperparameters"]["filter_config"]

# You can override if needed:
# FILTER_TYPE = 'none'
# FILTER_CONFIG = {}
```

### Direct API Usage

```python
from training.filter_strategies import create_filter

# Create filter
fs = 4.2144  # Sample rate
filter_strategy = create_filter("highpass", fs, cutoff=0.5, order=4)

# Batch mode (training)
filtered_signal = filter_strategy.apply_batch(raw_signal)

# Streaming mode (inference)
for sample in samples:
    filtered_value = filter_strategy.apply_streaming(sample, channel=0)
```

## Filter Selection Guide

### Decision Tree

```
Are you experiencing DC offset or baseline drift?
├─ YES → Use 'highpass' (cutoff=0.5, order=4)
└─ NO
    └─ Is your signal very noisy?
        ├─ YES → Try 'lowpass' (cutoff=2.0) OR 'moving_average' (window=5)
        └─ NO
            └─ Do you want both drift removal AND smoothing?
                ├─ YES → Use 'bandpass' (low=0.5, high=2.0, order=4)
                └─ NO → Use 'none' (no filtering)
```

### Recommended Starting Points

| Signal Characteristic | Recommended Filter | Config                                                |
| --------------------- | ------------------ | ----------------------------------------------------- |
| Clean, no drift       | `none`             | `{}`                                                  |
| DC bias, slow drift   | `highpass`         | `{'cutoff': 0.5, 'order': 4}`                         |
| High-frequency noise  | `lowpass`          | `{'cutoff': 2.0, 'order': 4}`                         |
| Both drift AND noise  | `bandpass`         | `{'low_cutoff': 0.5, 'high_cutoff': 2.0, 'order': 4}` |
| Simple smoothing      | `moving_average`   | `{'window_size': 5}`                                  |

## Important Notes

### ⚠️ Sampling Rate Constraints

- All cutoff frequencies must be **less than Nyquist frequency** (fs/2)
- For fs = 4.2144 Hz, Nyquist = 2.107 Hz
- Low-pass and band-pass filters will fail if cutoff ≥ Nyquist

### ⚠️ Filter Transients

- All filters (except `none`) have initial transients
- First ~100 samples may have artifacts
- Training: Skip early samples or accept transient in training data
- Inference: Allow warmup period before making predictions

### ⚠️ Consistency is Critical

- **MUST** use same filter in training and inference!
- Filter config is automatically saved and loaded
- Mismatch causes poor inference performance

### ⚠️ Phase Delay

- IIR filters (Butterworth) introduce phase delay
- Moving average has fixed delay: (window_size - 1) / 2 samples
- All filters use `lfilter` (causal) for real-time compatibility

## Experimentation Tips

1. **Start with default:** `highpass` with cutoff=0.5 Hz
1. **Visualize filtered data** to check for artifacts
1. **Monitor training metrics** - filter affects learning
1. **Test multiple filters** and compare performance
1. **Remember:** The best filter depends on your specific data!

## Troubleshooting

### Problem: "ValueError: Cutoff frequency too high"

**Solution:** Reduce cutoff frequency to be < Nyquist (fs/2)

### Problem: Poor inference performance

**Solution:** Check filter config matches between training and inference

### Problem: Oscillations or ringing in signal

**Solution:** Reduce filter order (e.g., from 4 to 2)

### Problem: Signal looks distorted

**Solution:** Try `none` filter to see raw data, then adjust cutoff

## Examples

See `training/filter_strategies.py` for complete implementation and examples.

Run the filter strategies module directly to see comparative examples:

```bash
python -m training.filter_strategies
```
