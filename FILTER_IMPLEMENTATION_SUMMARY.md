# Filter Strategy Pattern Implementation Summary

## Overview

Implemented a **Strategy Pattern** for EMG signal filtering that provides:

- ✅ Multiple filter options (high-pass, low-pass, band-pass, moving average, no filter)
- ✅ Consistent filtering between training and inference
- ✅ Easy experimentation with different filters
- ✅ Automatic configuration persistence in model checkpoints

## Files Created/Modified

### New Files

1. **`training/filter_strategies.py`** (389 lines)

   - Abstract `FilterStrategy` base class
   - 5 concrete filter implementations
   - Factory function `create_filter()`
   - Comprehensive testing in `__main__`

1. **`FILTER_STRATEGY_GUIDE.md`** (Complete user guide)

   - Detailed documentation for each filter type
   - Usage examples for training and inference
   - Filter selection decision tree
   - Troubleshooting tips

### Modified Files

1. **`training/inference_streaming.py`**

   - Added filter strategy imports
   - Updated `StreamingInference.__init__()` to accept filter config
   - Replaced hardcoded high-pass filter with strategy pattern
   - Updated `load_test_data()` to accept filter strategy
   - Modified `main()` to load filter config from model checkpoint
   - Removed old `StreamingHighPassFilter` class

1. **`training/notebooks/LSTM_spacial_filtering.ipynb`**

   - Added filter configuration cells at top
   - Replaced hardcoded `highpass_causal()` with filter strategy
   - Updated model saving to include filter configuration in hyperparameters

## Filter Strategy Classes

```
FilterStrategy (ABC)
├── NoFilterStrategy - Pass-through (no filtering)
├── HighPassFilterStrategy - Remove DC bias and drift
├── LowPassFilterStrategy - Smoothing, noise reduction
├── BandPassFilterStrategy - Combination filtering
└── MovingAverageFilterStrategy - Simple FIR smoothing
```

Each strategy implements:

- `apply_batch(signal)` - For training (batch processing)
- `apply_streaming(value, channel)` - For inference (streaming)
- `get_description()` - Human-readable description

## Usage

### In Training Notebook

```python
# Configure filter
FILTER_TYPE = "highpass"
FILTER_CONFIG = {"cutoff": 0.5, "order": 4}

# Create and apply filter
filter_strategy = create_filter(FILTER_TYPE, fs, **FILTER_CONFIG)
for col in ["env0", "env1", "env2", "env3"]:
    df_clean[col] = filter_strategy.apply_batch(df_clean[col].values)

# Save with model
torch.save(
    {
        "hyperparameters": {
            "filter_type": FILTER_TYPE,
            "filter_config": FILTER_CONFIG,
            "sampling_rate": MEASURED_SAMPLE_RATE,
        }
    },
    "model.pth",
)
```

### In Inference Script

```python
# Filter config automatically loaded from checkpoint
checkpoint = torch.load(model_path)
FILTER_TYPE = checkpoint["hyperparameters"]["filter_type"]
FILTER_CONFIG = checkpoint["hyperparameters"]["filter_config"]

# Create inference engine with loaded config
inference_engine = StreamingInference(
    filter_type=FILTER_TYPE,
    filter_config=FILTER_CONFIG,
)
```

## Available Filters

| Filter Type      | Use Case                         | Typical Config                                        |
| ---------------- | -------------------------------- | ----------------------------------------------------- |
| `none`           | Raw data, no filtering           | `{}`                                                  |
| `highpass`       | Remove DC bias, drift            | `{'cutoff': 0.5, 'order': 4}`                         |
| `lowpass`        | Smoothing, noise reduction       | `{'cutoff': 2.0, 'order': 4}`                         |
| `bandpass`       | Both drift removal and smoothing | `{'low_cutoff': 0.5, 'high_cutoff': 2.0, 'order': 4}` |
| `moving_average` | Simple smoothing                 | `{'window_size': 5}`                                  |

## Key Features

### 1. Consistency Between Training and Inference

- Same filter applied in both batch (training) and streaming (inference) modes
- Verified: batch and streaming produce identical results (tested in module)
- Filter config automatically saved and loaded with model

### 2. Easy Experimentation

- Change `FILTER_TYPE` and `FILTER_CONFIG` at top of notebook
- No need to modify filtering code
- All filters use same interface

### 3. Proper Causal Filtering

- All filters use `lfilter` (causal) not `filtfilt` (acausal)
- Compatible with real-time streaming inference
- Maintains filter state across streaming samples

### 4. Comprehensive Documentation

- Inline docstrings for all classes and methods
- Complete user guide (`FILTER_STRATEGY_GUIDE.md`)
- Example usage in module's `__main__`

## Testing

Run the filter strategies module to verify all filters work:

```bash
python -m training.filter_strategies
```

Output shows:

- All 5 filters tested on synthetic signal
- Batch and streaming modes match perfectly (✅ MATCH)
- Verified numerically (max difference < 1e-10)

## Benefits

1. **Flexibility**: Easy to try different filters without changing code
1. **Maintainability**: Single source of truth for filtering logic
1. **Consistency**: Guaranteed same filtering in training and inference
1. **Extensibility**: Easy to add new filter types (just extend `FilterStrategy`)
1. **Debuggability**: Clear filter description in output logs

## Example Output (Inference)

```
✅ Loaded filter configuration from model checkpoint:
   Filter type: highpass
   Filter config: {'cutoff': 0.5, 'order': 4}
   Sampling rate: 4.2144 Hz

======================================================================
FILTER CONFIGURATION
======================================================================
  High-Pass Filter (cutoff=0.5Hz, order=4, fs=4.2144Hz)
======================================================================

======================================================================
APPLYING BATCH FILTERING TO ENV CHANNELS
======================================================================
  High-Pass Filter (cutoff=0.5Hz, order=4, fs=4.2144Hz)
======================================================================
```

## Migration Notes

### Before (Hardcoded)

```python
# Old approach - hardcoded high-pass filter
def highpass_causal(signal, fs, cutoff=0.5, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype="high")
    return lfilter(b, a, signal)


# Scattered in multiple places
df_clean[col] = highpass_causal(df_clean[col].values, fs)
```

### After (Strategy Pattern)

```python
# New approach - configurable strategy
filter_strategy = create_filter("highpass", fs, cutoff=0.5, order=4)
df_clean[col] = filter_strategy.apply_batch(df_clean[col].values)

# Or try different filter easily:
filter_strategy = create_filter("lowpass", fs, cutoff=2.0, order=4)
```

## Next Steps

Users can now:

1. Experiment with different filters by changing `FILTER_TYPE`
1. Compare performance across filter types
1. Optimize filter parameters for their specific data
1. Add custom filters by extending `FilterStrategy`

See `FILTER_STRATEGY_GUIDE.md` for complete usage instructions.
