# Member 4 — Integration Handoff

This folder contains a small, ready-to-use subset of the
preprocessed SSL4EO-L OLI/TIRS TOA dataset for Streamlit,
visualization, and final application integration.

## Dataset

Dataset: SSL4EO-L OLI/TIRS TOA

Complete dataset: 25,000 samples

Samples included in this handoff: 5

## Data Format

Each sample contains three NumPy arrays.

### Optical

Shape: `(7, 264, 264)`

dtype: `float32`

Range: `[0, 1]`

### Thermal LR

Shape: `(2, 132, 132)`

dtype: `float32`

Range: `[0, 1]`

This is the low-resolution thermal input.

### Thermal HR

Shape: `(2, 264, 264)`

dtype: `float32`

Range: `[0, 1]`

This is the high-resolution thermal reference/target.

## Loading a Sample

```python
from load_sample import load_sample

optical, thermal_lr, thermal_hr = load_sample(1)
```

Expected shapes:

```text
Optical     : (7, 264, 264)
Thermal LR  : (2, 132, 132)
Thermal HR  : (2, 264, 264)
```

## Sample IDs

- sample_001: `0000000`
- sample_002: `0000002`
- sample_003: `0000003`
- sample_004: `0000004`
- sample_005: `0000006`

## Preprocessing

The arrays are already preprocessed.

```text
dtype       : float32
value range : [0, 1]
NaN / Inf   : none
```

No original-dataset preprocessing is required for
these samples.

## Streamlit Integration

Member 4 can use these samples to:

1. Load an optical sample.
2. Load the low-resolution thermal image.
3. Display the input images.
4. Pass the input to Member 2's ML model.
5. Receive the super-resolution output.
6. Display the predicted high-resolution thermal image.
7. Compare the prediction with the thermal HR reference.

## Folder Structure

```text
member_4_handoff/
├── README.md
├── preprocessing_info.json
├── load_sample.py
└── sample_data/
    ├── sample_001/
    │   ├── optical.npy
    │   ├── thermal_lr.npy
    │   └── thermal_hr.npy
    ├── sample_002/
    ├── sample_003/
    ├── sample_004/
    └── sample_005/
```

## Important

This handoff contains only a small demonstration subset.

The complete 25,000-sample dataset is NOT included here.

The original dataset and full preprocessed dataset remain
outside this handoff package.

## Metadata

See `preprocessing_info.json` for preprocessing information.
