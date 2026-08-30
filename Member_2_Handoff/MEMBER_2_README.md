# SSL4EO-L Preprocessed Dataset — Member 2 Handoff

## 1. What this dataset contains

This is the preprocessed subset of the SSL4EO-L OLI/TIRS TOA benchmark.

The original dataset contains 25,000 samples.

The original `.tif` files were **not modified**.

This folder contains a separate preprocessed copy intended for model development.

## 2. Dataset split
```text
| Split | Samples |
|-------|--------:|
| Train | 17,500  |
| Validation | 3,750 |
| Test | 3,750 |
| **Total** | **25,000** |
```
Split ratio: **70% / 15% / 15%**

## 3. Directory structure

```text
preprocessed_data/
│
├── train/
│   ├── optical/
│   │   ├── 0000000.npy
│   │   ├── 0000002.npy
│   │   └── ...
│   │
│   └── thermal/
│       ├── 0000000.npy
│       ├── 0000002.npy
│       └── ...
│
├── val/
│   ├── optical/
│   └── thermal/
│
└── test/
    ├── optical/
    └── thermal/
```

Each `sample_id` has one optical file and one matching thermal file.

Example:

```text
train/optical/0000000.npy
train/thermal/0000000.npy
```

These two files belong to the same original satellite scene.

## 4. Array format

### Optical

```text
Shape : (7, 264, 264)
Dtype : float32
Range : approximately 0–1
```

The seven optical channels correspond to:

```text
B2
B3
B4
B5
B6
B7
B8
```

### Thermal

```text
Shape : (2, 132, 132)
Dtype : float32
Range : approximately 0–1
```

The two thermal channels correspond to:

```text
B10
B11
```

## 5. Important: no thermal_lr files are stored

The handoff currently contains only:

```text
optical
thermal
```

There is no separate `thermal_lr` directory.

The thermal data at `(2, 132, 132)` is the lower-resolution thermal representation.

If the model requires a low-resolution thermal input to be brought back to another spatial size, Member 2 can perform that operation dynamically during training.

This avoids storing another 25,000 copies of the thermal data.

## 6. PyTorch loader

Copy `ssl4eo_l_dataset.py` into the model project.

Then:

```python
from ssl4eo_l_dataset import SSL4EOLDataset

train_dataset = SSL4EOLDataset(
    root="preprocessed_data",
    split="train"
)

val_dataset = SSL4EOLDataset(
    root="preprocessed_data",
    split="val"
)

test_dataset = SSL4EOLDataset(
    root="preprocessed_data",
    split="test"
)
```

Use with PyTorch DataLoader:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)
```

Example batch:

```python
batch = next(iter(train_loader))

print(batch["optical"].shape)
print(batch["thermal"].shape)
```

Expected:

```text
torch.Size([8, 7, 264, 264])
torch.Size([8, 2, 132, 132])
```

The batch size can be changed according to GPU memory.

## 7. Important training consideration

The dataset loader returns the preprocessed observations.

It does **not** automatically create a degraded thermal image.

If the model architecture expects:

```text
Optical + degraded thermal → reconstructed thermal
```

then the degradation/downsampling/upsampling operation should be implemented in the model/training pipeline.

For example:

```text
thermal (2,132,132)
        │
        ├── resize / degradation as required
        ↓
thermal input
        │
        ├── combined with optical
        ↓
      model
        │
        ↓
predicted thermal
```

The exact degradation strategy should match the final model design.

## 8. Verification already completed

The complete 25,000-sample dataset was verified.

Results:

```text
TRAIN
  17,500 samples
  Optical files: 17,500
  Thermal files: 17,500
  Shape errors: 0
  Dtype errors: 0
  NaN/Inf errors: 0
  Range errors: 0
  Load errors: 0

VAL
  3,750 samples
  Optical files: 3,750
  Thermal files: 3,750
  Shape errors: 0
  Dtype errors: 0
  NaN/Inf errors: 0
  Range errors: 0
  Load errors: 0

TEST
  3,750 samples
  Optical files: 3,750
  Thermal files: 3,750
  Shape errors: 0
  Dtype errors: 0
  NaN/Inf errors: 0
  Range errors: 0
  Load errors: 0
```

Final verification result:

```text
TRAIN : PASS
VAL   : PASS
TEST  : PASS
```

## 9. Original dataset location

The original dataset should remain untouched.

Original:

```text
ssl4eo_l_oli_tirs_toa_benchmark/
└── ssl4eo_l_oli_tirs_toa_benchmark/
    ├── 0000000/
    ├── 0000002/
    ├── ...
    └── 0029442/
```

The preprocessed dataset is a separate copy.

## 10. What Member 2 needs to do

After extracting/copying `preprocessed_data`:

1. Put it somewhere accessible to the model project.
2. Copy `ssl4eo_l_dataset.py`.
3. Run:

```bash
python ssl4eo_l_dataset.py
```

4. Confirm:

```text
Samples     : 17500
Optical     : (7, 264, 264)
Thermal     : (2, 132, 132)
```

5. Create the DataLoaders.
6. Implement the model/training pipeline.

## 11. File naming convention

The filename is the sample identifier:

```text
0000000.npy
0000002.npy
0000003.npy
...
```

Do not rename individual files unless the corresponding optical and thermal files are renamed identically.

## 12. Recommended transfer

The entire `preprocessed_data` folder can be compressed and uploaded to Google Drive.

Member 2 only needs the preprocessed dataset for model training.

The original 11 GB raw dataset does not need to be transferred unless Member 2 specifically needs access to the original TIFF metadata/data.
