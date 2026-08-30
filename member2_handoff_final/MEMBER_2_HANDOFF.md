# SSL4EO-L — Member 2 Data Handoff

This is the ready-to-load preprocessed dataset for the ML/model pipeline.

## Dataset
- Train: 17,500
- Validation: 3,750
- Test: 3,750
- Total: 25,000

## Per-sample arrays

| Array | Shape | dtype | Range | Role |
|---|---|---|---|---|
| optical | `(7, 264, 264)` | `float32` | `[0,1]` | Optical input |
| thermal_lr | `(2, 132, 132)` | `float32` | `[0,1]` | Low-resolution thermal input |
| thermal_hr | `(2, 264, 264)` | `float32` | `[0,1]` | High-resolution thermal target |

All three files use the same 7-digit sample ID.

Example:
```text
train/optical/0000000.npy
train/thermal_lr/0000000.npy
train/thermal_hr/0000000.npy
```

## Structure
```text
preprocessed_data/
├── metadata/preprocessing_info.json
├── train/{optical,thermal_lr,thermal_hr}/
├── val/{optical,thermal_lr,thermal_hr}/
└── test/{optical,thermal_lr,thermal_hr}/
```

## PyTorch usage
Copy `ssl4eo_l_dataset.py` next to the model project:

```python
from ssl4eo_l_dataset import SSL4EOLDataset
from torch.utils.data import DataLoader

train_dataset = SSL4EOLDataset("preprocessed_data", "train")
val_dataset = SSL4EOLDataset("preprocessed_data", "val")
test_dataset = SSL4EOLDataset("preprocessed_data", "test")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
```

Each batch provides:
```python
batch["optical"]       # [B, 7, 264, 264]
batch["thermal_lr"]    # [B, 2, 132, 132]
batch["thermal_hr"]    # [B, 2, 264, 264]
batch["sample_id"]
```

The data is already normalized to `[0,1]` and stored as `float32`.

## Verify after copying
```bash
python check_handoff.py
python ssl4eo_l_dataset.py
```

The original raw TIFF dataset is not required for normal model training.
