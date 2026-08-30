from pathlib import Path
import sys

from ssl4eo_l_dataset import SSL4EOLDataset

ROOT = sys.argv[1] if len(sys.argv) > 1 else "preprocessed_data"

print("=" * 60)
print("SSL4EO-L PYTORCH LOADER VERIFICATION")
print("=" * 60)
print(f"Dataset root: {Path(ROOT).resolve()}")

for split, expected in [
    ("train", 17500),
    ("val", 3750),
    ("test", 3750),
]:
    dataset = SSL4EOLDataset(ROOT, split=split)

    assert len(dataset) == expected, (
        f"{split}: expected {expected}, got {len(dataset)}"
    )

    sample = dataset[0]

    assert tuple(sample["optical"].shape) == (7, 264, 264)
    assert tuple(sample["thermal"].shape) == (2, 132, 132)

    assert sample["optical"].dtype.name == "torch.float32"
    assert sample["thermal"].dtype.name == "torch.float32"

    print(
        f"{split.upper():5} PASS | "
        f"samples={len(dataset):5} | "
        f"optical={tuple(sample['optical'].shape)} | "
        f"thermal={tuple(sample['thermal'].shape)} | "
        f"id={sample['sample_id']}"
    )

print("=" * 60)
print("✓ PYTORCH DATASET LOADER VERIFIED")
print("=" * 60)
