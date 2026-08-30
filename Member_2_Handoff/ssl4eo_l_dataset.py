"""
PyTorch Dataset loader for the SSL4EO-L preprocessed dataset.

Expected directory structure:
preprocessed_data/
├── train/
│   ├── optical/
│   └── thermal/
├── val/
│   ├── optical/
│   └── thermal/
└── test/
    ├── optical/
    └── thermal/

Each optical file:
    shape = (7, 264, 264), dtype=float32

Each thermal file:
    shape = (2, 132, 132), dtype=float32

The same sample_id must exist in both optical/ and thermal/.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SSL4EOLDataset(Dataset):
    """Load one split of the preprocessed SSL4EO-L dataset."""

    VALID_SPLITS = ("train", "val", "test")

    def __init__(self, root, split="train", return_id=True):
        self.root = Path(root)
        self.split = split
        self.return_id = return_id

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Choose from {self.VALID_SPLITS}."
            )

        self.optical_dir = self.root / split / "optical"
        self.thermal_dir = self.root / split / "thermal"

        if not self.optical_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.optical_dir}")
        if not self.thermal_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.thermal_dir}")

        self.sample_ids = sorted(
            p.stem for p in self.optical_dir.glob("*.npy")
        )

        if not self.sample_ids:
            raise RuntimeError(
                f"No .npy optical files found in {self.optical_dir}"
            )

        # Verify that every optical sample has a matching thermal sample.
        missing_thermal = [
            sid for sid in self.sample_ids
            if not (self.thermal_dir / f"{sid}.npy").exists()
        ]

        if missing_thermal:
            preview = ", ".join(missing_thermal[:10])
            raise RuntimeError(
                f"{len(missing_thermal)} optical samples have no "
                f"matching thermal file. Examples: {preview}"
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        optical_path = self.optical_dir / f"{sample_id}.npy"
        thermal_path = self.thermal_dir / f"{sample_id}.npy"

        optical = np.load(optical_path)
        thermal = np.load(thermal_path)

        # Safety checks.
        if optical.shape != (7, 264, 264):
            raise ValueError(
                f"{optical_path}: expected (7, 264, 264), "
                f"got {optical.shape}"
            )

        if thermal.shape != (2, 132, 132):
            raise ValueError(
                f"{thermal_path}: expected (2, 132, 132), "
                f"got {thermal.shape}"
            )

        if optical.dtype != np.float32:
            optical = optical.astype(np.float32)

        if thermal.dtype != np.float32:
            thermal = thermal.astype(np.float32)

        optical = torch.from_numpy(optical)
        thermal = torch.from_numpy(thermal)

        sample = {
            "optical": optical,
            "thermal": thermal,
        }

        if self.return_id:
            sample["sample_id"] = sample_id

        return sample


if __name__ == "__main__":
    # Change this path on Member 2's computer.
    DATASET_ROOT = "preprocessed_data"

    dataset = SSL4EOLDataset(DATASET_ROOT, split="train")

    print("=" * 60)
    print("SSL4EO-L PYTORCH DATASET TEST")
    print("=" * 60)
    print(f"Split       : {dataset.split}")
    print(f"Samples     : {len(dataset)}")

    sample = dataset[0]

    print(f"Sample ID   : {sample['sample_id']}")
    print(f"Optical     : {tuple(sample['optical'].shape)}")
    print(f"Thermal     : {tuple(sample['thermal'].shape)}")
    print(f"Optical dtype: {sample['optical'].dtype}")
    print(f"Thermal dtype: {sample['thermal'].dtype}")

    print("\nDataset loader test PASSED.")
