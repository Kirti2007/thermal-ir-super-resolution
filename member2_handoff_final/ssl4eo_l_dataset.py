from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class SSL4EOLDataset(Dataset):
    """Load verified SSL4EO-L data for model training."""
    VALID_SPLITS = ("train", "val", "test")
    OPTICAL_SHAPE = (7, 264, 264)
    THERMAL_LR_SHAPE = (2, 132, 132)
    THERMAL_HR_SHAPE = (2, 264, 264)

    def __init__(self, root, split="train", return_id=True, validate_files=True):
        self.root = Path(root)
        self.split = split
        self.return_id = return_id
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from {self.VALID_SPLITS}.")

        self.optical_dir = self.root / split / "optical"
        self.thermal_lr_dir = self.root / split / "thermal_lr"
        self.thermal_hr_dir = self.root / split / "thermal_hr"
        for d in (self.optical_dir, self.thermal_lr_dir, self.thermal_hr_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Missing directory: {d}")

        self.sample_ids = sorted(p.stem for p in self.optical_dir.glob("*.npy"))
        if not self.sample_ids:
            raise RuntimeError(f"No .npy files found in {self.optical_dir}")
        if validate_files:
            self._validate_pairing()

    def _validate_pairing(self):
        optical = set(self.sample_ids)
        lr = {p.stem for p in self.thermal_lr_dir.glob("*.npy")}
        hr = {p.stem for p in self.thermal_hr_dir.glob("*.npy")}
        if optical != lr or optical != hr:
            raise RuntimeError(
                "Sample IDs do not match across optical, thermal_lr and thermal_hr."
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sid = self.sample_ids[index]
        optical = np.load(self.optical_dir / f"{sid}.npy", allow_pickle=False)
        thermal_lr = np.load(self.thermal_lr_dir / f"{sid}.npy", allow_pickle=False)
        thermal_hr = np.load(self.thermal_hr_dir / f"{sid}.npy", allow_pickle=False)

        self._check(optical, self.OPTICAL_SHAPE, "optical", sid)
        self._check(thermal_lr, self.THERMAL_LR_SHAPE, "thermal_lr", sid)
        self._check(thermal_hr, self.THERMAL_HR_SHAPE, "thermal_hr", sid)

        sample = {
            "optical": torch.from_numpy(optical),
            "thermal_lr": torch.from_numpy(thermal_lr),
            "thermal_hr": torch.from_numpy(thermal_hr),
        }
        if self.return_id:
            sample["sample_id"] = sid
        return sample

    @staticmethod
    def _check(a, shape, name, sid):
        if a.shape != shape:
            raise ValueError(f"{name} {sid}: expected {shape}, got {a.shape}")
        if a.dtype != np.float32:
            raise ValueError(f"{name} {sid}: expected float32, got {a.dtype}")
        if not np.isfinite(a).all():
            raise ValueError(f"{name} {sid}: contains NaN/Inf")
        if a.min() < 0 or a.max() > 1:
            raise ValueError(f"{name} {sid}: values outside [0,1]")

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    root = "preprocessed_data"
    print("=" * 70)
    print("SSL4EO-L MEMBER 2 DATASET LOADER TEST")
    print("=" * 70)
    for split in ("train", "val", "test"):
        ds = SSL4EOLDataset(root, split=split)
        s = ds[0]
        print(f"\n{split.upper()}: {len(ds)} samples")
        print("  sample_id :", s["sample_id"])
        print("  optical   :", tuple(s["optical"].shape))
        print("  thermal_lr:", tuple(s["thermal_lr"].shape))
        print("  thermal_hr:", tuple(s["thermal_hr"].shape))
    loader = DataLoader(SSL4EOLDataset(root, "train"), batch_size=4, shuffle=True, num_workers=0)
    b = next(iter(loader))
    assert tuple(b["optical"].shape) == (4, 7, 264, 264)
    assert tuple(b["thermal_lr"].shape) == (4, 2, 132, 132)
    assert tuple(b["thermal_hr"].shape) == (4, 2, 264, 264)
    print("\nBatch shapes:")
    print("  optical   :", tuple(b["optical"].shape))
    print("  thermal_lr:", tuple(b["thermal_lr"].shape))
    print("  thermal_hr:", tuple(b["thermal_hr"].shape))
    print("\n✓ Loader test PASSED")
