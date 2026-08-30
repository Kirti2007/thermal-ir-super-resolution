from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"D:\Projects\Analytica\ssl4eo_l_oli_tirs_toa_benchmark"
    r"\ssl4eo_l_oli_tirs_toa_benchmark"
)

OUTPUT_FILE = Path("data/dataset_index.csv")

# Official SSL4EO-L / TorchGeo split
SEED = 0


# ============================================================
# STEP 1: FIND ALL TIFF FILES
# ============================================================

print("=" * 70)
print("SSL4EO-L FULL DATASET SPLIT")
print("=" * 70)

print("\nSearching for all_bands.tif files...")

tif_files = sorted(DATASET_ROOT.rglob("all_bands.tif"))

print(f"Total TIFF files found: {len(tif_files)}")

if len(tif_files) != 25000:
    raise ValueError(
        f"Expected 25,000 TIFF files, but found {len(tif_files)}"
    )


# ============================================================
# STEP 2: REPRODUCE OFFICIAL TORCHGEO SPLIT
# ============================================================

print("\nCreating official 70/15/15 split...")

np.random.seed(SEED)

indices = np.arange(len(tif_files))

np.random.shuffle(indices)


# 70% / 15% / 15%
train_end = int(0.70 * len(indices))
val_end = train_end + int(0.15 * len(indices))

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]


print(f"Train samples:      {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")
print(f"Test samples:       {len(test_indices)}")
print(f"Total samples:      {len(indices)}")


# ============================================================
# STEP 3: CREATE RECORDS
# ============================================================

records = []


def add_records(indices, split):

    for idx in indices:

        path = tif_files[idx]

        relative_path = path.relative_to(DATASET_ROOT)

        # Example:
        # 0000000/LC08_045030_20190814/all_bands.tif

        sample_id = relative_path.parts[0]

        scene_id = relative_path.parts[1]

        records.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "split": split,
                "source_path": str(path),
                "relative_path": str(relative_path),
            }
        )


add_records(train_indices, "train")
add_records(val_indices, "val")
add_records(test_indices, "test")


# ============================================================
# STEP 4: CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# Keep a predictable ordering
split_order = {
    "train": 0,
    "val": 1,
    "test": 2,
}

df["split_order"] = df["split"].map(split_order)

df = (
    df
    .sort_values(["split_order", "sample_id"])
    .drop(columns=["split_order"])
)


# ============================================================
# STEP 5: SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# STEP 6: VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(f"\nTrain:      {len(df[df['split'] == 'train'])}")
print(f"Validation: {len(df[df['split'] == 'val'])}")
print(f"Test:       {len(df[df['split'] == 'test'])}")
print(f"TOTAL:      {len(df)}")

print(f"\nIndex saved to:")
print(OUTPUT_FILE)

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nFirst 5 records:")
print(df.head().to_string(index=False))

print("\nLast 5 records:")
print(df.tail().to_string(index=False))

print("=" * 70)