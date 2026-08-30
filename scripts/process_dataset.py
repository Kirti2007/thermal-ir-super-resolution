import os
import time
import json
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"D:\Projects\Analytica"

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "ssl4eo_l_oli_tirs_toa_benchmark",
    "ssl4eo_l_oli_tirs_toa_benchmark"
)

INDEX_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset_index.csv"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "preprocessed_data"
)

# ============================================================
# EXPECTED SHAPES
# ============================================================

OPTICAL_SHAPE = (7, 264, 264)
THERMAL_HR_SHAPE = (2, 264, 264)
THERMAL_LR_SHAPE = (2, 132, 132)

DTYPE = np.float32

# ============================================================
# BAND DEFINITIONS
# ============================================================

# TIFF band numbering is 1-based.
#
# Landsat 8/9:
#   Bands 1-7 = multispectral OLI bands
#   Bands 8-9 = Panchromatic + Cirrus
#   Bands 10-11 = TIRS thermal bands
#
# We use only:
#   Optical  -> 1-7
#   Thermal  -> 10-11

OPTICAL_BANDS = [1, 2, 3, 4, 5, 6, 7]
THERMAL_BANDS = [10, 11]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    for split in ["train", "val", "test"]:

        os.makedirs(
            os.path.join(OUTPUT_ROOT, split, "optical"),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(OUTPUT_ROOT, split, "thermal_lr"),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(OUTPUT_ROOT, split, "thermal_hr"),
            exist_ok=True
        )

    os.makedirs(
        os.path.join(OUTPUT_ROOT, "metadata"),
        exist_ok=True
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(array):
    """
    Convert uint8 data to float32 [0, 1].
    """

    return array.astype(np.float32) / 255.0


# ============================================================
# THERMAL LR GENERATION
# ============================================================

def create_thermal_lr(thermal_hr):
    """
    Create 2x degraded thermal input.

    Input:
        (2, 264, 264)

    Output:
        (2, 132, 132)

    Uses area-style downsampling through PIL.
    """

    channels = []

    for channel in thermal_hr:

        # Convert [0,1] -> [0,255]
        image = np.clip(
            channel * 255.0,
            0,
            255
        ).astype(np.uint8)

        pil_image = Image.fromarray(image)

        pil_image = pil_image.resize(
            (132, 132),
            Image.Resampling.BOX
        )

        lr = np.asarray(
            pil_image,
            dtype=np.float32
        ) / 255.0

        channels.append(lr)

    return np.stack(channels, axis=0)


# ============================================================
# PROCESS SINGLE TIFF
# ============================================================

def process_tiff(source_path):

    with rasterio.open(source_path) as src:

        # ----------------------------------------------------
        # Validate number of bands
        # ----------------------------------------------------

        if src.count != 11:
            raise ValueError(
                f"Expected 11 bands, found {src.count}"
            )

        # ----------------------------------------------------
        # Read optical bands 1-7
        # ----------------------------------------------------

        optical = src.read(OPTICAL_BANDS)

        # ----------------------------------------------------
        # Read thermal bands 10-11
        # ----------------------------------------------------

        thermal_hr = src.read(THERMAL_BANDS)

    # --------------------------------------------------------
    # Validate source dimensions
    # --------------------------------------------------------

    if optical.shape != OPTICAL_SHAPE:
        raise ValueError(
            f"Unexpected optical shape: {optical.shape}"
        )

    if thermal_hr.shape != THERMAL_HR_SHAPE:
        raise ValueError(
            f"Unexpected thermal shape: {thermal_hr.shape}"
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    optical = normalize(optical)
    thermal_hr = normalize(thermal_hr)

    # --------------------------------------------------------
    # Generate degraded thermal input
    # --------------------------------------------------------

    thermal_lr = create_thermal_lr(thermal_hr)

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if optical.shape != OPTICAL_SHAPE:
        raise ValueError(
            f"Optical output shape incorrect: {optical.shape}"
        )

    if thermal_hr.shape != THERMAL_HR_SHAPE:
        raise ValueError(
            f"Thermal HR output shape incorrect: {thermal_hr.shape}"
        )

    if thermal_lr.shape != THERMAL_LR_SHAPE:
        raise ValueError(
            f"Thermal LR output shape incorrect: {thermal_lr.shape}"
        )

    for name, array in [
        ("optical", optical),
        ("thermal_hr", thermal_hr),
        ("thermal_lr", thermal_lr)
    ]:

        if array.dtype != DTYPE:
            raise ValueError(
                f"{name} dtype incorrect: {array.dtype}"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} contains NaN/Inf"
            )

        if array.min() < 0 or array.max() > 1:
            raise ValueError(
                f"{name} outside [0,1]"
            )

    return optical, thermal_lr, thermal_hr


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(df, split):

    print()
    print("=" * 70)
    print(f"PROCESSING: {split.upper()}")
    print("=" * 70)

    split_df = df[df["split"] == split]

    total = len(split_df)

    print(f"Samples in split: {total}")

    optical_dir = os.path.join(
        OUTPUT_ROOT,
        split,
        "optical"
    )

    thermal_lr_dir = os.path.join(
        OUTPUT_ROOT,
        split,
        "thermal_lr"
    )

    thermal_hr_dir = os.path.join(
        OUTPUT_ROOT,
        split,
        "thermal_hr"
    )

    new_count = 0
    skipped_count = 0
    failed_count = 0

    errors = []

    start_time = time.time()

    for position, (_, row) in enumerate(
        split_df.iterrows(),
        start=1
    ):

        sample_id = str(row["sample_id"]).zfill(7)
        source_path = row["source_path"]

        optical_file = os.path.join(
            optical_dir,
            f"{sample_id}.npy"
        )

        thermal_lr_file = os.path.join(
            thermal_lr_dir,
            f"{sample_id}.npy"
        )

        thermal_hr_file = os.path.join(
            thermal_hr_dir,
            f"{sample_id}.npy"
        )

        # ----------------------------------------------------
        # Skip only when ALL three files already exist
        # ----------------------------------------------------

        if (
            os.path.exists(optical_file)
            and
            os.path.exists(thermal_lr_file)
            and
            os.path.exists(thermal_hr_file)
        ):

            skipped_count += 1
            continue

        try:

            # ------------------------------------------------
            # Process TIFF
            # ------------------------------------------------

            optical, thermal_lr, thermal_hr = process_tiff(
                source_path
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            np.save(
                optical_file,
                optical
            )

            np.save(
                thermal_lr_file,
                thermal_lr
            )

            np.save(
                thermal_hr_file,
                thermal_hr
            )

            new_count += 1

        except Exception as e:

            failed_count += 1

            errors.append({
                "sample_id": sample_id,
                "split": split,
                "source_path": source_path,
                "error": str(e)
            })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if position % 100 == 0 or position == total:

            elapsed = time.time() - start_time

            rate = position / elapsed if elapsed > 0 else 0

            remaining = (
                total - position
            ) / rate if rate > 0 else 0

            print(
                f"[{position:6d}/{total}] "
                f"{position / total * 100:6.2f}% | "
                f"new: {new_count} | "
                f"skip: {skipped_count} | "
                f"failed: {failed_count} | "
                f"ETA: {remaining / 60:.1f} min"
            )

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print()
    print(f"{split.upper()} COMPLETE")
    print(f"  Total    : {total}")
    print(f"  New      : {new_count}")
    print(f"  Skipped  : {skipped_count}")
    print(f"  Failed   : {failed_count}")
    print(f"  Time     : {elapsed / 60:.2f} minutes")

    return {
        "total": total,
        "new": new_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "errors": errors
    }


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(df, results):

    metadata = {

        "dataset": "SSL4EO-L OLI/TIRS TOA",

        "source": (
            "torchgeo/ssl4eo_l_benchmark"
        ),

        "total_samples": int(len(df)),

        "splits": {
            "train": int(
                (df["split"] == "train").sum()
            ),
            "val": int(
                (df["split"] == "val").sum()
            ),
            "test": int(
                (df["split"] == "test").sum()
            )
        },

        "bands": {

            "optical": [
                "Band 1",
                "Band 2",
                "Band 3",
                "Band 4",
                "Band 5",
                "Band 6",
                "Band 7"
            ],

            "ignored": [
                "Band 8 - Panchromatic",
                "Band 9 - Cirrus"
            ],

            "thermal": [
                "Band 10 - TIRS",
                "Band 11 - TIRS"
            ]
        },

        "shapes": {

            "optical": list(
                OPTICAL_SHAPE
            ),

            "thermal_hr": list(
                THERMAL_HR_SHAPE
            ),

            "thermal_lr": list(
                THERMAL_LR_SHAPE
            )
        },

        "dtype": "float32",

        "value_range": [
            0.0,
            1.0
        ],

        "thermal_lr_generation": {
            "method": "2x spatial downsampling",
            "input": "264x264",
            "output": "132x132",
            "resampling": "BOX"
        },

        "results": results
    }

    metadata_file = os.path.join(
        OUTPUT_ROOT,
        "metadata",
        "preprocessing_info.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    return metadata_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SSL4EO-L FINAL DATASET PREPROCESSING")
    print("=" * 70)

    print()
    print("Original dataset:")
    print(DATASET_ROOT)

    print()
    print("Dataset index:")
    print(INDEX_FILE)

    print()
    print("Output directory:")
    print(OUTPUT_ROOT)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_ROOT}"
        )

    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            f"Dataset index not found:\n{INDEX_FILE}"
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Load index
    # --------------------------------------------------------

    print()
    print("Loading dataset index...")

    df = pd.read_csv(INDEX_FILE)

    print(
        f"Loaded {len(df)} records."
    )

    # --------------------------------------------------------
    # Verify split
    # --------------------------------------------------------

    expected = {
        "train": 17500,
        "val": 3750,
        "test": 3750
    }

    print()
    print("=" * 70)
    print("VERIFYING DATASET INDEX")
    print("=" * 70)

    for split, expected_count in expected.items():

        actual_count = int(
            (df["split"] == split).sum()
        )

        print(
            f"{split.upper():12s}: "
            f"{actual_count}"
        )

        if actual_count != expected_count:

            raise ValueError(
                f"{split} expected "
                f"{expected_count}, "
                f"found {actual_count}"
            )

    print()
    print("✓ Dataset index verified.")

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    overall_start = time.time()

    results = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        results[split] = process_split(
            df,
            split
        )

    # --------------------------------------------------------
    # Save errors
    # --------------------------------------------------------

    all_errors = []

    for split_result in results.values():
        all_errors.extend(
            split_result["errors"]
        )

    error_file = os.path.join(
        OUTPUT_ROOT,
        "processing_errors.csv"
    )

    if all_errors:

        pd.DataFrame(
            all_errors
        ).to_csv(
            error_file,
            index=False
        )

    else:

        pd.DataFrame(
            columns=[
                "sample_id",
                "split",
                "source_path",
                "error"
            ]
        ).to_csv(
            error_file,
            index=False
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_file = save_metadata(
        df,
        results
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_time = (
        time.time() - overall_start
    )

    total_new = sum(
        result["new"]
        for result in results.values()
    )

    total_skipped = sum(
        result["skipped"]
        for result in results.values()
    )

    total_failed = sum(
        result["failed"]
        for result in results.values()
    )

    print()
    print("=" * 70)
    print("FULL DATASET PROCESSING COMPLETE")
    print("=" * 70)

    print()
    print("TRAIN")
    print(
        f"  Total: {results['train']['total']}"
    )
    print(
        f"  New:   {results['train']['new']}"
    )
    print(
        f"  Skip:  {results['train']['skipped']}"
    )
    print(
        f"  Fail:  {results['train']['failed']}"
    )

    print()
    print("VALIDATION")
    print(
        f"  Total: {results['val']['total']}"
    )
    print(
        f"  New:   {results['val']['new']}"
    )
    print(
        f"  Skip:  {results['val']['skipped']}"
    )
    print(
        f"  Fail:  {results['val']['failed']}"
    )

    print()
    print("TEST")
    print(
        f"  Total: {results['test']['total']}"
    )
    print(
        f"  New:   {results['test']['new']}"
    )
    print(
        f"  Skip:  {results['test']['skipped']}"
    )
    print(
        f"  Fail:  {results['test']['failed']}"
    )

    print()
    print("-" * 70)

    print(
        f"TOTAL SAMPLES : {len(df)}"
    )

    print(
        f"NEW PROCESSED : {total_new}"
    )

    print(
        f"SKIPPED       : {total_skipped}"
    )

    print(
        f"FAILED        : {total_failed}"
    )

    print(
        f"TOTAL TIME    : {total_time / 60:.2f} minutes"
    )

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    print()
    print("Metadata:")
    print(metadata_file)

    print()
    print("Error log:")
    print(error_file)

    print()

    if total_failed == 0:

        print(
            "✓ ALL 25,000 SAMPLES PROCESSED SUCCESSFULLY"
        )

    else:

        print(
            f"⚠ PROCESSING FINISHED WITH "
            f"{total_failed} FAILURES"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()