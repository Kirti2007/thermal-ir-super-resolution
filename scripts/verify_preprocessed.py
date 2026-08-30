import os
import json
import time
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"D:\Projects\Analytica"

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "preprocessed_data"
)

INDEX_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset_index.csv"
)

METADATA_FILE = os.path.join(
    OUTPUT_ROOT,
    "metadata",
    "preprocessing_info.json"
)

ERROR_LOG = os.path.join(
    OUTPUT_ROOT,
    "verification_errors.csv"
)


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_COUNTS = {
    "train": 17500,
    "val": 3750,
    "test": 3750
}

EXPECTED_TOTAL = 25000


# ============================================================
# EXPECTED ARRAY SPECIFICATIONS
# ============================================================

EXPECTED_SPECS = {
    "optical": {
        "shape": (7, 264, 264),
        "dtype": np.float32
    },

    "thermal_hr": {
        "shape": (2, 264, 264),
        "dtype": np.float32
    },

    "thermal_lr": {
        "shape": (2, 132, 132),
        "dtype": np.float32
    }
}


# ============================================================
# HELPERS
# ============================================================

def print_status(ok, message):
    if ok:
        print(f"✓ {message}")
    else:
        print(f"✗ {message}")


def get_sample_id(value):
    """
    Convert sample ID to the same 7-digit format used
    by process_dataset.py.

    Example:
        1       -> 0000001
        12345   -> 0012345
    """

    try:
        return str(int(value)).zfill(7)
    except Exception:
        return str(value).zfill(7)


# ============================================================
# CHECK PATHS
# ============================================================

def verify_paths():

    print()
    print("=" * 70)
    print("VERIFYING PATHS")
    print("=" * 70)

    paths = {
        "Output directory": OUTPUT_ROOT,
        "Dataset index": INDEX_FILE,
        "Metadata": METADATA_FILE
    }

    all_ok = True

    for name, path in paths.items():

        exists = os.path.exists(path)

        print_status(
            exists,
            f"{name}: {path}"
        )

        if not exists:
            all_ok = False

    if not all_ok:
        raise FileNotFoundError(
            "One or more required paths do not exist."
        )

    return True


# ============================================================
# VERIFY METADATA
# ============================================================

def verify_metadata():

    print()
    print("=" * 70)
    print("VERIFYING METADATA")
    print("=" * 70)

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    errors = []

    # Dataset name
    if metadata.get("dataset") != "SSL4EO-L OLI/TIRS TOA":
        errors.append(
            f"Unexpected dataset name: "
            f"{metadata.get('dataset')}"
        )

    # Total
    if metadata.get("total_samples") != EXPECTED_TOTAL:
        errors.append(
            f"Expected total_samples={EXPECTED_TOTAL}, "
            f"found {metadata.get('total_samples')}"
        )

    # Splits
    metadata_splits = metadata.get("splits", {})

    for split, expected in EXPECTED_COUNTS.items():

        actual = metadata_splits.get(split)

        if actual != expected:

            errors.append(
                f"{split}: expected {expected}, "
                f"metadata contains {actual}"
            )

    # Shapes
    metadata_shapes = metadata.get("shapes", {})

    expected_shapes = {
        "optical": [7, 264, 264],
        "thermal_hr": [2, 264, 264],
        "thermal_lr": [2, 132, 132]
    }

    for name, expected_shape in expected_shapes.items():

        actual_shape = metadata_shapes.get(name)

        if actual_shape != expected_shape:

            errors.append(
                f"{name} shape mismatch: "
                f"expected {expected_shape}, "
                f"found {actual_shape}"
            )

    # dtype
    if metadata.get("dtype") != "float32":

        errors.append(
            f"Expected dtype=float32, "
            f"found {metadata.get('dtype')}"
        )

    # Value range
    if metadata.get("value_range") != [0.0, 1.0]:

        errors.append(
            f"Expected value range [0,1], "
            f"found {metadata.get('value_range')}"
        )

    # Thermal LR
    thermal_lr_info = metadata.get(
        "thermal_lr_generation",
        {}
    )

    if thermal_lr_info.get("output") != "132x132":

        errors.append(
            "Thermal LR output should be 132x132"
        )

    if errors:

        for error in errors:
            print(f"✗ {error}")

        return False

    print_status(
        True,
        "Metadata verified."
    )

    print()
    print("Dataset      : SSL4EO-L OLI/TIRS TOA")
    print("Total samples: 25,000")
    print("dtype        : float32")
    print("value range  : [0, 1]")

    print()
    print("Expected shapes:")
    print("  Optical     : (7, 264, 264)")
    print("  Thermal HR  : (2, 264, 264)")
    print("  Thermal LR  : (2, 132, 132)")

    return True


# ============================================================
# VERIFY DATASET INDEX
# ============================================================

def verify_index():

    print()
    print("=" * 70)
    print("VERIFYING DATASET INDEX")
    print("=" * 70)

    df = pd.read_csv(INDEX_FILE)

    errors = []

    # Required columns
    required_columns = [
        "sample_id",
        "split",
        "source_path"
    ]

    for column in required_columns:

        if column not in df.columns:

            errors.append(
                f"Missing required column: {column}"
            )

    if errors:

        for error in errors:
            print(f"✗ {error}")

        return None

    print(f"Records in index: {len(df)}")

    if len(df) != EXPECTED_TOTAL:

        errors.append(
            f"Expected {EXPECTED_TOTAL} records, "
            f"found {len(df)}"
        )

    print()
    print("Split counts:")

    for split, expected in EXPECTED_COUNTS.items():

        actual = int(
            (df["split"] == split).sum()
        )

        print(
            f"  {split.upper():8s}: "
            f"{actual:6d} / {expected:6d}"
        )

        if actual != expected:

            errors.append(
                f"{split}: expected {expected}, "
                f"found {actual}"
            )

    # Duplicate sample IDs
    duplicate_count = int(
        df["sample_id"].duplicated().sum()
    )

    print()
    print(
        f"Duplicate sample IDs: {duplicate_count}"
    )

    if duplicate_count > 0:

        errors.append(
            f"Found {duplicate_count} duplicate sample IDs"
        )

    if errors:

        print()

        for error in errors:
            print(f"✗ {error}")

        return None

    print()
    print_status(
        True,
        "Dataset index verified."
    )

    return df


# ============================================================
# VERIFY DIRECTORY STRUCTURE
# ============================================================

def verify_directories():

    print()
    print("=" * 70)
    print("VERIFYING DIRECTORY STRUCTURE")
    print("=" * 70)

    all_ok = True

    for split in EXPECTED_COUNTS:

        print()
        print(split.upper())

        for data_type in [
            "optical",
            "thermal_hr",
            "thermal_lr"
        ]:

            directory = os.path.join(
                OUTPUT_ROOT,
                split,
                data_type
            )

            exists = os.path.isdir(directory)

            print_status(
                exists,
                f"{data_type:12s}: {directory}"
            )

            if not exists:
                all_ok = False

    return all_ok


# ============================================================
# COUNT FILES
# ============================================================

def count_files():

    print()
    print("=" * 70)
    print("COUNTING PREPROCESSED FILES")
    print("=" * 70)

    results = {}

    all_ok = True

    for split, expected_count in EXPECTED_COUNTS.items():

        print()
        print(split.upper())

        results[split] = {}

        for data_type in [
            "optical",
            "thermal_hr",
            "thermal_lr"
        ]:

            directory = os.path.join(
                OUTPUT_ROOT,
                split,
                data_type
            )

            files = [
                f
                for f in os.listdir(directory)
                if f.lower().endswith(".npy")
            ]

            count = len(files)

            results[split][data_type] = count

            expected = expected_count

            ok = count == expected

            print_status(
                ok,
                f"{data_type:12s}: "
                f"{count:6d} / {expected:6d}"
            )

            if not ok:
                all_ok = False

    return results, all_ok


# ============================================================
# VERIFY FILENAMES
# ============================================================

def verify_filenames(df):

    print()
    print("=" * 70)
    print("VERIFYING SAMPLE FILENAMES")
    print("=" * 70)

    errors = []

    for split in EXPECTED_COUNTS:

        split_df = df[
            df["split"] == split
        ]

        expected_ids = {
            get_sample_id(sample_id)
            for sample_id in split_df["sample_id"]
        }

        print()
        print(split.upper())

        for data_type in [
            "optical",
            "thermal_hr",
            "thermal_lr"
        ]:

            directory = os.path.join(
                OUTPUT_ROOT,
                split,
                data_type
            )

            actual_ids = {
                os.path.splitext(filename)[0]
                for filename in os.listdir(directory)
                if filename.endswith(".npy")
            }

            missing = expected_ids - actual_ids
            unexpected = actual_ids - expected_ids

            if missing:

                errors.append({
                    "split": split,
                    "type": data_type,
                    "sample_id": "",
                    "error": (
                        f"Missing files: "
                        f"{len(missing)}"
                    )
                })

                print(
                    f"✗ {data_type}: "
                    f"{len(missing)} missing"
                )

            elif unexpected:

                errors.append({
                    "split": split,
                    "type": data_type,
                    "sample_id": "",
                    "error": (
                        f"Unexpected files: "
                        f"{len(unexpected)}"
                    )
                })

                print(
                    f"✗ {data_type}: "
                    f"{len(unexpected)} unexpected"
                )

            else:

                print_status(
                    True,
                    f"{data_type}: all sample IDs match"
                )

    return errors


# ============================================================
# VERIFY SINGLE ARRAY
# ============================================================

def verify_array(
    file_path,
    expected_shape,
    expected_dtype
):

    try:

        array = np.load(
            file_path,
            allow_pickle=False
        )

        # Shape
        if array.shape != expected_shape:

            return (
                False,
                f"shape {array.shape}, "
                f"expected {expected_shape}"
            )

        # dtype
        if array.dtype != expected_dtype:

            return (
                False,
                f"dtype {array.dtype}, "
                f"expected {expected_dtype}"
            )

        # NaN / Inf
        if not np.isfinite(array).all():

            return (
                False,
                "contains NaN or Inf"
            )

        # Range
        minimum = float(array.min())
        maximum = float(array.max())

        if minimum < 0.0 or maximum > 1.0:

            return (
                False,
                f"value range "
                f"[{minimum:.6f}, {maximum:.6f}] "
                f"outside [0,1]"
            )

        return True, ""

    except Exception as e:

        return False, str(e)


# ============================================================
# VERIFY ALL ARRAYS
# ============================================================

def verify_arrays(df):

    print()
    print("=" * 70)
    print("VERIFYING ARRAY CONTENT")
    print("=" * 70)

    errors = []

    total_checked = 0

    start_time = time.time()

    for split in EXPECTED_COUNTS:

        split_df = df[
            df["split"] == split
        ]

        print()
        print(f"CHECKING {split.upper()}")

        split_start = time.time()

        split_errors = 0

        for position, (_, row) in enumerate(
            split_df.iterrows(),
            start=1
        ):

            sample_id = get_sample_id(
                row["sample_id"]
            )

            for data_type, spec in EXPECTED_SPECS.items():

                file_path = os.path.join(
                    OUTPUT_ROOT,
                    split,
                    data_type,
                    f"{sample_id}.npy"
                )

                if not os.path.exists(file_path):

                    errors.append({
                        "split": split,
                        "type": data_type,
                        "sample_id": sample_id,
                        "error": "File does not exist"
                    })

                    split_errors += 1
                    continue

                ok, error = verify_array(
                    file_path,
                    spec["shape"],
                    spec["dtype"]
                )

                if not ok:

                    errors.append({
                        "split": split,
                        "type": data_type,
                        "sample_id": sample_id,
                        "error": error
                    })

                    split_errors += 1

            total_checked += 1

            if (
                position % 500 == 0
                or position == len(split_df)
            ):

                elapsed = (
                    time.time() - split_start
                )

                print(
                    f"  [{position:6d}/"
                    f"{len(split_df):6d}] "
                    f"errors: {split_errors} | "
                    f"time: {elapsed / 60:.1f} min"
                )

        if split_errors == 0:

            print_status(
                True,
                f"{split.upper()} arrays verified."
            )

        else:

            print(
                f"✗ {split.upper()} "
                f"contains {split_errors} errors."
            )

    total_time = time.time() - start_time

    print()
    print(
        f"Samples checked: {total_checked:,}"
    )

    print(
        f"Arrays checked : "
        f"{total_checked * 3:,}"
    )

    print(
        f"Verification time: "
        f"{total_time / 60:.2f} minutes"
    )

    return errors


# ============================================================
# VERIFY SAMPLE PAIRING
# ============================================================

def verify_pairing(df):

    print()
    print("=" * 70)
    print("VERIFYING SAMPLE PAIRING")
    print("=" * 70)

    errors = []

    for split in EXPECTED_COUNTS:

        split_df = df[
            df["split"] == split
        ]

        for _, row in split_df.iterrows():

            sample_id = get_sample_id(
                row["sample_id"]
            )

            paths = []

            for data_type in [
                "optical",
                "thermal_hr",
                "thermal_lr"
            ]:

                path = os.path.join(
                    OUTPUT_ROOT,
                    split,
                    data_type,
                    f"{sample_id}.npy"
                )

                paths.append(path)

            exists = [
                os.path.exists(path)
                for path in paths
            ]

            if not all(exists):

                errors.append({
                    "split": split,
                    "type": "pairing",
                    "sample_id": sample_id,
                    "error": (
                        "Not all three modalities "
                        "exist"
                    )
                })

    if errors:

        print(
            f"✗ Found {len(errors)} "
            f"pairing errors."
        )

    else:

        print_status(
            True,
            "All samples have optical + "
            "thermal HR + thermal LR."
        )

    return errors


# ============================================================
# WRITE ERROR REPORT
# ============================================================

def save_errors(errors):

    if not errors:

        # Remove old error file if present
        if os.path.exists(ERROR_LOG):

            os.remove(ERROR_LOG)

        return

    error_df = pd.DataFrame(errors)

    error_df.to_csv(
        ERROR_LOG,
        index=False
    )

    print()
    print(
        f"Error report saved to:"
    )

    print(ERROR_LOG)


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(
    metadata_ok,
    index_ok,
    directories_ok,
    file_counts_ok,
    filename_errors,
    array_errors,
    pairing_errors
):

    total_errors = (
        len(filename_errors)
        + len(array_errors)
        + len(pairing_errors)
    )

    everything_ok = (
        metadata_ok
        and index_ok
        and directories_ok
        and file_counts_ok
        and total_errors == 0
    )

    print()
    print("=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Metadata          : "
        f"{'PASS' if metadata_ok else 'FAIL'}"
    )

    print(
        f"Dataset index     : "
        f"{'PASS' if index_ok else 'FAIL'}"
    )

    print(
        f"Directories       : "
        f"{'PASS' if directories_ok else 'FAIL'}"
    )

    print(
        f"File counts        : "
        f"{'PASS' if file_counts_ok else 'FAIL'}"
    )

    print(
        f"Filename matching  : "
        f"{'PASS' if len(filename_errors) == 0 else 'FAIL'}"
    )

    print(
        f"Array validation   : "
        f"{'PASS' if len(array_errors) == 0 else 'FAIL'}"
    )

    print(
        f"Sample pairing     : "
        f"{'PASS' if len(pairing_errors) == 0 else 'FAIL'}"
    )

    print()

    print("-" * 70)

    print(
        f"Total verification errors: "
        f"{total_errors}"
    )

    print()

    if everything_ok:

        print("✓ ALL CHECKS PASSED")
        print()
        print(
            "✓ 25,000 samples verified."
        )

        print(
            "✓ 75,000 NumPy arrays verified."
        )

        print(
            "✓ Optical data verified."
        )

        print(
            "✓ Thermal HR data verified."
        )

        print(
            "✓ Thermal LR data verified."
        )

        print(
            "✓ Shapes verified."
        )

        print(
            "✓ dtype verified."
        )

        print(
            "✓ [0,1] value range verified."
        )

        print(
            "✓ No NaN/Inf detected."
        )

        print(
            "✓ Sample pairing verified."
        )

        print()
        print("=" * 70)
        print(
            "PREPROCESSED DATASET IS READY"
        )
        print("=" * 70)

        return True

    print("✗ VERIFICATION FAILED")

    print()
    print(
        "Check the errors above and:"
    )

    print(
        f"  {ERROR_LOG}"
    )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print("=" * 70)
    print("SSL4EO-L PREPROCESSED DATASET VERIFICATION")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Output directory:")
    print(OUTPUT_ROOT)

    print()
    print(
        "This script performs READ-ONLY verification."
    )

    print(
        "It will NOT modify your dataset."
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    verify_paths()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_ok = verify_metadata()

    # --------------------------------------------------------
    # Dataset index
    # --------------------------------------------------------

    df = verify_index()

    index_ok = df is not None

    if not index_ok:

        print()
        print(
            "Cannot continue without a valid "
            "dataset index."
        )

        return 1

    # --------------------------------------------------------
    # Directory structure
    # --------------------------------------------------------

    directories_ok = verify_directories()

    if not directories_ok:

        print()
        print(
            "Required directories are missing."
        )

        return 1

    # --------------------------------------------------------
    # File counts
    # --------------------------------------------------------

    _, file_counts_ok = count_files()

    # --------------------------------------------------------
    # Filename verification
    # --------------------------------------------------------

    filename_errors = verify_filenames(df)

    # --------------------------------------------------------
    # Array verification
    # --------------------------------------------------------

    array_errors = verify_arrays(df)

    # --------------------------------------------------------
    # Pairing verification
    # --------------------------------------------------------

    pairing_errors = verify_pairing(df)

    # --------------------------------------------------------
    # Save errors
    # --------------------------------------------------------

    all_errors = (
        filename_errors
        + array_errors
        + pairing_errors
    )

    save_errors(all_errors)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    success = final_summary(
        metadata_ok,
        index_ok,
        directories_ok,
        file_counts_ok,
        filename_errors,
        array_errors,
        pairing_errors
    )

    total_time = (
        time.time() - overall_start
    )

    print()
    print(
        f"Total verification time: "
        f"{total_time / 60:.2f} minutes"
    )

    return 0 if success else 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )