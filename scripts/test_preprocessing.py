from pathlib import Path
import sys
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT
# ============================================================

from preprocessing.preprocess import (
    preprocess_sample,
    save_preprocessed_sample,
    load_preprocessed_sample,
)


# ============================================================
# PATHS
# ============================================================

INPUT_TIF = (
    PROJECT_ROOT
    / "ssl4eo_l_oli_tirs_toa_benchmark"
    / "ssl4eo_l_oli_tirs_toa_benchmark"
    / "0000000"
    / "LC08_045030_20190814"
    / "all_bands.tif"
)

OUTPUT_ROOT = PROJECT_ROOT / "preprocessed_data"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SSL4EO-L PREPROCESSING TEST")
    print("=" * 70)

    print("\nOriginal TIFF:")
    print(INPUT_TIF)

    print("\nOutput directory:")
    print(OUTPUT_ROOT)

    # --------------------------------------------------------
    # 1. Check original file
    # --------------------------------------------------------

    if not INPUT_TIF.exists():
        raise FileNotFoundError(
            f"Original TIFF not found:\n{INPUT_TIF}"
        )

    print("\n[1/5] Original TIFF found.")

    # --------------------------------------------------------
    # 2. Process
    # --------------------------------------------------------

    print("\n[2/5] Processing sample...")

    processed = preprocess_sample(INPUT_TIF)

    optical = processed["optical"]
    thermal = processed["thermal"]

    print("      Processing complete.")

    # --------------------------------------------------------
    # 3. Inspect
    # --------------------------------------------------------

    print("\n[3/5] Processed data:")

    print("\nOptical")
    print(f"  Shape : {optical.shape}")
    print(f"  Dtype : {optical.dtype}")
    print(f"  Min   : {optical.min()}")
    print(f"  Max   : {optical.max()}")

    print("\nThermal")
    print(f"  Shape : {thermal.shape}")
    print(f"  Dtype : {thermal.dtype}")
    print(f"  Min   : {thermal.min()}")
    print(f"  Max   : {thermal.max()}")

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert optical.shape == (
        7,
        264,
        264
    )

    assert thermal.shape == (
        2,
        132,
        132
    )

    assert optical.dtype == np.float32
    assert thermal.dtype == np.float32

    assert optical.min() >= 0
    assert optical.max() <= 1

    assert thermal.min() >= 0
    assert thermal.max() <= 1

    print("\n      ✓ Shapes, dtype and ranges are correct.")

    # --------------------------------------------------------
    # 4. Save
    # --------------------------------------------------------

    print("\n[4/5] Saving processed sample...")

    optical_path, thermal_path = save_preprocessed_sample(
        processed=processed,
        output_root=OUTPUT_ROOT,
        split="sample_data",
        sample_id="0000000"
    )

    print("\n      Optical:")
    print(f"      {optical_path}")

    print("\n      Thermal:")
    print(f"      {thermal_path}")

    # --------------------------------------------------------
    # 5. Reload
    # --------------------------------------------------------

    print("\n[5/5] Reloading saved data...")

    reloaded = load_preprocessed_sample(
        output_root=OUTPUT_ROOT,
        split="sample_data",
        sample_id="0000000"
    )

    assert reloaded["optical"].shape == (
        7,
        264,
        264
    )

    assert reloaded["thermal"].shape == (
        2,
        132,
        132
    )

    print("      ✓ Reload successful.")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST PASSED")
    print("=" * 70)

    print("\nFinal processed dataset:")

    print(
        "  Optical : "
        f"{reloaded['optical'].shape}"
    )

    print(
        "  Thermal : "
        f"{reloaded['thermal'].shape}"
    )

    print("\nOriginal TIFF was NOT modified.")

    print("\nProcessed files:")
    print(OUTPUT_ROOT / "sample_data")


if __name__ == "__main__":
    main()