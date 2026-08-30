from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import zoom


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 264
IMAGE_WIDTH = 264

THERMAL_HEIGHT = 132
THERMAL_WIDTH = 132


# ============================================================
# 1. LOAD SAMPLE
# ============================================================

def load_sample(tif_path):
    """
    Load one SSL4EO-L all_bands.tif file.

    Input:
        all_bands.tif

    Expected output:
        [11, 264, 264]
        uint8

    The original TIFF is only READ.
    It is never modified.
    """

    tif_path = Path(tif_path)

    if not tif_path.exists():
        raise FileNotFoundError(
            f"TIFF file not found:\n{tif_path}"
        )

    with rasterio.open(tif_path) as src:
        sample = src.read()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if sample.ndim != 3:
        raise ValueError(
            f"Expected [bands, height, width], "
            f"got {sample.shape}"
        )

    if sample.shape[0] != 11:
        raise ValueError(
            f"Expected 11 bands, got {sample.shape[0]}"
        )

    if sample.shape[1:] != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    ):
        raise ValueError(
            f"Expected spatial size "
            f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}, "
            f"got {sample.shape[1:]}"
        )

    return sample


# ============================================================
# 2. SEPARATE BANDS
# ============================================================

def separate_bands(sample):
    """
    Separate optical and thermal bands.

    Dataset band layout:

        Index 0  -> B1
        Index 1  -> B2
        Index 2  -> B3
        Index 3  -> B4
        Index 4  -> B5
        Index 5  -> B6
        Index 6  -> B7
        Index 7  -> B8
        Index 8  -> B9
        Index 9  -> B10
        Index 10 -> B11

    Project:

        Optical = B2-B8
        Thermal = B10-B11

    Returns:

        optical -> [7,264,264]
        thermal -> [2,264,264]
    """

    if sample.shape[0] != 11:
        raise ValueError(
            f"Expected 11 bands, got {sample.shape[0]}"
        )

    # B2-B8
    optical = sample[1:8]

    # B10-B11
    thermal = sample[9:11]

    return optical, thermal


# ============================================================
# 3. NORMALIZE OPTICAL
# ============================================================

def normalize_optical(optical):
    """
    Normalize optical data.

    Input:
        uint8 [0,255]

    Output:
        float32 [0,1]

    Shape remains:
        [7,264,264]
    """

    optical = optical.astype(np.float32)

    optical = optical / 255.0

    return optical


# ============================================================
# 4. NORMALIZE THERMAL
# ============================================================

def normalize_thermal(thermal):
    """
    Normalize thermal data.

    Input:
        uint8 [0,255]

    Output:
        float32 [0,1]

    Shape:
        [2,264,264]
    """

    thermal = thermal.astype(np.float32)

    thermal = thermal / 255.0

    return thermal


# ============================================================
# 5. RESIZE
# ============================================================

def resize(array, target_height, target_width):
    """
    Resize a [C,H,W] array.

    Parameters
    ----------
    array : np.ndarray
        Shape [C,H,W]

    target_height : int
        Target height

    target_width : int
        Target width

    Returns
    -------
    np.ndarray
        Resized array
    """

    if array.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W], got {array.shape}"
        )

    channels, height, width = array.shape

    zoom_y = target_height / height
    zoom_x = target_width / width

    resized = zoom(
        array,
        zoom=(1, zoom_y, zoom_x),
        order=1
    )

    return resized.astype(np.float32)


# ============================================================
# 6. CREATE LOW-RESOLUTION THERMAL
# ============================================================

def create_low_res_thermal(thermal):
    """
    Convert thermal data from 264x264 to 132x132.

    Input:
        [2,264,264]

    Output:
        [2,132,132]

    This is simply spatial downsampling.

    IMPORTANT:
        The original TIFF is NOT modified.
        This creates a new processed array.
    """

    if thermal.shape != (
        2,
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    ):
        raise ValueError(
            f"Expected thermal shape "
            f"(2,{IMAGE_HEIGHT},{IMAGE_WIDTH}), "
            f"got {thermal.shape}"
        )

    thermal_132 = resize(
        thermal,
        target_height=THERMAL_HEIGHT,
        target_width=THERMAL_WIDTH
    )

    return thermal_132


# ============================================================
# 7. CREATE PATCHES
# ============================================================

def create_patches(array, patch_size):
    """
    Optional utility for creating non-overlapping patches.

    This is NOT used in the main preprocessing pipeline.

    The main processed data remains:

        Optical -> 264x264
        Thermal -> 132x132
    """

    if array.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W], got {array.shape}"
        )

    channels, height, width = array.shape

    patches = []

    for y in range(
        0,
        height - patch_size + 1,
        patch_size
    ):
        for x in range(
            0,
            width - patch_size + 1,
            patch_size
        ):

            patch = array[
                :,
                y:y + patch_size,
                x:x + patch_size
            ]

            patches.append(patch)

    return patches


# ============================================================
# 8. PREPROCESS ONE SAMPLE
# ============================================================

def preprocess_sample(tif_path):
    """
    Complete preprocessing pipeline for ONE TIFF.

    Input:
        all_bands.tif
        [11,264,264]

    Output:
        optical
            [7,264,264]

        thermal
            [2,132,132]

    No target is created.
    No thermal_hr is created.
    No modification is made to the original TIFF.
    """

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    sample = load_sample(tif_path)

    # --------------------------------------------------------
    # Separate
    # --------------------------------------------------------

    optical, thermal = separate_bands(sample)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    optical = normalize_optical(optical)

    thermal = normalize_thermal(thermal)

    # --------------------------------------------------------
    # Resize thermal
    # --------------------------------------------------------

    thermal = create_low_res_thermal(thermal)

    # --------------------------------------------------------
    # Validate final shapes
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

    return {
        "optical": optical,
        "thermal": thermal
    }


# ============================================================
# 9. SAVE PROCESSED SAMPLE
# ============================================================

def save_preprocessed_sample(
    processed,
    output_root,
    split,
    sample_id
):
    """
    Save one processed sample.

    Directory structure:

        preprocessed_data/
        └── train/
            ├── optical/
            │   └── 0000000.npy
            │
            └── thermal/
                └── 0000000.npy
    """

    output_root = Path(output_root)

    valid_splits = {
        "train",
        "val",
        "test",
        "sample_data"
    }

    if split not in valid_splits:
        raise ValueError(
            f"Invalid split: {split}"
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    optical_dir = (
        output_root /
        split /
        "optical"
    )

    thermal_dir = (
        output_root /
        split /
        "thermal"
    )

    optical_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    thermal_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    optical_path = (
        optical_dir /
        f"{sample_id}.npy"
    )

    thermal_path = (
        thermal_dir /
        f"{sample_id}.npy"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        optical_path,
        processed["optical"]
    )

    np.save(
        thermal_path,
        processed["thermal"]
    )

    return optical_path, thermal_path


# ============================================================
# 10. LOAD PROCESSED SAMPLE
# ============================================================

def load_preprocessed_sample(
    output_root,
    split,
    sample_id
):
    """
    Load one processed sample.

    Returns:

        optical -> [7,264,264]
        thermal -> [2,132,132]
    """

    output_root = Path(output_root)

    optical_path = (
        output_root /
        split /
        "optical" /
        f"{sample_id}.npy"
    )

    thermal_path = (
        output_root /
        split /
        "thermal" /
        f"{sample_id}.npy"
    )

    if not optical_path.exists():
        raise FileNotFoundError(
            f"Optical file not found:\n{optical_path}"
        )

    if not thermal_path.exists():
        raise FileNotFoundError(
            f"Thermal file not found:\n{thermal_path}"
        )

    optical = np.load(optical_path)

    thermal = np.load(thermal_path)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if optical.shape != (
        7,
        264,
        264
    ):
        raise ValueError(
            f"Invalid optical shape: "
            f"{optical.shape}"
        )

    if thermal.shape != (
        2,
        132,
        132
    ):
        raise ValueError(
            f"Invalid thermal shape: "
            f"{thermal.shape}"
        )

    return {
        "optical": optical,
        "thermal": thermal
    }