"""
Visual sanity-check for the SSL4EO-L preprocessed dataset.

Usage from the project root:

    python visualize_sample.py

Optional:

    python visualize_sample.py 0000000 train

Expected data:
    preprocessed_data/train/optical/0000000.npy
    preprocessed_data/train/thermal/0000000.npy

The script creates:
    visualization_output/sample_0000000_train.png
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "preprocessed_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visualization_output"


def load_sample(data_root, split, sample_id):
    optical_path = (
        Path(data_root) / split / "optical" / f"{sample_id}.npy"
    )
    thermal_path = (
        Path(data_root) / split / "thermal" / f"{sample_id}.npy"
    )

    if not optical_path.exists():
        raise FileNotFoundError(f"Optical file not found:\n{optical_path}")

    if not thermal_path.exists():
        raise FileNotFoundError(f"Thermal file not found:\n{thermal_path}")

    optical = np.load(optical_path)
    thermal = np.load(thermal_path)

    if optical.shape != (7, 264, 264):
        raise ValueError(
            f"Unexpected optical shape: {optical.shape}; "
            "expected (7, 264, 264)"
        )

    if thermal.shape != (2, 132, 132):
        raise ValueError(
            f"Unexpected thermal shape: {thermal.shape}; "
            "expected (2, 132, 132)"
        )

    return optical, thermal


def make_rgb(optical):
    """
    Landsat optical RGB:
        B4 = Red
        B3 = Green
        B2 = Blue

    optical channel order:
        [B2, B3, B4, B5, B6, B7, B8]
    """
    rgb = np.stack(
        [optical[2], optical[1], optical[0]],
        axis=-1
    )

    # Percentile stretch makes the visualization easier to inspect
    # while leaving the stored .npy data untouched.
    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    if high > low:
        rgb = (rgb - low) / (high - low)

    return np.clip(rgb, 0, 1)


def save_visualization(optical, thermal, sample_id, split, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = make_rgb(optical)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title(
        f"Optical RGB\nB4-B3-B2 | {optical.shape}"
    )
    axes[0].axis("off")

    im1 = axes[1].imshow(thermal[0])
    axes[1].set_title(
        f"Thermal B10\n{thermal[0].shape}"
    )
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(thermal[1])
    axes[2].set_title(
        f"Thermal B11\n{thermal[1].shape}"
    )
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"SSL4EO-L Preprocessed Sample: {sample_id} ({split})",
        fontsize=14
    )

    fig.tight_layout()

    output_path = (
        output_dir / f"sample_{sample_id}_{split}.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    sample_id = sys.argv[1] if len(sys.argv) > 1 else "0000000"
    split = sys.argv[2] if len(sys.argv) > 2 else "train"

    print("=" * 70)
    print("SSL4EO-L SAMPLE VISUALIZATION")
    print("=" * 70)

    print(f"Data root : {DEFAULT_DATA_ROOT}")
    print(f"Split     : {split}")
    print(f"Sample ID : {sample_id}")

    optical, thermal = load_sample(
        DEFAULT_DATA_ROOT,
        split,
        sample_id
    )

    print("\nLoaded successfully:")
    print(f"  Optical : {optical.shape} | {optical.dtype}")
    print(f"  Thermal : {thermal.shape} | {thermal.dtype}")

    print("\nValue ranges:")
    print(
        f"  Optical : {optical.min():.6f} → "
        f"{optical.max():.6f}"
    )
    print(
        f"  Thermal : {thermal.min():.6f} → "
        f"{thermal.max():.6f}"
    )

    output_path = save_visualization(
        optical,
        thermal,
        sample_id,
        split,
        DEFAULT_OUTPUT_DIR
    )

    print("\n" + "=" * 70)
    print("VISUALIZATION CREATED")
    print("=" * 70)
    print(f"Output:\n{output_path}")
    print("\n✓ Original .npy files were not modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
