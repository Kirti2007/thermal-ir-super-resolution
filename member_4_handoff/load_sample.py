"""
Member 4 Integration Data Loader

Loads one sample from the Member 4 handoff package.

Optical:
    Shape: (7, 264, 264)

Thermal LR:
    Shape: (2, 132, 132)

Thermal HR:
    Shape: (2, 264, 264)

dtype:
    float32

Value range:
    [0, 1]
"""

from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / 'sample_data'


def load_sample(sample_number=1):
    """Load one sample."""

    sample_dir = (
        SAMPLES_DIR
        / f'sample_{sample_number:03d}'
    )

    if not sample_dir.exists():
        raise FileNotFoundError(
            f'Sample not found: {sample_dir}'
        )

    optical = np.load(
        sample_dir / 'optical.npy'
    )

    thermal_lr = np.load(
        sample_dir / 'thermal_lr.npy'
    )

    thermal_hr = np.load(
        sample_dir / 'thermal_hr.npy'
    )

    return optical, thermal_lr, thermal_hr


if __name__ == '__main__':

    optical, thermal_lr, thermal_hr = load_sample(1)

    print('Sample loaded successfully!')
    print()

    print('Optical:')
    print(f'  Shape : {optical.shape}')
    print(f'  dtype : {optical.dtype}')
    print(
        f'  range : [{optical.min():.6f}, {optical.max():.6f}]'
    )

    print()

    print('Thermal LR:')
    print(f'  Shape : {thermal_lr.shape}')
    print(f'  dtype : {thermal_lr.dtype}')
    print(
        f'  range : [{thermal_lr.min():.6f}, {thermal_lr.max():.6f}]'
    )

    print()

    print('Thermal HR:')
    print(f'  Shape : {thermal_hr.shape}')
    print(f'  dtype : {thermal_hr.dtype}')
    print(
        f'  range : [{thermal_hr.min():.6f}, {thermal_hr.max():.6f}]'
    )
