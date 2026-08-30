from pathlib import Path
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"
HANDOFF_DIR = PROJECT_ROOT / "member_4_handoff"
SAMPLES_DIR = HANDOFF_DIR / "sample_data"

METADATA_FILE = (
    PREPROCESSED_DIR
    / "metadata"
    / "preprocessing_info.json"
)

NUM_SAMPLES = 5


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COPY METADATA
# ============================================================

def copy_metadata():

    if not METADATA_FILE.exists():
        print("ERROR: Metadata file not found:")
        print(METADATA_FILE)
        return False

    destination = HANDOFF_DIR / "preprocessing_info.json"

    shutil.copy2(
        METADATA_FILE,
        destination
    )

    print("✓ Copied preprocessing_info.json")

    return True


# ============================================================
# FIND SAMPLE IDS
# ============================================================

def find_sample_ids():

    optical_dir = (
        PREPROCESSED_DIR
        / "train"
        / "optical"
    )

    if not optical_dir.exists():
        raise FileNotFoundError(
            f"Optical directory not found:\n{optical_dir}"
        )

    files = sorted(
        optical_dir.glob("*.npy")
    )

    if len(files) < NUM_SAMPLES:
        raise RuntimeError(
            f"Only {len(files)} samples found. "
            f"Need at least {NUM_SAMPLES}."
        )

    return [
        file.stem
        for file in files[:NUM_SAMPLES]
    ]


# ============================================================
# COPY SAMPLE DATA
# ============================================================

def copy_samples(sample_ids):

    for index, sample_id in enumerate(
        sample_ids,
        start=1
    ):

        sample_folder = (
            SAMPLES_DIR
            / f"sample_{index:03d}"
        )

        sample_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print(f"Sample {index:03d}")
        print(f"ID: {sample_id}")

        source_files = {
            "optical.npy": (
                PREPROCESSED_DIR
                / "train"
                / "optical"
                / f"{sample_id}.npy"
            ),

            "thermal_lr.npy": (
                PREPROCESSED_DIR
                / "train"
                / "thermal_lr"
                / f"{sample_id}.npy"
            ),

            "thermal_hr.npy": (
                PREPROCESSED_DIR
                / "train"
                / "thermal_hr"
                / f"{sample_id}.npy"
            ),
        }

        for filename, source in source_files.items():

            if not source.exists():
                print(f"  ✗ Missing: {source}")
                continue

            destination = (
                sample_folder
                / filename
            )

            shutil.copy2(
                source,
                destination
            )

            print(f"  ✓ {filename}")


# ============================================================
# CREATE LOAD_SAMPLE.PY
# ============================================================

def create_loader():

    lines = [
        '"""',
        "Member 4 Integration Data Loader",
        "",
        "Loads one sample from the Member 4 handoff package.",
        "",
        "Optical:",
        "    Shape: (7, 264, 264)",
        "",
        "Thermal LR:",
        "    Shape: (2, 132, 132)",
        "",
        "Thermal HR:",
        "    Shape: (2, 264, 264)",
        "",
        "dtype:",
        "    float32",
        "",
        "Value range:",
        "    [0, 1]",
        '"""',
        "",
        "from pathlib import Path",
        "import numpy as np",
        "",
        "",
        "BASE_DIR = Path(__file__).resolve().parent",
        "SAMPLES_DIR = BASE_DIR / 'sample_data'",
        "",
        "",
        "def load_sample(sample_number=1):",
        '    """Load one sample."""',
        "",
        "    sample_dir = (",
        "        SAMPLES_DIR",
        "        / f'sample_{sample_number:03d}'",
        "    )",
        "",
        "    if not sample_dir.exists():",
        "        raise FileNotFoundError(",
        "            f'Sample not found: {sample_dir}'",
        "        )",
        "",
        "    optical = np.load(",
        "        sample_dir / 'optical.npy'",
        "    )",
        "",
        "    thermal_lr = np.load(",
        "        sample_dir / 'thermal_lr.npy'",
        "    )",
        "",
        "    thermal_hr = np.load(",
        "        sample_dir / 'thermal_hr.npy'",
        "    )",
        "",
        "    return optical, thermal_lr, thermal_hr",
        "",
        "",
        "if __name__ == '__main__':",
        "",
        "    optical, thermal_lr, thermal_hr = load_sample(1)",
        "",
        "    print('Sample loaded successfully!')",
        "    print()",
        "",
        "    print('Optical:')",
        "    print(f'  Shape : {optical.shape}')",
        "    print(f'  dtype : {optical.dtype}')",
        "    print(",
        "        f'  range : [{optical.min():.6f}, "
        "{optical.max():.6f}]'",
        "    )",
        "",
        "    print()",
        "",
        "    print('Thermal LR:')",
        "    print(f'  Shape : {thermal_lr.shape}')",
        "    print(f'  dtype : {thermal_lr.dtype}')",
        "    print(",
        "        f'  range : [{thermal_lr.min():.6f}, "
        "{thermal_lr.max():.6f}]'",
        "    )",
        "",
        "    print()",
        "",
        "    print('Thermal HR:')",
        "    print(f'  Shape : {thermal_hr.shape}')",
        "    print(f'  dtype : {thermal_hr.dtype}')",
        "    print(",
        "        f'  range : [{thermal_hr.min():.6f}, "
        "{thermal_hr.max():.6f}]'",
        "    )",
    ]

    output_file = HANDOFF_DIR / "load_sample.py"

    output_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print()
    print("✓ Created load_sample.py")


# ============================================================
# CREATE README
# ============================================================

def create_readme(sample_ids):

    lines = [
        "# Member 4 — Integration Handoff",
        "",
        "This folder contains a small, ready-to-use subset of the",
        "preprocessed SSL4EO-L OLI/TIRS TOA dataset for Streamlit,",
        "visualization, and final application integration.",
        "",
        "## Dataset",
        "",
        "Dataset: SSL4EO-L OLI/TIRS TOA",
        "",
        "Complete dataset: 25,000 samples",
        "",
        f"Samples included in this handoff: {len(sample_ids)}",
        "",
        "## Data Format",
        "",
        "Each sample contains three NumPy arrays.",
        "",
        "### Optical",
        "",
        "Shape: `(7, 264, 264)`",
        "",
        "dtype: `float32`",
        "",
        "Range: `[0, 1]`",
        "",
        "### Thermal LR",
        "",
        "Shape: `(2, 132, 132)`",
        "",
        "dtype: `float32`",
        "",
        "Range: `[0, 1]`",
        "",
        "This is the low-resolution thermal input.",
        "",
        "### Thermal HR",
        "",
        "Shape: `(2, 264, 264)`",
        "",
        "dtype: `float32`",
        "",
        "Range: `[0, 1]`",
        "",
        "This is the high-resolution thermal reference/target.",
        "",
        "## Loading a Sample",
        "",
        "```python",
        "from load_sample import load_sample",
        "",
        "optical, thermal_lr, thermal_hr = load_sample(1)",
        "```",
        "",
        "Expected shapes:",
        "",
        "```text",
        "Optical     : (7, 264, 264)",
        "Thermal LR  : (2, 132, 132)",
        "Thermal HR  : (2, 264, 264)",
        "```",
        "",
        "## Sample IDs",
        "",
    ]

    for index, sample_id in enumerate(sample_ids, start=1):
        lines.append(
            f"- sample_{index:03d}: `{sample_id}`"
        )

    lines.extend(
        [
            "",
            "## Preprocessing",
            "",
            "The arrays are already preprocessed.",
            "",
            "```text",
            "dtype       : float32",
            "value range : [0, 1]",
            "NaN / Inf   : none",
            "```",
            "",
            "No original-dataset preprocessing is required for",
            "these samples.",
            "",
            "## Streamlit Integration",
            "",
            "Member 4 can use these samples to:",
            "",
            "1. Load an optical sample.",
            "2. Load the low-resolution thermal image.",
            "3. Display the input images.",
            "4. Pass the input to Member 2's ML model.",
            "5. Receive the super-resolution output.",
            "6. Display the predicted high-resolution thermal image.",
            "7. Compare the prediction with the thermal HR reference.",
            "",
            "## Folder Structure",
            "",
            "```text",
            "member_4_handoff/",
            "├── README.md",
            "├── preprocessing_info.json",
            "├── load_sample.py",
            "└── sample_data/",
            "    ├── sample_001/",
            "    │   ├── optical.npy",
            "    │   ├── thermal_lr.npy",
            "    │   └── thermal_hr.npy",
            "    ├── sample_002/",
            "    ├── sample_003/",
            "    ├── sample_004/",
            "    └── sample_005/",
            "```",
            "",
            "## Important",
            "",
            "This handoff contains only a small demonstration subset.",
            "",
            "The complete 25,000-sample dataset is NOT included here.",
            "",
            "The original dataset and full preprocessed dataset remain",
            "outside this handoff package.",
            "",
            "## Metadata",
            "",
            "See `preprocessing_info.json` for preprocessing information.",
        ]
    )

    output_file = HANDOFF_DIR / "README.md"

    output_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("✓ Created README.md")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATING MEMBER 4 HANDOFF")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Source:")
    print(PREPROCESSED_DIR)

    print()
    print("Destination:")
    print(HANDOFF_DIR)

    print()
    print("-" * 70)

    create_directories()

    if not copy_metadata():
        return

    sample_ids = find_sample_ids()

    print()
    print(f"Selected {len(sample_ids)} samples:")

    for sample_id in sample_ids:
        print(f"  {sample_id}")

    copy_samples(sample_ids)

    create_loader()

    create_readme(sample_ids)

    print()
    print("=" * 70)
    print("MEMBER 4 HANDOFF CREATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Location:")
    print(HANDOFF_DIR)

    print()
    print("Contents:")
    print("  ✓ README.md")
    print("  ✓ preprocessing_info.json")
    print("  ✓ load_sample.py")
    print(f"  ✓ {len(sample_ids)} sample folders")

    print()
    print("The original dataset was NOT modified.")
    print("The full preprocessed dataset was NOT copied.")
    print()


if __name__ == "__main__":
    main()