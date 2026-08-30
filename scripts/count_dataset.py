from pathlib import Path

DATASET_ROOT = Path(
    r"D:\Projects\Analytica\ssl4eo_l_oli_tirs_toa_benchmark"
    r"\ssl4eo_l_oli_tirs_toa_benchmark"
)

tif_files = list(DATASET_ROOT.rglob("all_bands.tif"))

print("=" * 60)
print("SSL4EO-L DATASET STRUCTURE")
print("=" * 60)

print(f"Dataset root: {DATASET_ROOT}")
print(f"Number of all_bands.tif files: {len(tif_files)}")

print("\nFirst 10 files:")

for path in tif_files[:10]:
    print(path)

print("\nLast 5 files:")

for path in tif_files[-5:]:
    print(path)

print("=" * 60)