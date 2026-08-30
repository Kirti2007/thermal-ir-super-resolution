from pathlib import Path

ROOT = Path("preprocessed_data")
EXPECTED = {"train": 17500, "val": 3750, "test": 3750}

if not ROOT.is_dir():
    raise FileNotFoundError(f"Could not find {ROOT.resolve()}")

print("SSL4EO-L MEMBER 2 HANDOFF CHECK")
print("=" * 60)
for split, expected in EXPECTED.items():
    ids = {}
    print(f"\n{split.upper()}")
    for kind in ("optical", "thermal_lr", "thermal_hr"):
        d = ROOT / split / kind
        if not d.is_dir():
            raise FileNotFoundError(d)
        ids[kind] = {p.stem for p in d.glob("*.npy")}
        print(f"{kind:12s}: {len(ids[kind]):5d} / {expected:5d}")
        if len(ids[kind]) != expected:
            raise RuntimeError(f"{split}/{kind}: expected {expected}, found {len(ids[kind])}")
    if not (ids["optical"] == ids["thermal_lr"] == ids["thermal_hr"]):
        raise RuntimeError(f"{split}: sample IDs do not match")
print("\n✓ Directory counts and sample pairing PASSED")
