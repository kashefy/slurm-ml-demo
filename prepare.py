"""Write the bundled digits dataset and three reproducible stratified splits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_digits(return_X_y=True)
    data_path = args.output_dir / "data.npz"
    np.savez(data_path, X=X, y=y)
    manifest = {
        "seed": 42,
        "sample_count": len(y),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "folds": [],
    }
    splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(splits.split(X, y)):
        fold_dir = args.output_dir / f"fold-{fold}"
        fold_dir.mkdir(exist_ok=True)
        entry = {"id": fold}
        for name, indices in (("train", train), ("test", test)):
            path = fold_dir / f"{name}.npy"
            np.save(path, indices)
            entry[f"{name}_index"] = path.relative_to(args.output_dir).as_posix()
            entry[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            entry[f"{name}_count"] = len(indices)
        manifest["folds"].append(entry)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
