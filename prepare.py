import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(description="Prepare the digits dataset and three stratified folds.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    np.savez_compressed(args.output_dir / "digits.npz", X=digits.data, y=digits.target)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(folds.split(digits.data, digits.target)):
        np.save(args.output_dir / f"train_{fold}.npy", train)
        np.save(args.output_dir / f"test_{fold}.npy", test)


if __name__ == "__main__":
    main()
