"""Train a fixed scaled RBF SVM using explicit dataset and index files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--test-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        with np.load(args.data, allow_pickle=False) as data:
            X, y = data["X"], data["y"]
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("data must contain aligned X matrix and y vector")
        train = np.load(args.train_index, allow_pickle=False)
        test = np.load(args.test_index, allow_pickle=False)
        for name, indices in (("train", train), ("test", test)):
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or len(indices) == 0 or np.any(indices < 0)
                    or np.any(indices >= len(y))
                    or len(np.unique(indices)) != len(indices)):
                raise ValueError(f"invalid {name} indices")
        if np.intersect1d(train, test).size or len(train) + len(test) != len(y):
            raise ValueError("train and test must partition the dataset")
        model_parameters = {
            "scaler": "StandardScaler", "kernel": "rbf", "C": 1.0, "gamma": "scale"
        }
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
        model.fit(X[train], y[train])
        correct = int(np.count_nonzero(model.predict(X[test]) == y[test]))
        result = {
            "accuracy": correct / len(test),
            "correct_count": correct,
            "train_count": len(train),
            "test_count": len(test),
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "train_sha256": hashlib.sha256(args.train_index.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(args.test_index.read_bytes()).hexdigest(),
            "model": model_parameters,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"train: {error}\n")


if __name__ == "__main__":
    main()
