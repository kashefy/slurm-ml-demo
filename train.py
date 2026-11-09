import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an RBF SVM.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--test-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.data, allow_pickle=False) as data:
        X, y = data["X"], data["y"]
    train = np.load(args.train_indices, allow_pickle=False)
    test = np.load(args.test_indices, allow_pickle=False)
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    model.fit(X[train], y[train])
    result = {
        "accuracy": float(model.score(X[test], y[test])),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
