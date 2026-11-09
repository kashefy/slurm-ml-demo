"""Validate all three fold results and summarize their accuracies."""

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        folds = manifest["folds"]
        if len(folds) != 3 or [fold["id"] for fold in folds] != [0, 1, 2]:
            raise ValueError("manifest must describe folds 0, 1 and 2 in order")
        expected = {f"fold-{i}.json" for i in range(3)}
        actual = {path.name for path in args.results_dir.glob("*.json")}
        if actual != expected:
            raise ValueError(
                f"result files differ: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        model = {"scaler": "StandardScaler", "kernel": "rbf", "C": 1.0, "gamma": "scale"}
        accuracies = []
        for fold in folds:
            path = args.results_dir / f"fold-{fold['id']}.json"
            result = json.loads(path.read_text())
            for key in ("train_sha256", "test_sha256", "train_count", "test_count"):
                if result[key] != fold[key]:
                    raise ValueError(f"{path.name}: inconsistent {key}")
            if result["data_sha256"] != manifest["data_sha256"] or result["model"] != model:
                raise ValueError(f"{path.name}: inconsistent dataset or model")
            train, test, correct = (result[key] for key in
                                    ("train_count", "test_count", "correct_count"))
            if (any(type(value) is not int for value in (train, test, correct))
                    or train <= 0 or test <= 0 or not 0 <= correct <= test
                    or train + test != manifest["sample_count"]):
                raise ValueError(f"{path.name}: invalid counts")
            accuracy = result["accuracy"]
            if (type(accuracy) not in (int, float) or not math.isfinite(accuracy)
                    or not math.isclose(accuracy, correct / test, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"{path.name}: inconsistent accuracy")
            accuracies.append(accuracy)
        if sum(fold["test_count"] for fold in folds) != manifest["sample_count"]:
            raise ValueError("manifest test counts do not cover the dataset")
        summary = {
            "folds": [{"id": fold["id"], "accuracy": accuracy}
                      for fold, accuracy in zip(folds, accuracies)],
            "mean_accuracy": statistics.mean(accuracies),
            "std_accuracy": statistics.pstdev(accuracies),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"aggregate: {error}\n")


if __name__ == "__main__":
    main()
