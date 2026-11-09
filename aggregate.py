import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Summarize three fold results in the given order.")
    parser.add_argument("results", type=Path, nargs=3)
    args = parser.parse_args()

    accuracies = []
    for path in args.results:
        try:
            with path.open() as result_file:
                accuracies.append(float(json.load(result_file)["accuracy"]))
        except FileNotFoundError:
            parser.error(f"expected result is missing: {path}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            parser.error(f"invalid result {path}: {error}")
    print(json.dumps({
        "fold_accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }, indent=2))


if __name__ == "__main__":
    main()
