import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]


def run_script(script, *args, check=True):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        check=check, capture_output=True, text=True,
    )


class WorkloadTest(unittest.TestCase):
    def test_complete_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, repeat_dir = root / "data", root / "repeat"
            run_script("prepare.py", "--output-dir", data_dir)
            run_script("prepare.py", "--output-dir", repeat_dir)
            with np.load(data_dir / "digits.npz") as data:
                X, y = data["X"], data["y"]
            with np.load(repeat_dir / "digits.npz") as data:
                np.testing.assert_array_equal(X, data["X"])
                np.testing.assert_array_equal(y, data["y"])

            results, tests = [], []
            for fold in range(3):
                train_path = data_dir / f"train_{fold}.npy"
                test_path = data_dir / f"test_{fold}.npy"
                train, test = np.load(train_path), np.load(test_path)
                np.testing.assert_array_equal(train, np.load(repeat_dir / train_path.name))
                np.testing.assert_array_equal(test, np.load(repeat_dir / test_path.name))
                np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(len(y)))
                tests.extend(test)
                for label in np.unique(y):
                    self.assertLessEqual(abs(np.mean(y[test] == label) - np.mean(y == label)), 1 / len(test))

                output = root / "results" / f"{fold}.json"
                run_script(
                    "train.py", "--data", data_dir / "digits.npz",
                    "--train-indices", train_path, "--test-indices", test_path,
                    "--output", output,
                )
                result = json.loads(output.read_text())
                expected = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
                expected.fit(X[train], y[train])
                self.assertEqual(result["accuracy"], expected.score(X[test], y[test]))
                self.assertEqual(result["train_samples"], len(train))
                self.assertEqual(result["test_samples"], len(test))
                results.append(output)

            np.testing.assert_array_equal(np.sort(tests), np.arange(len(y)))
            summary = json.loads(run_script("aggregate.py", *results).stdout)
            accuracies = [json.loads(path.read_text())["accuracy"] for path in results]
            self.assertEqual(summary["fold_accuracies"], accuracies)
            self.assertAlmostEqual(summary["mean_accuracy"], np.mean(accuracies))
            self.assertAlmostEqual(summary["std_accuracy"], np.std(accuracies))

    def test_missing_and_invalid_results_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.json"
            good.write_text('{"accuracy": 0.9}')
            missing = root / "missing.json"
            result = run_script("aggregate.py", good, good, missing, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f"expected result is missing: {missing}", result.stderr)

            bad = root / "bad.json"
            bad.write_text("{broken")
            result = run_script("aggregate.py", good, good, bad, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f"invalid result {bad}", result.stderr)

    def test_training_requires_all_paths(self):
        result = run_script("train.py", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--train-indices", result.stderr)


if __name__ == "__main__":
    unittest.main()
