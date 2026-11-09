"""Local CLI checks; run with python -m unittest discover -s tests -v."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold


PROJECT = Path(__file__).resolve().parents[1]


class MLCommandsTest(unittest.TestCase):
    @classmethod
    def command(cls, script, *args, succeeds=True):
        result = subprocess.run(
            [sys.executable, str(PROJECT / script), *map(str, args)],
            capture_output=True, text=True, timeout=120,
        )
        if succeeds and result.returncode:
            raise AssertionError(f"{script} failed:\n{result.stdout}\n{result.stderr}")
        if not succeeds and result.returncode == 0:
            raise AssertionError(f"{script} unexpectedly succeeded")
        return result

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="digits CLI tests ")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.workspace = Path(cls.temporary.name)
        cls.data = cls.workspace / "prepared data"
        cls.command("prepare.py", "--output-dir", cls.data)
        cls.manifest = json.loads((cls.data / "manifest.json").read_text())
        cls.results = cls.workspace / "fold results"
        cls.results.mkdir()
        for fold in cls.manifest["folds"]:
            cls.command(
                "train.py", "--data", cls.data / "data.npz",
                "--train-index", cls.data / fold["train_index"],
                "--test-index", cls.data / fold["test_index"],
                "--output", cls.results / f"fold-{fold['id']}.json",
            )

    def test_reproducible_stratified_splits_and_coverage(self):
        repeated = self.workspace / "repeated preparation"
        self.command("prepare.py", "--output-dir", repeated)
        self.assertEqual(self.manifest["seed"], 42)
        self.assertEqual(self.manifest["sample_count"], 1797)
        self.assertEqual(self.manifest["data_sha256"], hashlib.sha256((self.data / "data.npz").read_bytes()).hexdigest())
        self.assertEqual([f["id"] for f in self.manifest["folds"]], [0, 1, 2])
        with np.load(self.data / "data.npz") as data:
            self.assertEqual(data["X"].shape, (1797, 64))
            self.assertEqual(data["y"].shape, (1797,))
            expected = list(StratifiedKFold(3, shuffle=True, random_state=42).split(data["X"], data["y"]))
        test_indices = []
        for fold, (expected_train, expected_test) in zip(self.manifest["folds"], expected):
            train = np.load(self.data / fold["train_index"])
            test = np.load(self.data / fold["test_index"])
            np.testing.assert_array_equal(train, expected_train)
            np.testing.assert_array_equal(test, expected_test)
            self.assertEqual(np.intersect1d(train, test).size, 0)
            np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(1797))
            for name, indices in (("train", train), ("test", test)):
                np.testing.assert_array_equal(indices, np.load(repeated / fold[f"{name}_index"]))
                self.assertEqual(fold[f"{name}_count"], len(indices))
                self.assertEqual(fold[f"{name}_sha256"], hashlib.sha256((self.data / fold[f"{name}_index"]).read_bytes()).hexdigest())
            test_indices.extend(test)
        np.testing.assert_array_equal(np.sort(test_indices), np.arange(1797))

    def test_training_and_aggregation(self):
        accuracies = []
        for fold in self.manifest["folds"]:
            result = json.loads((self.results / f"fold-{fold['id']}.json").read_text())
            accuracy = result["accuracy"]
            self.assertTrue(np.isfinite(accuracy))
            self.assertGreaterEqual(accuracy, 0)
            self.assertLessEqual(accuracy, 1)
            self.assertEqual(result["train_count"], fold["train_count"])
            self.assertEqual(result["test_count"], fold["test_count"])
            self.assertEqual(accuracy, result["correct_count"] / result["test_count"])
            self.assertEqual(result["data_sha256"], self.manifest["data_sha256"])
            for name in ("train", "test"):
                self.assertEqual(result[f"{name}_sha256"], fold[f"{name}_sha256"])
            self.assertEqual(result["model"], {"scaler": "StandardScaler", "kernel": "rbf", "C": 1.0, "gamma": "scale"})
            accuracies.append(accuracy)
        output = self.workspace / "summary results.json"
        self.command("aggregate.py", "--manifest", self.data / "manifest.json", "--results-dir", self.results, "--output", output)
        summary = json.loads(output.read_text())
        self.assertEqual(summary["folds"], [{"id": i, "accuracy": value} for i, value in enumerate(accuracies)])
        self.assertAlmostEqual(summary["mean_accuracy"], float(np.mean(accuracies)))
        self.assertAlmostEqual(summary["std_accuracy"], float(np.std(accuracies, ddof=0)))

    def test_training_accepts_arbitrarily_named_explicit_paths(self):
        explicit = self.workspace / "explicit input paths"
        explicit.mkdir()
        fold = self.manifest["folds"][0]
        for source, destination in (("data.npz", "samples.npz"), (fold["train_index"], "fit rows.npy"), (fold["test_index"], "evaluation rows.npy")):
            shutil.copyfile(self.data / source, explicit / destination)
        output = explicit / "metrics.json"
        self.command("train.py", "--data", explicit / "samples.npz", "--train-index", explicit / "fit rows.npy", "--test-index", explicit / "evaluation rows.npy", "--output", output)
        self.assertEqual(json.loads(output.read_text()), json.loads((self.results / "fold-0.json").read_text()))

    def test_aggregation_rejects_missing_wrong_fold_and_inconsistent_counts(self):
        for case in ("missing", "wrong fold", "inconsistent counts"):
            with self.subTest(case=case):
                results = self.workspace / case
                shutil.copytree(self.results, results)
                target = results / "fold-1.json"
                if case == "missing":
                    target.unlink()
                elif case == "wrong fold":
                    shutil.copyfile(results / "fold-0.json", target)
                else:
                    result = json.loads(target.read_text())
                    result["train_count"] += 1
                    target.write_text(json.dumps(result))
                self.command("aggregate.py", "--manifest", self.data / "manifest.json", "--results-dir", results, "--output", self.workspace / f"{case} summary.json", succeeds=False)


if __name__ == "__main__":
    unittest.main()
