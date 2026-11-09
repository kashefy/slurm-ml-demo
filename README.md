# slurm-svm-demo

A small, local-first example for trying a Slurm workflow: three stratified
folds of scikit-learn's bundled digits dataset and a fixed RBF SVM. Each
training run fits its `StandardScaler` only on the training partition.
No data download is needed.

## Run locally

Use Python with NumPy and scikit-learn already installed. For example, set up
a development environment beforehand:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy scikit-learn
```

Then run from the repository root:

```bash
python prepare.py --output-dir work/data
for fold in 0 1 2; do
  python train.py --data work/data/digits.npz \
    --train-indices "work/data/train_${fold}.npy" \
    --test-indices "work/data/test_${fold}.npy" \
    --output "work/results/${fold}.json"
done
python aggregate.py work/results/0.json work/results/1.json work/results/2.json
```

Preparation uses seed 42. Each result contains accuracy and training/test
sample counts. Aggregation prints accuracies in the supplied order, their
mean, and population standard deviation; it fails if a result is missing.
The training script accepts explicit paths and does not depend on Slurm or
fold numbering. Output files are overwritten on rerun.

Run the local checks with:

```bash
python -m unittest discover -s tests -v
```

## Slurm layer: pending

The local checkout of
[slurm-workflow-skill](https://github.com/kashefy/slurm-workflow-skill) was
not available during development. The fold wrapper, array batch script,
human-run dependency launcher, and private configuration example have
therefore not been generated. They should be added using that skill's
guidance when a checkout is available, without making the skill a runtime
dependency. Cluster settings and the staged CPU container image path must
remain in configuration outside this repository; workload code should be
mounted read-only.

The Python workflow and local failure cases have been checked. No scheduler
mock, Apptainer/Singularity container, or cluster submission checks have
been performed; those remain pending with the Slurm layer.