# Slurm ML demo

Bundled sklearn digits → three stratified folds → scaled RBF SVM → accuracy
summary. Python trains from explicit index files; Bash knows the folds; Slurm
schedules them. No dataset download, tuning or tracking service.
See [PLAN.md](PLAN.md) for the implementation checklist and remaining checks.

## Local ML pipeline

Use an existing environment or install requirements on a connected development
machine. Compute-node jobs do not install anything.

```bash
cd /path/to/slurm-ml-demo
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python prepare.py --output-dir artifacts/data
for fold in 0 1 2; do
  python train.py --data artifacts/data/data.npz \
    --train-index "artifacts/data/fold-$fold/train.npy" \
    --test-index "artifacts/data/fold-$fold/test.npy" \
    --output "artifacts/results/fold-$fold.json"
done
python aggregate.py --manifest artifacts/data/manifest.json \
  --results-dir artifacts/results --output artifacts/summary.json
cat artifacts/summary.json
```

Use fresh outputs for new runs. StratifiedKFold with seed 42 gives 1,198 train
and 599 test examples per fold. The Pipeline fits StandardScaler only on train
data, then SVC(kernel="rbf", C=1.0, gamma="scale"). Results contain accuracy,
counts, dataset/index hashes and model settings. Aggregation requires three
matching results and reports mean and population standard deviation (ddof=0).

## Off-the-shelf HPC container

The publisher's [scipy-notebook image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
includes scikit-learn. Registry inspection of its `2026-07-28` tag verified this
**linux/amd64** digest (about 1.21 GiB compressed):

```text
quay.io/jupyter/scipy-notebook@sha256:3f4d9163f207aa6f5e32b33c581f04fd143fbeba7b438f96ea8f074a1a3ce40f
```

No custom image build is needed. On a connected machine with Apptainer and
sufficient image/cache/temp space, confirm the HPC architecture, then stage:

```bash
cd /path/to/slurm-ml-demo
apptainer pull /chosen/shared/scipy-notebook.sif \
  docker://quay.io/jupyter/scipy-notebook@sha256:3f4d9163f207aa6f5e32b33c581f04fd143fbeba7b438f96ea8f074a1a3ce40f
apptainer exec --cleanenv /chosen/shared/scipy-notebook.sif \
  /opt/conda/bin/python -I -c \
  'import numpy, sklearn; from sklearn.datasets import load_digits; print(numpy.__version__, sklearn.__version__, load_digits().data.shape)'
sha256sum /chosen/shared/scipy-notebook.sif
```

Expect dataset shape `(1797, 64)`; record actual package versions and SIF hash.
For another CPU architecture select its matching manifest. Container execution
is **unverified here**: Docker extraction failed with "no space left on device",
and Apptainer/Singularity is unavailable. requirements.txt pins the tested host
versions, not verified image contents. Do not submit until the staged image
passes the import and manual workflow checks below.

## Private configuration and human submission

Copy [config.example.sh](config.example.sh) outside this project **and** the
skill repository. Use a private directory with mode 700 and file mode 600.
Fill PROJECT, RUN_ROOT, RUNTIME, IMAGE, concurrency and per-stage resource/site
options. Source only a profile you own and trust; it is executable Bash.
Choose export/invalid-dependency policy with your site. Load any required
runtime modules in the profile. Paths must be shared identically across nodes.

The runner binds project read-only at /project and a fresh writable run at /run;
runtime default mounts still follow site policy. Bind paths support spaces but
not colons or commas. It invokes `/opt/conda/bin/python -I` in the staged SIF.
No image pulls or downloads happen in jobs. First test the container manually:

```bash
cd /chosen/shared/slurm-ml-demo
mkdir -p /chosen/shared/private/manual-run
bash jobs/run.sh prepare "$PWD" /chosen/shared/private/manual-run /chosen/private/profile.sh
for fold in 0 1 2; do
  bash jobs/run.sh train "$PWD" /chosen/shared/private/manual-run /chosen/private/profile.sh "$fold"
done
bash jobs/run.sh aggregate "$PWD" /chosen/shared/private/manual-run /chosen/private/profile.sh
```

Then you can preview and submit from the cluster submission host:

```bash
cd /chosen/shared/slurm-ml-demo
bash jobs/submit.sh /chosen/private/profile.sh --preview
bash jobs/submit.sh /chosen/private/profile.sh
```

Preview creates a fresh private run and prints commands with symbolic IDs but
never calls sbatch. Submission creates another run and chains preparation →
array `0-2` → aggregation with whole-array afterok dependencies. External runs
hold logs and receipts; keep those and preview output private. Do not edit
prepared inputs while jobs run. On partial failure, inspect accepted IDs and
receipts before retrying; an unclear response may still represent an accepted
job. Failed prerequisites may leave dependents pending. No automatic retry or
cancellation is performed.

## Verification and skill usage

```bash
cd /path/to/slurm-ml-demo
python3 -m unittest discover -s tests -v
for file in jobs/*.sh jobs/*.sbatch config.example.sh; do bash -n "$file"; done
shellcheck jobs/*.sh jobs/*.sbatch config.example.sh  # if available
```

Four host ML tests and five scheduler/runtime-mock tests pass. Checks cover
reproducible splits, result rejection, dependency wiring, fold mapping, spooling,
spaces, preview and partial failures. Recorded host accuracies: 0.9833, 0.9933,
0.9733; mean 0.9833, population std 0.00818. ShellCheck unavailable. Container
imports/execution, SIF conversion and real Slurm remain unverified. No cluster
was contacted; mocks do not execute a container or establish site compatibility.

After the Python interfaces existed, the Slurm agent used
[slurm-workflow-skill](https://github.com/kashefy/slurm-workflow-skill) commit
`fbed1f053129e4f1c081bf9917bbf358e4dad14e`, reading SKILL.md and the setup,
dependencies, arrays and validation references. It preserved the explicit
Python inputs and produced external configuration and spool-safe wrappers.
This is development evidence, not a controlled skill-benefit benchmark.
Running or maintaining this demo requires no skill installation.
