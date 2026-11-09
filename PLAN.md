# Minimal demo checklist

Goal: sklearn digits → three stratified folds → scaled RBF SVM → accuracy
summary, first locally and then through containerized, human-submitted Slurm.
No tuning, tracking services, custom image builds, downloads in jobs, or cluster
connections. Curriculearn is not involved.

- [x] ML core: prepare.py, train.py, aggregate.py. All three folds run locally.
- [x] Container: select an existing scipy-notebook CPU image; record what was
  verified and what still needs human SIF staging. No invented digest.
- [x] Slurm layer: use the workflow skill after the Python contracts exist;
  Bash owns folds, batch array passes its task ID, launcher chains all stages.
- [x] Validation: check split reproducibility/coverage, aggregation rejection,
  Bash syntax, dependency wiring/failure reporting and script relocation locally.
- [x] Handoff: README, external private config example, MIT license,
  contributor instructions without mandating any skill, and actual check results.

## Agreed interfaces

`python prepare.py --output-dir RUN/data` writes data.npz (X,y), manifest.json,
and fold-0/{train,test}.npy through fold-2/{train,test}.npy. Use all 1,797 digits
examples, StratifiedKFold(3, shuffle=True, random_state=42). Manifest records
seed, data hash, and each fold's ID, relative index filenames, hashes and counts.

`python train.py --data DATA --train-index TRAIN --test-index TEST --output JSON`
is fold- and Slurm-agnostic. Fit Pipeline(StandardScaler, SVC with kernel=rbf,
C=1.0, gamma=scale). Output accuracy, correct_count, train_count, test_count,
data_sha256, train_sha256, test_sha256 and model parameters. Use file SHA-256 for
artifact identity; no model registry or prediction archive.

`python aggregate.py --manifest MANIFEST --results-dir RESULTS --output JSON`
requires exactly fold-0.json, fold-1.json, fold-2.json and validates their
identity/counts/metrics. Report fold accuracies, arithmetic mean and population
standard deviation (ddof=0). Fail on missing or inconsistent results.

Host commands use existing Python; container commands bind project read-only at
/project and a fresh writable run directory at /run. Keep image/runtime paths,
shared host paths and per-stage sbatch options in a user-selected external
private profile. No real site configuration is needed to author/test this demo.

The Slurm layer must preserve these interfaces. Record the skill commit used
and concrete findings in the handoff; the project must run without that skill.

## Remaining human checks

- [ ] Stage and verify the pinned image/SIF on a capable connected machine.
  Registry lookup passed; local Docker extraction ran out of storage, so no
  container-execution success is claimed.
- [ ] Fill the external profile and run the three stages manually in the SIF.
- [ ] Check site options/resources, then choose when to submit on the cluster.

Host verification: four ML tests and seven scheduler/runtime-mock tests
pass. ShellCheck unavailable. Container and real scheduler behavior unverified.

Configuration defaults to `${XDG_CONFIG_HOME:-$HOME/.config}/slurm-ml-demo/cluster.sh`.
Set SLURM_ML_CONFIG_DIR to change its directory or pass an explicit profile file;
selection does not require source edits and is tested with local mocks.
