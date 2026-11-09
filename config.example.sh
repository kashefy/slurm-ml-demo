# Copy OUTSIDE this repository and the workflow skill repository; chmod 600.
# Default: ${XDG_CONFIG_HOME:-$HOME/.config}/slurm-ml-demo/cluster.sh.
# Override its directory with SLURM_ML_CONFIG_DIR; no code changes required.
# The profile itself must be readable at the same path on compute nodes.
# Source only a profile you own and trust. All paths must be shared identically
# on submission and compute nodes. Stage the image yourself before submission.
PROJECT='/absolute/shared/path/to/slurm-ml-demo'
RUN_ROOT='/absolute/shared/private/demo-runs' # Outside both repositories; mode 700.
RUNTIME='/absolute/path/to/apptainer' # Or singularity; already available on nodes.
IMAGE='/absolute/shared/path/to/existing-cpu-image.sif'
ARRAY_CONCURRENCY=3
# Populate with site-approved resources/account/partition/QoS as applicable.
# Choose export and invalid-dependency policy explicitly with your site.
# Do not put --array, --dependency, --parsable, log paths or script args here.
PREPARE_SBATCH_OPTIONS=()
TRAIN_SBATCH_OPTIONS=()
AGGREGATE_SBATCH_OPTIONS=()
