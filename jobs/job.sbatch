#!/usr/bin/env bash
set -euo pipefail
[[ $# == 5 ]] || { echo 'usage: job.sbatch RUNNER STAGE PROJECT RUN PROFILE' >&2; exit 2; }
runner=$1 stage=$2
shift 2
if [[ $stage == train ]]; then
  exec bash "$runner" "$stage" "$@" "${SLURM_ARRAY_TASK_ID:?missing array task ID}"
fi
exec bash "$runner" "$stage" "$@"
