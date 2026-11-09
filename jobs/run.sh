#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 || $# == 5 ]] || { echo 'usage: run.sh STAGE PROJECT RUN PROFILE [FOLD]' >&2; exit 2; }
stage=$1 project=$2 run=$3 profile=$4
source "$profile" # Trusted, user-owned shell configuration.
: "${RUNTIME:?}" "${IMAGE:?}"
[[ $project == /* && $run == /* && $profile == /* && $IMAGE == /* ]] || { echo 'paths must be absolute' >&2; exit 2; }
case "$stage" in
  prepare) args=(prepare.py --output-dir /run/data) ;;
  train)
    folds=(0 1 2)
    [[ ${5-} =~ ^[012]$ ]] || { echo 'array index must be 0, 1 or 2' >&2; exit 2; }
    fold=${folds[$5]}
    args=(train.py --data /run/data/data.npz --train-index "/run/data/fold-$fold/train.npy" --test-index "/run/data/fold-$fold/test.npy" --output "/run/results/fold-$fold.json") ;;
  aggregate) args=(aggregate.py --manifest /run/data/manifest.json --results-dir /run/results --output /run/summary.json) ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
# Bind specifications cannot represent colons or commas in host paths.
[[ $project != *[:,]* && $run != *[:,]* ]] || { echo 'bind paths cannot contain colons or commas' >&2; exit 2; }
exec "$RUNTIME" exec --cleanenv --bind "$project:/project:ro" --bind "$run:/run:rw" --pwd /project "$IMAGE" /opt/conda/bin/python -I "${args[@]}"
