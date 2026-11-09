#!/usr/bin/env bash
set -euo pipefail
umask 077
usage() { echo 'usage: submit.sh [PROFILE] [--preview]' >&2; exit 2; }
profile_arg='' preview=''
for arg in "$@"; do
  case "$arg" in
    --preview) [[ -z $preview ]] || usage; preview=$arg ;;
    --*) usage ;;
    *) [[ -z $profile_arg ]] || usage; profile_arg=$arg ;;
  esac
done
profile_path=${profile_arg:-${SLURM_ML_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/slurm-ml-demo}/cluster.sh}
[[ -f $profile_path && -r $profile_path ]] || {
  printf 'Configuration not found or unreadable: %s\nCopy config.example.sh there, set SLURM_ML_CONFIG_DIR, or pass PROFILE.\n' "$profile_path" >&2
  exit 2
}
profile=$(realpath "$profile_path")
source "$profile" # Trusted, user-owned shell configuration.
: "${PROJECT:?}" "${RUN_ROOT:?}" "${RUNTIME:?}" "${IMAGE:?}" "${ARRAY_CONCURRENCY:?}"
PROJECT=$(realpath "$PROJECT")
RUN_ROOT=$(realpath -m "$RUN_ROOT")
[[ $profile != "$PROJECT"/* && $RUN_ROOT != "$PROJECT" && $RUN_ROOT != "$PROJECT"/* ]] || { echo 'profile and runs must be outside project repository' >&2; exit 2; }
[[ $RUN_ROOT == /* && $IMAGE == /* && $ARRAY_CONCURRENCY =~ ^[1-3]$ ]] || { echo 'absolute paths and concurrency 1..3 required' >&2; exit 2; }
PREPARE_SBATCH_OPTIONS=("${PREPARE_SBATCH_OPTIONS[@]}")
TRAIN_SBATCH_OPTIONS=("${TRAIN_SBATCH_OPTIONS[@]}")
AGGREGATE_SBATCH_OPTIONS=("${AGGREGATE_SBATCH_OPTIONS[@]}")
for option in "${PREPARE_SBATCH_OPTIONS[@]}" "${TRAIN_SBATCH_OPTIONS[@]}" "${AGGREGATE_SBATCH_OPTIONS[@]}"; do
  case "$option" in --array*|--dependency*|--parsable*|--output*|--error*|-a|-d|-o|-e|--clusters*|-M) echo "reserved sbatch option: $option" >&2; exit 2;; esac
done
mkdir -p "$RUN_ROOT"
run=$(mktemp -d "$RUN_ROOT/run.XXXXXXXX")
mkdir "$run/logs" "$run/receipts"
printf 'Run: %s\n' "$run"
accepted=() previous='' cluster='' cluster_seen=0
fail() { printf '%s\n' "$1" >&2; printf 'Accepted: %s\n' "${accepted[*]:-(none)}" >&2; printf 'Inspect receipts: %s; do not retry blindly.\n' "$run/receipts" >&2; exit 1; }
for stage in prepare train aggregate; do
  case "$stage" in
    prepare) options=("${PREPARE_SBATCH_OPTIONS[@]}");;
    train) options=("${TRAIN_SBATCH_OPTIONS[@]}" "--array=0-2%$ARRAY_CONCURRENCY");;
    aggregate) options=("${AGGREGATE_SBATCH_OPTIONS[@]}");;
  esac
  [[ -z $previous ]] || options+=("--dependency=afterok:$previous")
  log="$run/logs/$stage-%j"
  [[ $stage != train ]] || log="$run/logs/train-%A_%a"
  command=(sbatch --parsable "${options[@]}" "--output=$log.out" "--error=$log.err" "$PROJECT/jobs/job.sbatch" "$PROJECT/jobs/run.sh" "$stage" "$PROJECT" "$run" "$profile")
  if [[ $preview == --preview ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
    previous="<$stage-jobid>"
    continue
  fi
  if ! "${command[@]}" >"$run/receipts/$stage.stdout" 2>"$run/receipts/$stage.stderr"; then
    fail "$stage submission failed"
  fi
  response=$(cat "$run/receipts/$stage.stdout")
  [[ $response =~ ^([0-9]+)(\;[[:alnum:]_.-]+)?$ ]] || fail "$stage returned malformed output; a job may have been accepted"
  previous=${BASH_REMATCH[1]}
  returned_cluster=${response#*;}; [[ $response == *';'* ]] || returned_cluster=''
  accepted+=("$stage=$response")
  if (( cluster_seen )) && [[ $cluster != "$returned_cluster" ]]; then fail 'returned cluster changed; submissions stopped'; fi
  cluster=$returned_cluster cluster_seen=1
  printf 'Accepted %s=%s\n' "$stage" "$response"
done
