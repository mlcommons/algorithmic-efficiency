#!/bin/bash

# Usage:
# ./algorithmic-efficiency/scoring/utils/slurm/run_submission.sh \
#   --submission_path submissions_algorithms/submissions/self_tuning/schedule_free_adamw_v2
#
# Note: --dry_run is true by default (sets MAX_GLOBAL_STEPS=10).
# To perform a full run, explicitly set --dry_run false.

set -e
set -x

# --- Global Variables ---
SUBMISSION_PATH=""
DRY_RUN=true
MAX_GLOBAL_STEPS=10
SUBMISSION_NAME=""
RULESET=""
FRAMEWORK=""
ARRAY_RANGE=""
WORKLOADS=""
PROJECT="mlcommons-algoperf"
ARTIFACT_REPO="europe-west4-docker.pkg.dev/mlcommons-algoperf/algoperf-docker-repo"

# --- Helper Functions ---

install_yq() {
  if ! command -v yq &> /dev/null; then
    echo "yq not found. Attempting to install locally to $HOME/.local/bin..."
    mkdir -p "$HOME/.local/bin"
    local OS=$(uname | tr '[:upper:]' '[:lower:]')
    local ARCH=$(uname -m)
    case "$ARCH" in
      x86_64) ARCH="amd64" ;;
      aarch64) ARCH="arm64" ;;
    esac
    
    local YQ_URL="https://github.com/mikefarah/yq/releases/latest/download/yq_${OS}_${ARCH}"
    if command -v curl &> /dev/null; then
      curl -L "$YQ_URL" -o "$HOME/.local/bin/yq"
    elif command -v wget &> /dev/null; then
      wget "$YQ_URL" -O "$HOME/.local/bin/yq"
    else
      echo "Error: Neither curl nor wget found. Please install yq manually: https://github.com/mikefarah/yq"
      exit 1
    fi
    chmod +x "$HOME/.local/bin/yq"
    export PATH="$HOME/.local/bin:$PATH"
    echo "yq installed successfully to $HOME/.local/bin"
  fi
}

check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo "Error: $1 could not be found. Please install it."
    exit 1
  fi
}

verify_environment() {
  if [[ "$PWD" != "$HOME" ]]; then
    echo "Error: This script must be run from your home directory ($HOME)."
    echo "Expected directory structure:"
    echo "  $HOME/"
    echo "  ├── algorithmic-efficiency/"
    echo "  └── submissions_algorithms/"
    exit 1
  fi

  if [[ ! -d "algorithmic-efficiency" || ! -d "submissions_algorithms" ]]; then
    echo "Error: Required repositories not found in the current directory."
    echo "Please ensure both 'algorithmic-efficiency' and 'submissions_algorithms' are present in $HOME."
    exit 1
  fi

  install_yq
  check_command "jq"
}

parse_flags() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --submission_path)
        SUBMISSION_PATH="$2"
        shift 2
        ;;
      --dry_run)
        DRY_RUN="$2"
        shift 2
        ;;
      --workloads)
        WORKLOADS="$2"
        shift 2
        ;;
      --project)
        PROJECT="$2"
        shift 2
        ;;
      *)
        echo "Unknown option $1"
        exit 1
        ;;
    esac
  done

  if [ -z "$SUBMISSION_PATH" ]; then
    echo "Error: --submission_path is required."
    exit 1
  fi

  if [ "$DRY_RUN" = false ]; then
    MAX_GLOBAL_STEPS=""
  fi

  if [ "$PROJECT" = "mlcommons-algoperf" ]; then
    ARTIFACT_REPO="europe-west4-docker.pkg.dev/mlcommons-algoperf/algoperf-docker-repo"
    LOGS_BUCKET="algoperf-runs"
  elif [ "$PROJECT" = "training-algorithms-external" ]; then
    ARTIFACT_REPO="us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo"
    LOGS_BUCKET="internal-algoperf-runs"
  else
    echo "Error: Unknown project $PROJECT. Supported: mlcommons-algoperf, training-algorithms-external."
    exit 1
  fi
}

extract_submission_info() {
  SUBMISSION_NAME=$(basename "$SUBMISSION_PATH")
  local info_file="$SUBMISSION_PATH/submission_info.yml"

  if [ ! -f "$info_file" ]; then
    echo "Error: $info_file not found."
    exit 1
  fi

  local raw_ruleset=$(yq eval '.ruleset' "$info_file" | tr '[:upper:]' '[:lower:]')
  FRAMEWORK=$(yq eval '.framework' "$info_file" | tr '[:upper:]' '[:lower:]')

  # Parse ruleset by checking for substrings "self" or "external"
  if [[ "$raw_ruleset" == *"self"* ]]; then
    RULESET="self"
  elif [[ "$raw_ruleset" == *"external"* ]]; then
    RULESET="external"
  else
    echo "Error: Expected 'ruleset' in $info_file to contain 'self' or 'external' (got '$raw_ruleset')."
    exit 1
  fi

  # Verify framework
  if [[ "$FRAMEWORK" != "jax" && "$FRAMEWORK" != "pytorch" ]]; then
    echo "Error: 'framework' in $info_file must be either 'jax' or 'pytorch' (got '$FRAMEWORK')."
    exit 1
  fi

  echo "Submission Name: $SUBMISSION_NAME"
  echo "Ruleset: $RULESET"
  echo "Framework: $FRAMEWORK"
  echo "Dry Run: $DRY_RUN"
  echo "Max Global Steps: $MAX_GLOBAL_STEPS"
}

generate_config() {
  local exp_prefix="submissions_a100_dry_run"
  if [ "$DRY_RUN" = false ]; then
    exp_prefix="submissions_a100"
  fi

  local workloads_flag=""
  if [ -n "$WORKLOADS" ]; then
    workloads_flag="--workloads=$WORKLOADS"
  fi

  docker run \
    --rm \
    -v "$(pwd)":/algorithmic-efficiency \
    -w /algorithmic-efficiency \
    --entrypoint python \
    "$ARTIFACT_REPO/algoperf_${FRAMEWORK}_main:latest" \
    algorithmic-efficiency/scoring/utils/slurm/make_job_config.py \
    --framework="$FRAMEWORK" \
    --tuning_ruleset="$RULESET" \
    --submission_path="$SUBMISSION_PATH/submission.py" \
    --experiment_dir="${exp_prefix}/$SUBMISSION_NAME" \
    $workloads_flag

  mv config.json "$SUBMISSION_NAME.json"
}

prepare_sbatch_array() {
  local num_jobs=$(jq 'length' "$SUBMISSION_NAME.json")
  if [[ "$num_jobs" -eq 0 ]]; then
    echo "Error: No jobs found in $SUBMISSION_NAME.json."
    exit 1
  fi

  ARRAY_RANGE="0-$((num_jobs - 1))"
  echo "Number of jobs: $num_jobs"
  echo "Sbatch array range: $ARRAY_RANGE"

  mkdir -p "experiments/tests/$SUBMISSION_NAME"
}

run_sbatch() {
  local sbatch_cmd=(
    sbatch
    --array="$ARRAY_RANGE"
    --output="experiments/tests/$SUBMISSION_NAME/job_%A_%a.out"
    --error="experiments/tests/$SUBMISSION_NAME/job_%A_%a.err"
    "algorithmic-efficiency/scoring/utils/slurm/run_jobs.sh"
    --config_file "$(pwd)/$SUBMISSION_NAME.json"
    --image "$ARTIFACT_REPO/algoperf_${FRAMEWORK}_main:latest"
    --logs_bucket "$LOGS_BUCKET"
  )

  local req_file="$SUBMISSION_PATH/requirements.txt"
  if [ -f "$req_file" ]; then
    sbatch_cmd+=(--additional_requirements_path "$req_file")
  fi

  if [ -n "$MAX_GLOBAL_STEPS" ]; then
    sbatch_cmd+=(--max_global_steps "$MAX_GLOBAL_STEPS")
  fi

  "${sbatch_cmd[@]}"
}

# --- Main ---

main() {
  verify_environment
  parse_flags "$@"
  extract_submission_info
  generate_config
  prepare_sbatch_array
  run_sbatch
}

main "$@"
