#!/bin/bash

#SBATCH --nodes=1 # give it a full node
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-26
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --exclusive #this will not allow other jobs to run on this cluster
#SBATCH --output=experiments/tests/updated_schedule_free/job_%A_%a.out
#SBATCH --error=experiments/tests/updated_schedule_free/job_%A_%a.err

# Usage: sbatch <this file>.sh [options]
# This script reads config.json and launches a sbatch job using task
# arrays where each job in the array corresponds to a training run
# for a workload given a random seed and tuning trial index.
# To generate the config.json use make_job_config.py.

set -x

# Function to read a JSON file and extract a value by key
read_json_value() {
  local json_file="$1"
  local index="$2"
  local key="$3"
  local value=$(jq -r ".[\"$index\"].$key" "$json_file")
  echo "$value"
}

# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Please install it."
    exit 1
fi

# Default values
REPO="europe-west4-docker.pkg.dev"
IMAGE="europe-west4-docker.pkg.dev/mlcommons-algoperf/algoperf-docker-repo/algoperf_pytorch_main:latest"
CONFIG_FILE="$HOME/algorithmic-efficiency/config.json"
LOGS_BUCKET="algoperf-runs"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo)
      REPO="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --config_file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --logs_bucket)
      LOGS_BUCKET="$2"
      shift 2
      ;;
    --max_global_steps)
      MAX_GLOBAL_STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Pull docker image
yes | gcloud auth configure-docker "$REPO"
docker pull "$IMAGE"

# Set variables from config file
FRAMEWORK=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "framework")
DATASET=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "dataset")
SUBMISSION_PATH=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "submission_path")
TUNING_SEARCH_SPACE=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "tuning_search_space")
EXPERIMENT_DIR=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "experiment_dir")
RNG_SEED=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "rng_seed")
WORKLOAD=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "workload")
HPARAM_START_INDEX=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "hparam_start_index")
HPARAM_END_INDEX=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "hparam_end_index")
NUM_TUNING_TRIALS=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "num_tuning_trials")
TUNING_RULESET=$(read_json_value "$CONFIG_FILE" "$TASK_ID" "tuning_ruleset")

DOCKER_CMD=(
  docker run
  -v /opt/data/:/data/
  -v "$HOME/experiment_runs:/experiment_runs"
  -v "$HOME/algorithmic-efficiency/:/algorithmic-efficiency/"
  -v "$HOME/submissions_algorithms/:/algorithmic-efficiency/submissions_algorithms"
  -v "$HOME/algorithmic-efficiency/docker/scripts/startup.sh:/algorithmic-efficiency/docker/scripts/startup.sh"
  --gpus all
  --ipc=host
  "$IMAGE"
  -d "$DATASET"
  -f "$FRAMEWORK"
  -s "$SUBMISSION_PATH"
  -w "$WORKLOAD"
  -t "$TUNING_SEARCH_SPACE"
  -e "$EXPERIMENT_DIR"
  -c False
  -o True
  --rng_seed "$RNG_SEED"
  --hparam_start_index "$HPARAM_START_INDEX"
  --hparam_end_index "$HPARAM_END_INDEX"
  --num_tuning_trials "$NUM_TUNING_TRIALS"
  --tuning_ruleset "$TUNING_RULESET"
  -i true
  -r false
  --logs_bucket "$LOGS_BUCKET"
)

if [ -n "$MAX_GLOBAL_STEPS" ]; then
  DOCKER_CMD+=(-m "$MAX_GLOBAL_STEPS")
fi

"${DOCKER_CMD[@]}"
