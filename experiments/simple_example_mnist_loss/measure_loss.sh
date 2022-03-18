# This script will run an MNIST training workload and save measurements as CSV and metadata as JSON.
#
# Author: Daniel Snider <danielsnider12@gmail.com>
#
# Usage:
## Configure output location:
## export LOGGING_DIR=./experiments/simple_example_mnist_loss/logs && mkdir -p $LOGGING_DIR/
##
## Run this script:
## bash ./experiments/simple_example_mnist_loss/measure_loss.sh 2>&1 | tee -a $LOGGING_DIR/console_output.log

set -e # exit on error

# Quick data collection
NUM_TRIALS='1'

# Full data collection
# NUM_TRIALS='20'

echo -e "\n[INFO $(date +"%d-%I:%M%p")] Starting."
set -x
python3 algorithmic_efficiency/submission_runner.py \
    --framework=jax \
    --workload=mnist_jax \
    --submission_path=baselines/mnist/mnist_jax/submission.py \
    --tuning_search_space=baselines/mnist/tuning_search_space.json \
    --num_tuning_trials=$NUM_TRIALS \
    --eval_frequency_override='10 step' \
    --logging_dir=$LOGGING_DIR
set +x

# Combine all tuning trials into one CSV
python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOGGING_DIR')"

echo "[INFO $(date +"%d-%I:%M%p")] Generated files:"
find $LOGGING_DIR
echo "[INFO $(date +"%d-%I:%M%p")] Finished."
