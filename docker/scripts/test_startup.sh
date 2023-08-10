#!/bin/bash

# Test for startup script for docker startup script.
#
# Usage:
#     bash algorithmic-efficiency/docker/scripts/test_startup.sh


test_startup_script_short_flags() {
    command="bash algorithmic-efficiency/docker/scripts/startup.sh \
    -d mnist \
    -f jax \
    -s baselines/adamw/jax/submission.py \
    -w mnist \
    -t baselines/adamw/tuning_search_space.json \
    -e test_docker_entrypoint/adamw \
    -m 10 \
    -c false \
    -o true \
    -r false \
    -h $HOME
    "
    echo $command
    eval $command
}

test_startup_script_long_flags() {
    command="bash algorithmic-efficiency/docker/scripts/startup.sh 
    --dataset mnist \
    --framework jax \
    --submission_path baselines/adamw/jax/submission.py \
    --workload mnist \
    --tuning_search_space baselines/adamw/tuning_search_space.json \
    --experiment_name test_docker_entrypoint/adamw \
    --max_global_steps 10 \
    --save_checkpoints False \
    --overwrite true 
    --rsync_data false
    --home_dir $HOME"
    echo $command
    eval $command
}

test_startup_script_short_flags
test_startup_script_long_flags

