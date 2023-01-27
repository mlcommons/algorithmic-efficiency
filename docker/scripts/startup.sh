#!/bin/sh

while getopts f:s:t:e:w: flag
do
    case "${flag}" in
        f) FRAMEWORK=${OPTARG};;
        s) SUBMISSION_PATH=${OPTARG};;
        t) TUNING_SEARCH_SPACE=${OPTARG};;
        e) EXPERIMENT_NAME=${OPTARG};;
        w) WORKLOAD=${OPTARG};;
    esac
done

cd ..
./google-cloud-sdk/bin/gsutil -m cp -r gs://mlcommons-data/criteo/criteo_parts/* data/
# yes | python3 dataset_setup.py --data_dir=~/data --temp_dir=~/data --all=False --criteo

cd algorithmic-efficiency
echo "Checking GPU presence and CUDA linking"

python3 docker/scripts/check_gpu.py
echo "Running Experiment"

python3 submission_runner.py     --framework=$FRAMEWORK     --workload=$WORKLOAD --submission_path=$SUBMISSION_PATH     --tuning_search_space=$TUNING_SEARCH_SPACE --data_dir=../data --num_tuning_trials=1 --experiment_dir=../experiment_runs/  --experiment_name=$EXPERIMENT_NAME

cd ..
./google-cloud-sdk/bin/gsutil -m cp -r experiment_runs/* gs://mlcommons-data/experiment_runs/
