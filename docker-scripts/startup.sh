#!/bin/sh

echo "This trick works!!"

echo "Running OGBG"
cd algorithmic-efficiency
python3 submission_runner.py     --framework=jax     --workload=ogbg --submission_path=reference_algorithms/target_setting_algorithms/jax_nesterov.py     --tuning_search_space=reference_algorithms/target_setting_algorithms/ogbg/tuning_search_space.json --data_dir=../data/ --num_tuning_trials=1 --experiment_dir=../experiment_runs/  --max_global_steps=10 --experiment_name=docker_ogbg

cd ..
./google-cloud-sdk/bin/gsutil -m cp -r experiment_runs/* gs://mlcommons-data/experiment_runs/docker_ogbg 
