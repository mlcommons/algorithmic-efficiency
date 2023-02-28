#!bin/bash

while getopts d:f:s:t:e:w:b: flag
do
    case "${flag}" in
        d) DATASET=${OPTARG};;
        f) FRAMEWORK=${OPTARG};;
        s) SUBMISSION_PATH=${OPTARG};;
        t) TUNING_SEARCH_SPACE=${OPTARG};;
        e) EXPERIMENT_NAME=${OPTARG};;
        w) WORKLOAD=${OPTARG};;
        b) DEBUG_MODE=${OPTARG};;
    esac
done

RUNS_BUCKET="gs://mlcommons-runs/"
ROOT_DATA_BUCKET="gs://mlcommons-data/"
ROOT_DATA_DIR="/data/"
EXPERIMENT_DIR="/experiment_runs/"

if [ "${DATASET}" == "imagenet" ]
then 
    DATA_DIR="${ROOT_DATA_DIR}/${DATASET}/${FRAMEWORK}"
    DATA_BUCKET="${ROOT_DATA_BUCKET}/${DATASET}/${FRAMEWORK}"
else
    DATA_DIR="${ROOT_DATA_DIR}/${DATASET}"
    DATA_BUCKET="${ROOT_DATA_BUCKET}/${DATASET}/"
fi

# Copy data from MLCommons bucket if data does not downloaded yet
if [ ! -d ${DATA_DIR} ]
then
    ./google-cloud-sdk/bin/gsutil -m cp -r ${DATA_BUCKET}/* ${DATA_DIR}
fi 

# Check GPU requirements and run experiment
python3 docker/scripts/check_gpu.py

# Optionally run workload
if $SUBMISSION_PATH
then
python3 algorithmic-efficiency/submission_runner.py \
    --framework=${FRAMEWORK}  \
    --workload=${WORKLOAD} \
    --submission_path=${SUBMISSION_PATH}  \
    --tuning_search_space=${TUNING_SEARCH_SPACE}  \
    --data_dir=${DATA_DIR} 
    --num_tuning_trials=1  \
    --experiment_dir=${EXPERIMENT_DIR}  \
    --experiment_name=${EXPERIMENT_NAME}  \

gsutil -m cp -r ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/* ${RUNS_BUCKET}
fi

# Keep main process running in debug mode to avoid the container from stopping
if [${DEBUG_MODE} == 'true']
then 
    while true
    do 
        sleep 1000
    done 
fi