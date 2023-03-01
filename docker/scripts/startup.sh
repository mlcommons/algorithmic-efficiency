#!/bin/bash

# Defaults
DEBUG_MODE="false"

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

ROOT_DATA_BUCKET="gs://mlcommons-data"
ROOT_DATA_DIR="/data"

EXPERIMENT_BUCKET="gs://mlcommons-runs"
EXPERIMENT_DIR="/experiment_runs"

if [ "${FRAMEWORK}" == "pytorch" ]
then 
    TORCH_RUN_COMMAND_PREFIX="torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8"
fi

if [ "${DATASET}" == "imagenet" ]
then 
    DATA_DIR="${ROOT_DATA_DIR}/${DATASET}/${FRAMEWORK}"
    DATA_BUCKET="${ROOT_DATA_BUCKET}/${DATASET}/${FRAMEWORK}"
else
    DATA_DIR="${ROOT_DATA_DIR}/${DATASET}"
    DATA_BUCKET="${ROOT_DATA_BUCKET}/${DATASET}/"
fi

# Copy data from MLCommons bucket if data has not been downloaded yet
if [ ! -d ${DATA_DIR} ]
then
    mkdir -p ${DATA_DIR}
    ./google-cloud-sdk/bin/gsutil -m cp -r ${DATA_BUCKET}/* ${DATA_DIR}
fi 

# Check GPU requirements and run experiment
# python3 scripts/check_gpu.py

# Optionally run workload
if [ ! -z ${SUBMISSION_PATH+x} ]
    then
    LOG_DIR="/logs"
    LOG_FILE="$LOG_DIR/${WORKLOAD}_${FRAMEWORK}_submission.log"
    mkdir -p ${LOG_DIR}
    cd algorithmic-efficiency
    # The TORCH_RUN_COMMAND_PREFIX is only set if FRAMEWORK is "pytorch"
    ${TORCH_RUN_COMMAND_PREFIX} python3 submission_runner.py \
        --framework=${FRAMEWORK}  \
        --workload=${WORKLOAD} \
        --submission_path=${SUBMISSION_PATH}  \
        --tuning_search_space=${TUNING_SEARCH_SPACE}  \
        --data_dir=${DATA_DIR} \
        --num_tuning_trials=1  \
        --experiment_dir=${EXPERIMENT_DIR}  \
        --experiment_name=${EXPERIMENT_NAME}  2>&1 | tee ${LOG_FILE}

    ./google-cloud-sdk/bin/ gsutil -m cp -r ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/* gs://${EXPERIMENT_BUCKET}/${EXPERIMENT_NAME}
    ./google-cloud-sdk/bin/ gsutil -m cp -r ${LOG_FILE} gs://${EXPERIMENT_BUCKET}/${EXPERIMENT_NAME}

fi

# Keep main process running in debug mode to avoid the container from stopping
if [ ${DEBUG_MODE} == 'true' ]
then 
    while true
    do 
        sleep 1000
    done 
fi