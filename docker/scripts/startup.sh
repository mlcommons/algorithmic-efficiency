#!/bin/bash
# Bash script to run submission_runner.py.
# The primary purpose of this script is to serve as
# an entrypoint for our Docker images.
# Internal contributors may enable this script
# to download data and upload experiment results to
# our algorithmic-efficiency repo. To do so 
# set the -i flag to true.

# Defaults
DEBUG_MODE="false"

while getopts d:f:s:t:e:w:b:m:o:c:r: flag
do
    case "${flag}" in
        d) DATASET=${OPTARG};;
        f) FRAMEWORK=${OPTARG};;
        s) SUBMISSION_PATH=${OPTARG};;
        t) TUNING_SEARCH_SPACE=${OPTARG};;
        e) EXPERIMENT_NAME=${OPTARG};;
        w) WORKLOAD=${OPTARG};;
        b) DEBUG_MODE=${OPTARG};;
        m) MAX_STEPS=${OPTARG};;
        o) OVERWRITE=${OPTARG};;
        c) SAVE_CHECKPOINTS=${OPTARG};;
        r) RSYNC_DATA=${OPTARG};;
        i) INTERNAL_CONTRIBUTOR_MODE=${OPTARG};;
    esac
done

# Check if arguments are valid
VALID_DATASETS=("criteo1tb" "imagenet"  "fastmri" "ogbg" "librispeech" \
                "wmt" "mnist")
VALID_WORKLOADS=("criteo1tb" "imagenet_resnet" "imagenet_vit" "fastmri" "ogbg" \
                 "wmt" "librispeech_deepspeech" "librispeech_conformer" "mnist")

if [[ -n ${DATASET+x} ]]; then 
    if [[ ! " ${VALID_DATASETS[@]} " =~ " $DATASET " ]]; then
        echo "Error: invalid argument for dataset (d)."
        exit 1
    fi
fi

if [[ -n ${WORKLOAD+x} ]]; then 
    if [[ ! " ${VALID_WORKLOADS[@]} " =~ " $WORKLOAD " ]]; then
        echo "Error: invalid argument for workload (w)."
        exit 1
    fi
fi

# Set data and experiment paths
ROOT_DATA_BUCKET="gs://mlcommons-data"
ROOT_DATA_DIR="/data"

EXPERIMENT_BUCKET="gs://mlcommons-runs"
EXPERIMENT_DIR="/experiment_runs"

# Set run command prefix depending on framework
if [[ "${FRAMEWORK}" == "jax" ]]; then
    COMMAND_PREFIX="python3"
else 
    COMMAND_PREFIX="torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8"
fi

# Set data directory and bucket (bucket is only relevant in internal mode)
if [[ "${DATASET}" == "imagenet" ]]; then 
    DATA_DIR="${ROOT_DATA_DIR}/${DATASET}/${FRAMEWORK}"
    DATA_BUCKET="${ROOT_DATA_BUCKET}/${DATASET}/${FRAMEWORK}"
elif [[ ! -z "${DATASET}" ]]; then
    DATA_DIR="${ROOT_DATA_DIR}/${DATASET}"
    DATA_BUCKET="${ROOT_DATA_BUCKET}/${DATASET}"
fi

# Copy data from MLCommons bucket if data has not been downloaded yet
if [[ -z ${INTERNAL_CONTRIBUTOR_MODE+x} ]]; then
    INTERNAL_CONTRIBUTOR_MODE='false' # Set default for contributor mode to false
fi 

if [[ -z ${RSYNC_DATA+x} ]]; then 
    RSYNC_DATA='true' # Set default value for rsync to true
fi 

if [[ ! -z $DATA_DIR ]] && [[ ! -d ${DATA_DIR} ]]; then
    mkdir -p ${DATA_DIR}
fi 

if [[ ! -z $DATA_DIR ]] && [[ ${RSYNC_DATA} == 'true' ]] && [[ $INTERNAL_CONTRIBUTOR_MODE == 'true' ]]; then
    ./google-cloud-sdk/bin/gsutil -m rsync -r ${DATA_BUCKET} ${DATA_DIR}
fi 

# Optionally run workload if SUBMISSION_PATH is set
if [[ ! -z ${SUBMISSION_PATH+x} ]]; then
    NOW=$(date +"%m-%d-%Y-%H-%M-%S")
    LOG_DIR="/logs"
    LOG_FILE="$LOG_DIR/${WORKLOAD}_${FRAMEWORK}_${NOW}.log"
    mkdir -p ${LOG_DIR}
    cd algorithmic-efficiency

    # Optionally define max steps flag for submission runner 
    if [[ ! -z ${MAX_STEPS+x} ]]; then 
        MAX_STEPS_FLAG="--max_global_steps=${MAX_STEPS}"
    fi

    # Set overwrite flag to false by default if not set
    if [[  -z ${OVERWRITE+x} ]]; then 
        OVERWRITE='False'
    fi

    if [[  -z ${SAVE_CHECKPOINTS+x} ]]; then 
        SAVE_CHECKPOINTS='True'
    fi

    # Define special flags for imagenet and librispeech workloads
    if [[ ${DATASET} == 'imagenet' ]]; then 
        SPECIAL_FLAGS="--imagenet_v2_data_dir=${DATA_DIR}"
    elif [[ ${DATASET} == 'librispeech' ]]; then 
        SPECIAL_FLAGS="--librispeech_tokenizer_vocab_path=${DATA_DIR}/spm_model.vocab"
    fi 

    # Optionally run torch compile
    if [[ ${FRAMEWORK} == 'pytorch' ]]; then
        TORCH_COMPILE_FLAG="torch_compile=True"
    fi
    
    # The TORCH_RUN_COMMAND_PREFIX is only set if FRAMEWORK is "pytorch"
    COMMAND="${COMMAND_PREFIX} submission_runner.py \
        --framework=${FRAMEWORK}  \
        --workload=${WORKLOAD} \
        --submission_path=${SUBMISSION_PATH}  \
        --tuning_search_space=${TUNING_SEARCH_SPACE}  \
        --data_dir=${DATA_DIR} \
        --num_tuning_trials=1  \
        --experiment_dir=${EXPERIMENT_DIR}  \
        --experiment_name=${EXPERIMENT_NAME} \
        --overwrite=${OVERWRITE} \
        --save_checkpoints=${SAVE_CHECKPOINTS} \
        ${MAX_STEPS_FLAG}  \
        ${SPECIAL_FLAGS} \
        ${TORCH_COMPILE_FLAG} 2>&1 | tee -a ${LOG_FILE}"
    echo $COMMAND > ${LOG_FILE}
    eval $COMMAND

    if [[ $INTERNAL_CONTRIBUTOR_MODE == 'true' ]]; then 
        /google-cloud-sdk/bin/gsutil -m cp -r ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${WORKLOAD}_${FRAMEWORK} ${EXPERIMENT_BUCKET}/${EXPERIMENT_NAME}/
        /google-cloud-sdk/bin/gsutil -m cp ${LOG_FILE} ${EXPERIMENT_BUCKET}/${EXPERIMENT_NAME}/${WORKLOAD}_${FRAMEWORK}/
    fi

fi

# Keep main process running in debug mode to avoid the container from stopping
if [[ ${DEBUG_MODE} == 'true' ]]
then 
    while true
    do 
        sleep 1000
    done 
fi
