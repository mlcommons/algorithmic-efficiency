#!/bin/bash
# Bash script to run submission_runner.py.
# The primary purpose of this script is to serve as
# an entrypoint for our Docker images.
# Internal contributors may enable this script
# to download data and upload experiment results to
# our algorithmic-efficiency repo. To do so 
# set the -i flag to true.

function usage() {
    cat <<USAGE

    Usage:
        $0  [--dataset dataset] [--framework framework] [--submission_path submission_path]
            [--tuning_search_space tuning_search_space] [--experiment_name experiment_name] 
            [--workload workload] [--max_global_steps max_global_steps] [--rsync_data rsync_data]
            [--internal_contributor true] [--traindiffs_test false]

    Options:
        -d | --dataset:                 Can be imagenet, criteo1tb, ogbg, fastmri, wmt, librispeech.
        -f | --framework:               Can be jax or pytorch.
        -s | --submission_path:         Path to submission module. If relative path, from algorithmic-efficiency top directory.
        -t | --tuning_search_space:     Path to tuning search space. If relative path, from algorithmic-efficiency top directory.
        -e | --experiment_name:         Name of experiment.
        -w | --workload:                Can be imagenet_resnet, imagenet_vit, criteo1tb, fastmri,
                                        wmt, librispeech_deepspeech, librispeech_conformer or variant workload.
        -a | --keep_container_alive:    If true, docker container will be kept alive. Useful for 
                                        developing or debugging.
        -m | --max_global_steps:        Maximum number of global steps for submission.
        -o | --overwrite:               If true, overwrite the experiment directory with the identical
                                        experiment name.
        -c | --save_checkpoints         If true, save all checkpoints from all evals. 
        -r | --rsync_data:              If true and if --internal_contributor mode is true, rsync data
                                        from internal GCP bucket.
        -i | --internal_contributor:    If true, allow rsync of data and transfer of experiment results 
                                        with GCP project.
        --traindiffs_test:              If true, ignore all other options and run the traindiffs test.
USAGE
    exit 1
}

# Defaults
TEST="false"
INTERNAL_CONTRIBUTOR_MODE="false"
HOME_DIR=""
RSYNC_DATA="true"
OVERWRITE="false"
SAVE_CHECKPOINTS="true"

# Pass flag
while [ "$1" != "" ]; do
    case $1 in
	--traindiffs_test)
	    shift
            TEST=$1
	    ;;
        -d | --dataset) 
            shift
            DATASET=$1
            ;;
        -f | --framework)
            shift 
            FRAMEWORK=$1
            ;;
        -s | --submission_path)
            shift
            SUBMISSION_PATH=$1
            ;;
        -t | --tuning_search_space)
            shift
            TUNING_SEARCH_SPACE=$1
            ;;
        -e | --experiment_name)
            shift
            EXPERIMENT_NAME=$1
            ;;
        -w | --workload)
            shift
            WORKLOAD=$1
            ;;
        -a | --keep_container_alive)
            shift 
            KEEP_CONTAINER_ALIVE=$1
            ;;
        -m | --max_global_steps)
            shift
            MAX_GLOBAL_STEPS=$1
            ;;
        -o | --overwrite)
            shift
            OVERWRITE=$1
            ;;
        -c | --save_checkpoints)
            shift
            SAVE_CHECKPOINTS=$1
            ;;
        -r | --rsync_data)
            shift
            RSYNC_DATA=$1
            ;;
        -i | --internal_contributor)
            shift
            INTERNAL_CONTRIBUTOR_MODE=$1
            ;;
        -h | --home_dir)
            shift
            HOME_DIR=$1
            ;;
        *) 
            usage 
            exit 1
            ;;
    esac
    shift 
done

if [[ ${TEST} == "true" ]]; then
  cd algorithmic-efficiency
  COMMAND="python3 tests/test_traindiffs.py"
  echo $COMMAND
  eval $COMMAND
  exit
fi

# Check if arguments are valid
VALID_DATASETS=("criteo1tb" "imagenet"  "fastmri" "ogbg" "librispeech" \
                "wmt" "mnist")
VALID_WORKLOADS=("criteo1tb" "imagenet_resnet" "imagenet_resnet_silu" "imagenet_resnet_gelu" \
                 "imagenet_resnet_large_bn_init" "imagenet_vit" "imagenet_vit_glu" \
                 "imagenet_vit_post_ln" "imagenet_vit_map" "fastmri" "ogbg" \
                 "criteo1tb_resnet" "criteo1tb_layernorm" "criteo1tb_embed_init" \
                 "wmt" "wmt_post_ln" "wmt_attention_temp" "wmt_glu_tanh" \
                 "librispeech_deepspeech" "librispeech_conformer" "mnist" \
                 "conformer_layernorm" "conformer_attention_temperature" \
                 "conformer_gelu" "fastmri_model_size" "fastmri_tanh" \
                 "fastmri_layernorm" "ogbg_gelu" "ogbg_silu" "ogbg_model_size")

# Set data and experiment paths
ROOT_DATA_BUCKET="gs://mlcommons-data"
ROOT_DATA_DIR="${HOME_DIR}/data"

EXPERIMENT_BUCKET="gs://mlcommons-runs"
EXPERIMENT_DIR="${HOME_DIR}/experiment_runs"

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

if [[ ! -z $DATA_DIR ]] && [[ ! -d ${DATA_DIR} ]]; then
    mkdir -p ${DATA_DIR}
fi 

if [[ ! -z $DATA_DIR ]] && [[ ${RSYNC_DATA} == 'true' ]] && [[ $INTERNAL_CONTRIBUTOR_MODE == 'true' ]]; then
    ./google-cloud-sdk/bin/gsutil -m rsync -r ${DATA_BUCKET} ${DATA_DIR}
fi 

# Optionally run workload if SUBMISSION_PATH is set
if [[ ! -z ${SUBMISSION_PATH+x} ]]; then
    NOW=$(date +"%m-%d-%Y-%H-%M-%S")
    LOG_DIR="${HOME_DIR}/logs"
    LOG_FILE="$LOG_DIR/${WORKLOAD}_${FRAMEWORK}_${NOW}.log"
    mkdir -p ${LOG_DIR}
    cd algorithmic-efficiency

    # Optionally define max steps flag for submission runner 
    if [[ ! -z ${MAX_GLOBAL_STEPS+x} ]]; then 
        MAX_STEPS_FLAG="--max_global_steps=${MAX_GLOBAL_STEPS}"
    fi

    # Define special flags for imagenet and librispeech workloads
    if [[ ${DATASET} == "imagenet" ]]; then 
        SPECIAL_FLAGS="--imagenet_v2_data_dir=${DATA_DIR}"
    elif [[ ${DATASET} == "librispeech" ]]; then 
        SPECIAL_FLAGS="--librispeech_tokenizer_vocab_path=${DATA_DIR}/spm_model.vocab"
    fi 

    # Optionally run torch compile
    if [[ ${FRAMEWORK} == "pytorch" ]]; then
        TORCH_COMPILE_FLAG="--torch_compile=true"
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
    echo $COMMAND
    eval $COMMAND
    RETURN_CODE=$?

    if [[ $INTERNAL_CONTRIBUTOR_MODE == "true" ]]; then 
        /google-cloud-sdk/bin/gsutil -m cp -r ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${WORKLOAD}_${FRAMEWORK} ${EXPERIMENT_BUCKET}/${EXPERIMENT_NAME}/
        /google-cloud-sdk/bin/gsutil -m cp ${LOG_FILE} ${EXPERIMENT_BUCKET}/${EXPERIMENT_NAME}/${WORKLOAD}_${FRAMEWORK}/
    fi

fi

# Keep main process running in debug mode to avoid the container from stopping
if [[ ${KEEP_CONTAINER_ALIVE} == "true" ]]
then 
    while true
    do 
        sleep 1000
    done 
fi

echo "Exiting with $RETURN_CODE"
exit $RETURN_CODE
