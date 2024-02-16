#!/bin/bash

# activate conda env, export DATA_DIR
source exp/data_setup/set_env.sh

python3 datasets/dataset_setup.py \
    --data_dir $DATA_DIR \
    --temp_dir $DATA_DIR/tmp \
    --librispeech \
    --interactive_deletion=False