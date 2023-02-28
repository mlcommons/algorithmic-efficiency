#!bin/bash

while getopts d:f: flag
do
    case "${flag}" in
        d) DATASET=${OPTARG};;
        f) FRAMEWORK=${OPTARG};;
    esac
done

ROOT_DATA_BUCKET="gs://mlcommons-data/"
ROOT_DATA_DIR="/data/"

if [ "${DATASET}" == "imagenet" ]
then 
    DATA_DIR="${ROOT_DATA_DIR}/${DATASET}/${FRAMEWORK}"
    DATA_BUCKET="${ROOT_DATA_BUCKET}/${DATASET}/${FRAMEWORK}"
else
    DATA_DIR="${ROOT_DATA_DIR}/${DATASET}"
    DATA_BUCKET="${ROOT_DATA_BUCKET}/${DATASET}/"
fi

# Copy data from MLCommons bucket if data does not downloaded yet
if [ ! -d /data/$DATASET/$FRAMEWORK ]
then
    ./google-cloud-sdk/bin/gsutil -m cp -r ${DATA_BUCKET}/* ${DATA_DIR}
fi 

while true
do 
    sleep 1000
done 