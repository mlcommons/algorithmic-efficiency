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

cd algorithmic-efficiency
git pull

chmod a+x docker/scripts/startup.sh
docker/scripts/startup.sh -f $FRAMEWORK -s $SUBMISSION_PATH -t $TUNING_SEARCH_SPACE -e $EXPERIMENT_NAME -w $WORKLOAD
