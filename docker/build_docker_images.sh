#!/bin/bash
# Bash script to build and push dev docker images to artifact repo
# Usage:
#   bash build_docker_images.sh -b <git_branch>

# Make program exit with non-zero exit code if any command fails.
set -e

while getopts b: flag
do
    case "${flag}" in
        b) GIT_BRANCH=${OPTARG};;
    esac
done

# Artifact repostiory
ARTIFACT_REPO="europe-docker.pkg.dev/mlcommons-algoperf/algoperf-docker-repo"

if [[ -z ${GIT_BRANCH+x} ]]
then 
GIT_BRANCH='main' # Set default argument
fi 

for FRAMEWORK in "jax" "pytorch" "both"
do
    IMAGE_NAME="algoperf_${FRAMEWORK}_${GIT_BRANCH}"
    DOCKER_BUILD_COMMAND="docker build --no-cache -t $IMAGE_NAME . --build-arg framework=$FRAMEWORK --build-arg branch=$GIT_BRANCH"
    DOCKER_TAG_COMMAND="docker tag $IMAGE_NAME $ARTIFACT_REPO/$IMAGE_NAME"
    DOCKER_PUSH_COMMAND="docker push $ARTIFACT_REPO/$IMAGE_NAME"
    DOCKER_PULL_COMMAND="docker pull $ARTIFACT_REPO/$IMAGE_NAME"

    echo "On branch: ${GIT_BRANCH}"
    echo $DOCKER_BUILD_COMMAND
    eval $DOCKER_BUILD_COMMAND
    echo $DOCKER_TAG_COMMAND
    eval $DOCKER_TAG_COMMAND
    echo $DOCKER_PUSH_COMMAND
    eval $DOCKER_PUSH_COMMAND
    echo "To pull container run: "
    echo $DOCKER_PULL_COMMAND
done
