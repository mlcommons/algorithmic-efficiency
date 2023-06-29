# Bash script to build and push dev docker images to artifact repo
# Usage:
# bash build_docker_images.sh -b <git_branch>

while getopts b: flag
do
    case "${flag}" in
        b) GIT_BRANCH=${OPTARG};;
    esac
done

if [[ -z ${GIT_BRANCH+x} ]]
then 
GIT_BRANCH='main' # Set default argument
fi 

for FRAMEWORK in "jax" "pytorch" "both"
do
    IMAGE_NAME="algoperf_$FRAMEWORK:$GIT_BRANCH"
    DOCKER_BUILD_COMMAND="docker build --no-cache -t $IMAGE_NAME . --build-arg framework=$FRAMEWORK --build-arg branch=dockerfile_framework_arg"
    DOCKER_TAG_COMMAND="docker tag $IMAGE_NAME us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/$IMAGE_NAME"
    DOCKER_PUSH_COMMAND="docker push us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/$IMAGE_NAME"
    DOCKER_PULL_COMMAND="docker pull us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/$IMAGE_NAME"

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