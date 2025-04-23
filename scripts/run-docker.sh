#!/bin/bash
# this script builds the docker container and runs it, mounting the data folder to the container
set -e # exit on error

CWD=$(pwd)

# check if we are inside scalable-multimodal-similarity-search, if not, exit
case "$CWD" in
  */scalable-multimodal-similarity-search)
    # you are in a 'project/' directory. Proceeding...
    ;;
  *)
    echo "Error: run this script from the root of the repository."
    exit 1
    ;;
esac

# start docker (on Mac)
open -a Docker

# build the docker container
docker build --platform linux/amd64 -t scalable-multimodal-similarity-search .

# run the docker container, mounting the data folder
docker run -it --rm \
    -v /Users/yavuz/data/:/Users/yavuz/data \
    scalable-multimodal-similarity-search
