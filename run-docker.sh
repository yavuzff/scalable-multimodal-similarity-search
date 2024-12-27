#!/bin/bash
# this script builds the docker container and runs it, mounting the data folder to the container

CWD=$(pwd)

# check if we are inside scalable-multimodal-similarity-search, if not, exit
if [[ "$CWD" == *"cpp" ]]; then
  cd ..
fi

if [[ "$CWD" != *"scalable-multimodal-similarity-search" ]]; then
  echo "Error: run this script from the root of the repository."
  exit 1
fi

open -a Docker

docker build --platform linux/amd64 -t scalable-multimodal-similarity-search .

docker run -it --rm \
    -v /Users/yavuz/data/:/Users/yavuz/data \
    scalable-multimodal-similarity-search


