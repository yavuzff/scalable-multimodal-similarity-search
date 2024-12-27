#!/bin/bash
# this script builds the docker container and runs it, mounting the data folder to the container

CWD=$(pwd)

if [[ "$CWD" != *"scalable-multimodal-similarity-search" ]]; then
  echo "Error: run this script from the root of the repository."
  exit 1
fi


cd cpp
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug -j 6 --target cppindex
python3 pybinding/setup.py install

cd ..
pytest -v tests/
