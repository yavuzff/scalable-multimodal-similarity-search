#!/bin/bash
# this script compiles the code, running both the end-to-end python tests and cpp unit tests

set -e # exit on error

CWD=$(pwd)
case "$CWD" in
  */scalable-multimodal-similarity-search)
    # you are in a 'project/' directory. Proceeding...
    ;;
  *)
    echo "Error: run this script from the root of the repository."
    exit 1
    ;;
esac

# compile and set up bindings
cd cpp
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug -j 6 --target cppindex
python3 pybinding/setup.py install

# run end-to-end python tests
cd ..
pytest -v tests/

# run cpp tests
cd cpp
cmake --build cmake-build-debug -j 6 --target tests
./cmake-build-debug/tests
