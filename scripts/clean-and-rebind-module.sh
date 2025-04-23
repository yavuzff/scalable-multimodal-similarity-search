#!/bin/bash
set -e # exit on error
# This script is used to remove the previously bound module (.so file) in .venv, and then recompile and rebind it.
# This is needed because when the C library code is updated and binded,
# the imported C++ module from does not reflect the changes as it is cached.

CWD=$(pwd)

# check if we are inside scalable-multimodal-similarity-search, if not, exit
if [[ "$CWD" != *"scalable-multimodal-similarity-search" ]]; then
  echo "Error: run this script from the root of the repository."
  exit 1
fi

# delete where the module is stored so that python has to reimport it
rm -rf ./.venv/lib/python3.11/site-packages/multivec_cpp_index-0.0.1-py3.11-macosx-14-x86_64.egg/

cd cpp || exit
python3 pybinding/setup.py install

cd ..
