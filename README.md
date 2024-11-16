# scalable-multimodal-similarity-search

This project is for the Part II Computer Science course at the University of Cambridge.

### Setup

You can set up environment locally (involving venv, Cmake, conda, pybind), or run the Docker container.

#### Local:
1. Run `pip install -r requirements.txt` in a virtual environment.

In order to run C++ code:

1. Ensure CMake is installed, e.g. Homebrew: `brew install cmake`
2. Setup pybind11:
    - Ensure conda is installed, e.g. [miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install).
    - Update with `conda-forge` channel: `conda update -n base --all -c conda-forge -c defaults`
    - Install pybind11: `conda install -c conda-forge pybind11`

#### Docker:
Alternatively, you can run the project in a Docker container.

0. (On Mac) Launch Docker Desktop or run `open -a Docker`
1. `docker build -t scalable-multimodal-similarity-search .`
2.  `docker run -it --rm -v $(pwd):/scalable-multimodal-similarity-search -v /Users/yavuz/data/:/Users/yavuz/data scalable-multimodal-similarity-search`

### Running

#### Compile C++ code and setup Python bindings:
1. `cd cpp`
2. `pip install ./pybinding`. Note: IDE may not index the module correctly. This is not the case if you run `python3 pybinding/setup.py install` instead.

#### Run sample Python code:
1. From the project root, run `python3 -m src.main`


### Additional Information

#### C++ development:
Run cmake manually from `cpp/`:
1. `cmake -S . -B cmake-build-debug`
2. `cmake --build cmake-build-debug -j 6 --target cppindex`


#### Dataset generation:
- Used data_processing/dataset_preparation to download images and save image/metadata.
- Used data_processing/embedding_generation to generate text and image vectors, saving them, and identifying placeholder images.

#### Testing:

To run the Python tests: `pytest -v tests` in `scalable-multimodal-similarity-search/` 

- Tests regarding datasets (test/data_processing):
    - test_dataset: validates the raw dataset of images and metadata is consistent
    - test_vector_dataset: validates the generated vectors are consistent with the dataset