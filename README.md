# scalable-multimodal-similarity-search

This project is for the Part II Computer Science course at the University of Cambridge.


### Prerequisites

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
2. `cmake -S . -B cmake-build-debug`
3. `cmake --build cmake-build-debug -j 6 --target cppindex`
4. `pip install ./pybinding`

      Note: IDE may not index the module correctly. This is not the case if you run `python3 pybinding/setup.py install` instead.

#### Using the C++ index through Python:

The C++ index can be imported and used in Python. In a Python file, you can:
1. Import the index:

    `from cppindex import ExactMultiIndex`


2. Initialise the index, e.g.:
```
index = ExactMultiIndex(
    num_modalities=2,
    dimensions=[128, 64],
    distance_metrics=["euclidean", "cosine"],  # optional: distance metrics for each modality. Default: "euclidean"
    weights=[0.3, 0.7]  # optional: Weights for each modality. Default: uniform weights.
)
```
3. Add items to the index:
```
# Example: adding 3 entities to an index with 2 modalities
modality_1_data = np.random.rand(3, 128)  # 128-dimensional data for modality 1
modality_2_data = np.random.rand(3, 64)   # 64-dimensional data for modality 2

index.add_entities([modality_1_data, modality_2_data])
```

4. Search the index:
```
query = [
    np.random.rand(128),  # modality 1 vector
    np.random.rand(64)    # modality 2 vector
]

k = 5
results = index.search(query, k) # can optionally provide query_weights
print("Indices of nearest neighbors:", results)
```


5. Example usage can be found in `src/main`. To run it, run the below command from the project root:

    `python3 -m src.main`


### Additional Information

This section contains information on how to develop the project, generate datasets, and run tests.

#### C++ development:
Run from `cpp/`:
1. `cmake -S . -B cmake-build-debug`
2. `cmake --build cmake-build-debug -j 6 --target main`
3. `./cmake-build-debug/main`

Alternatively, to develop with the Python bindings:
1. Change C++ code.
2. Run `sh clean-and-rebind-module.sh` to recompile the module.
3. Run `python3 -m src.main` to test the changes.

#### Dataset generation:
- Used data_processing/dataset_preparation to download images and save image/metadata.
- Used data_processing/embedding_generation to generate text and image vectors, saving them, and identifying placeholder images.

#### Testing:

To run the Python tests: `pytest -v tests` in `scalable-multimodal-similarity-search/` 

- Tests regarding datasets (test/data_processing):
    - test_dataset: validates the raw dataset of images and metadata is consistent
    - test_vector_dataset: validates the generated vectors are consistent with the dataset
- Tests for the exact multi search C++ library are in `test_exact_multi_index_bindings.py`
