# scalable-multimodal-similarity-search

This project is being developed for the Part II Computer Science course at the University of Cambridge. 
It provides a multimodal similarity search system in C++ for efficient indexing and searching, with Python bindings exposing it as a Python module.


### Prerequisites

You can run the project using Docker (recommended) or set up the environment locally.

#### Using Docker
The project includes a Dockerfile to build an image with all required dependencies.

1. Start Docker Desktop (on macOS):
```
open -a Docker
```
2. Build the Docker image:
```
docker build -t scalable-multimodal-similarity-search . 
```
3. Run the container (optionally mounting data directories, and the repository directory using argument `-v $(pwd):/scalable-multimodal-similarity-search`):

```
docker run -it --rm \
    -v /Users/yavuz/data/:/Users/yavuz/data \
    scalable-multimodal-similarity-search
```

#### Local:

Follow these steps to set up the environment locally (involving venv, Cmake, conda, pybind):

1. Ensure Python 3.11 is installed, and create a virtual environment with the required libraries:
```bash
python3.11 -m venv ".venv"
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure the following is installed to compile and run C++ code:

   1. Install CMake version >3.29 (e.g., using Homebrew): 
   ```
   brew install cmake
   ```
   2. Setup pybind11 (using Conda):
      1. Ensure conda is installed, e.g. [miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install).
      2. Update with `conda-forge` channel:
        ```
        conda update -n base --all -c conda-forge -c defaults
        ```
      3. Install pybind11:
        ```
        conda install -c conda-forge pybind11
        ```

    
### Compile C++ code and setup Python bindings:
To use the C++ index in Python, compile the code and set up the Python bindings:

1. Navigate to the cpp/ directory:
```
cd cpp
```

2. Build the project using CMake:
```
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug -j 6 --target cppindex
```
3. Install the Python bindings:
```
python3 pybinding/setup.py install
```

### Using the C++ index through Python:

The C++ index can be imported and used within Python scripts.

1. Import the index:

```
from cppindex import ExactMultiIndex
```

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
```
python3 -m src.main
```

6. For memory profiling that is not just on the C++ index, use `filprofiler`. From the project root, run:
```
fil-profile run -m src.main
```

### Additional Information

This section contains information on how to develop the project, generate datasets, and run tests.

#### C++ development:
To develop and test the C++ code:

1. Build and run the C++ executable:
```
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug -j 6 --target main
./cmake-build-debug/main
```

2. Optionally, to develop with Python bindings:
    - Change C++ code. 
    - Run `sh clean-and-rebind-module.sh` to recompile the module. 
    - Run `python3 -m src.main` to test the changes.

#### Dataset generation:
- Used data_processing/dataset_preparation to download images and save image/metadata.
- Used data_processing/embedding_generation to generate text and image vectors, saving them, and identifying placeholder images.

#### Testing:

To run the Python tests: `pytest -v tests` in `scalable-multimodal-similarity-search/` 

- Tests regarding datasets (test/data_processing):
    - test_dataset: validates the raw dataset of images and metadata is consistent
    - test_vector_dataset: validates the generated vectors are consistent with the dataset
- Tests for the exact multi search C++ index are in `test_exact_multi_index_bindings.py`
