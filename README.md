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
docker build --platform linux/amd64 -t scalable-multimodal-similarity-search . 
```
3. Run the container (optionally mounting data directories, and the repository directory using argument `-v $(pwd):/scalable-multimodal-similarity-search`):

```
docker run -it --rm \
    -v /Users/yavuz/data/:/Users/yavuz/data \
    scalable-multimodal-similarity-search
```

e.g. to edit cpp source code in IDE:
```
docker run -it --rm \
    -v /Users/yavuz/data/:/Users/yavuz/data -v ./cpp/include/:/scalable-multimodal-similarity-search/cpp/include/ -v ./cpp/index/:/scalable-multimodal-similarity-search/cpp/index/ -v ./cpp/tests/:/scalable-multimodal-similarity-search/cpp/tests/ \
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
   2. Install catch2 (v3) for testing (e.g., using Homebrew):
   ```
   brew install catch2
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
cmake -S . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-debug -j 6 --target multivec_index
```
3. Install the Python bindings:
```
python3 pybinding/setup.py install
```

### Using the C++ index through Python:

The C++ index can be imported and used within Python scripts.

1. Import the index:

```
from multivec_index import ExactMultiVecIndex
```

2. Initialise the index, e.g.:
```
index = ExactMultiVecIndex(
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


### Running the visual demonstration of the multimodal similarity search framework:
1. Ensure you have a valid dataset prepared through src/dataset_processing.
2. Run the following command from the project root:
    ```
    python3 -m src.demo.visual_demo
    ```
3. Enter dataset path, embedding models, weights and metrics to build the index.
4. Search the index by selecting a query image and text, and k.

Note: For 2 modality dataset (text, image), use `visual_demo.py`. For 4 modality dataset (text, image, audio, video), use `visual_demo_4_modalities.py`.

### Additional Information

This section contains information on how to develop the project, generate datasets, and run tests.

#### C++ development:
To develop and test the C++ code:

1. Build and run the C++ executable:
```
cmake -S . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug -j 6 --target main
./cmake-build-debug/main
```

2. Optionally, to develop with Python bindings:
    - Change C++ code. 
    - Run `sh clean-and-rebind-module.sh` to recompile the module. 
    - Run `python3 -m src.main` to test the changes.

#### Dataset generation:
- Run the scripts in `src/dataset_processing` to generate the dataset: 
    - `data_download.py` to download the LAION dataset metadata and images.
    - `embedding_generation.py` to generate vectors for the texts and images.
    - `find_duplicates.py` to identify placeholder images in the dataset (which tend to be duplicated many times)
- You can also use the notebooks in `notebooks/`.

#### Testing:

To run the Python tests: `pytest -v tests` in `scalable-multimodal-similarity-search/` 

- Tests regarding datasets (test/data_processing):
    - test_dataset: validates the raw dataset of images and metadata is consistent
    - test_vector_dataset: validates the generated vectors are consistent with the dataset
- Tests for the exact multi vector search C++ index are in `test_exact_multivec_index_bindings.py`
- Tests for multi vector HNSW are in `test_multivec_hnsw_index.py`


To run the C++ tests:

1. Navigate to the cpp/ directory:
```
cd cpp
```

2. Build the CMake tests target:
```
cmake -S . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug -j 6 --target tests
```
3. Run the `tests` executable:
```
./cmake-build-debug/tests
```

**Using Valgrind in the Docker container:**
```
G_SLICE=always-malloc G_DEBUG=gc-friendly  valgrind -v --tool=memcheck --leak-check=full --num-callers=40 --log-file=valgrind.log $(which <program>) <arguments>
```

We can also use MSan and USan for memory and undefined behaviour checks - enable these in CMakeLists.txt.

**Profiling with `gprof`:**

1. Compile the code with -pg flag, set through `CMakeLists.txt`.
2. Run the executable.
3. Run `gprof` on the executable:
   ```
   gprof ./cmake-build-debug/performance gmon.out > analysis.txt
   ```