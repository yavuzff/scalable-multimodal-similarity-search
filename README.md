# scalable-multimodal-similarity-search

This project is for the Part II Computer Science course at the University of Cambridge.

### Setup

You can set up environment locally within a venv, or set up a Docker container.

#### Local:
1. Run `pip install -r requirements.txt` in a virtual environment.

#### Docker:
0. (On Mac) Launch Docker Desktop or run `open -a Docker`
1. `docker build -t scalable-multimodal-similarity-search .`
2.  `docker run -it --rm -v $(pwd):/scalable-multimodal-similarity-search -v /Users/yavuz/data/:/Users/yavuz/data scalable-multimodal-similarity-search`

### Running

1. Run `python3 -m src.index.multimodal_hnsw` in `scalable-multimodal-similarity-search/`

### Additional Information

#### Dataset generation:
- Used data_processing/dataset_preparation to download images and save image/metadata.
- Used data_processing/embedding_generation to generate text and image vectors, saving them, and identifying placeholder images.

#### Testing:
To run the tests, `pytest -v tests` in `scalable-multimodal-similarity-search/` 

- Tests regarding datasets (test/data_processing):
    - test_dataset: validates the raw dataset of images and metadata is consistent
    - test_vector_dataset: validates the generated vectors are consistent with the dataset