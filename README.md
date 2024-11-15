# scalable-multimodal-similarity-search

This project is for the Part II Computer Science course at the University of Cambridge.

#### Running locally:
1. Run `pip install -r requirements.txt`
2. From `scalable-multimodal-similarity-search/` run `python3 -m src.index.multimodal_hnsw`

Docker:

0. (On Mac) Launch Docker Desktop or run `open -a Docker`
1. `docker build -t scalable-multimodal-similarity-search .`
2. `docker run -it --rm -v $(pwd):/scalable-multimodal-similarity-search scalable-multimodal-similarity-search`


#### Dataset generation:
- Used data_processing/dataset_preparation to download images and save image/metadata.
- Used data_processing/embedding_generation to generate text and image vectors, saving them, and identifying placeholder images.

#### Testing:

Run tests from `scalable-multimodal-similarity-search/` with `pytest -v tests`

- Tests regarding datasets (test/data_processing):
    - test_dataset: validates the raw dataset of images and metadata is consistent
    - test_vector_dataset: validates the generated vectors are consistent with the dataset