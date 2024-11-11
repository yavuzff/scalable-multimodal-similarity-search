# scalable-multimodal-similarity-search

This project is for the Part II Computer Science course at the University of Cambridge.

1. Run `export PYTHONPATH="${PYTHONPATH}:/path/to/scalable-multimodal-similarity-search"`

Dataset generation:
- Used data_processing/dataset_preparation to download images and save image/metadata.
- Used data_processing/embedding_generation to generate text and image vectors, saving them, and identifying placeholder images.

Testing:
- Tests regarding datasets (test/data_processing):
    - test_dataset: validates the raw dataset of images and metadata is consistent
    - test_vector_dataset: validates the generated vectors are consistent with the dataset