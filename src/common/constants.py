"""
Stores the constants used in the project.
"""

DATASET_ENTITY_COUNT = 20000

PREP_DATASET_PATH = f"/Users/yavuz/data/LAION-{DATASET_ENTITY_COUNT}/"
METADATA_PATH = PREP_DATASET_PATH + "metadata.parquet"
IMAGES_PATH = PREP_DATASET_PATH + "images"

IMAGE_VECTORS_PATH = PREP_DATASET_PATH + "vectors/image_vectors.npy"
TEXT_VECTORS_PATH = PREP_DATASET_PATH + "vectors/text_vectors.npy"

