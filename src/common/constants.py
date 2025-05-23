"""
Stores the constants used in the project.
"""

DATASET_ENTITY_COUNT = 1_900_000  # e.g. 20000, 150, 1_900_000

PREP_DATASET_PATH = f"/Users/yavuz/data/LAION-{DATASET_ENTITY_COUNT}/"
METADATA_PATH = PREP_DATASET_PATH + "metadata.parquet"
IMAGES_PATH = PREP_DATASET_PATH + "images"

IMAGE_VECTORS_PATH = PREP_DATASET_PATH + "vectors/image_vectors.npy"
IMAGE_VECTORS32_PATH = PREP_DATASET_PATH + "vectors/image_vectors32.npy"
TEXT_VECTORS_PATH = PREP_DATASET_PATH + "vectors/text_vectors.npy"


LARGE_ENTITY_DATASET_ENTITY_COUNT = 20000 #150, 20000
LARGE_ENTITY_BASE_PATH = f"/Users/yavuz/data/LAION-{LARGE_ENTITY_DATASET_ENTITY_COUNT}-4-modalities/"

LARGE_ENTITY_METADATA_PATH = LARGE_ENTITY_BASE_PATH + "metadata-4-modalities.parquet"
LARGE_ENTITY_VECTOR_PATH = LARGE_ENTITY_BASE_PATH + "vectors-4-modalities/"

LARGE_ENTITY_TEXT_VECTORS_PATH = LARGE_ENTITY_VECTOR_PATH + "text_vectors.npy"
LARGE_ENTITY_IMAGE_VECTORS_PATH = LARGE_ENTITY_VECTOR_PATH + "image_vectors.npy"
LARGE_ENTITY_AUDIO_VECTORS_PATH = LARGE_ENTITY_VECTOR_PATH + "audio_vectors.npy"
LARGE_ENTITY_VIDEO_VECTORS_PATH = LARGE_ENTITY_VECTOR_PATH + "video_vectors.npy"