from src.common.constants import *
import numpy as np


def load_dataset():
    """
    Load the dataset from the prepared paths.
    """
    text_vectors = np.load(TEXT_VECTORS_PATH)
    image_vectors = np.load(IMAGE_VECTORS_PATH)

    # check that the type of the stored vectors is float32
    if text_vectors.dtype != np.float32:
        # raise exception
        raise ValueError(f"Text vectors are not float32. They are {text_vectors.dtype}.")

    if image_vectors.dtype != np.float32:
        # raise exception
        # cast to float 32 and raise exception
        # image_vectors = image_vectors.astype(np.float32)
        # np.save(IMAGE_VECTORS32_PATH, image_vectors)
        raise ValueError(f"Image vectors are not float32. They are {image_vectors.dtype}.")

    print(
        f"Loaded 32-bit vectors. Text vectors shape: {text_vectors.shape}. Image vectors shape: {image_vectors.shape}.")
    return text_vectors, image_vectors
