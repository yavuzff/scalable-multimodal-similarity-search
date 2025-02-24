from src.common.constants import *
import numpy as np
from PIL import Image

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

def load_vectors_from_dataset_base_path(path:str):
    text_vectors = np.load(path + "/vectors/text_vectors.npy")
    if text_vectors.dtype != np.float32:
        raise ValueError(f"Text vectors are not float32. They are {text_vectors.dtype}.")

    image_vectors = np.load(path + "/vectors/image_vectors.npy")
    if image_vectors.dtype != np.float32:
        raise ValueError(f"Image vectors are not float32. They are {image_vectors.dtype}.")

    return text_vectors, image_vectors

def load_image(vector_id: int, images_path: str):
    """
    Given a vector id and base images path (IMAGES_PATH), returns the image.
    """
    shard = str(vector_id // 10000).zfill(5)
    index = str(vector_id % 10000).zfill(4)
    image_path = f"{images_path}/{shard}/{shard}{index}.jpg"
    return Image.open(image_path)
