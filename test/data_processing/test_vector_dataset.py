import pandas as pd
import numpy as np
from src.common.constants import *


def verify_vectors():
    """
    Verify that the length of the vectors are consistent with the metadata.
    """
    data = pd.read_parquet(METADATA_PATH)
    print(f"Read {len(data)} entries from {METADATA_PATH}.")
    image_vectors = np.load(IMAGE_VECTORS_PATH)
    text_vectors = np.load(TEXT_VECTORS_PATH)
    print(f"Image vectors shape: {image_vectors.shape}, Text vectors shape: {text_vectors.shape}")
    assert len(data) == len(image_vectors) == len(text_vectors), "Lengths do not match."


if __name__ == "__main__":
    verify_vectors()
