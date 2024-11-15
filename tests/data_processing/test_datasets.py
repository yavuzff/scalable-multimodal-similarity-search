"""
Check that the prepared dataset=(metadata parquet, images) is consistent.
"""
import pandas as pd
import numpy as np
import os
import json
from src.common.constants import *


def test_image_dataset():
    """
    Verify that the images at the image_path is consistent with the data frame.
    """
    image_path = IMAGES_PATH
    data = pd.read_parquet(METADATA_PATH)
    # iterate over the URLS of the data frame
    for i, row in data.iterrows():
        shard = str(i // 10000).zfill(5)
        index = str(i % 10000).zfill(4)

        # check if image exists
        image_file = f"{image_path}/{shard}/{shard}{index}.jpg"
        assert os.path.exists(image_file), f"Image {image_file} does not exist for this row: {index, row}"

        # check if json exists
        json_file = f"{image_path}/{shard}/{shard}{index}.json"
        assert os.path.exists(json_file), f"Json {json_file} does not exist for this row: {index, row}"

        # compare the data URL with the json URL
        assert row["URL"] == json.load(open(json_file))["url"], \
            f"URL does not match for this index, row: {index, row}.\nImage file: {image_file}\nJson file: {json_file}"


def test_vector_dataset():
    """
    Verify that the length of the vectors are consistent with the metadata.
    """
    data = pd.read_parquet(METADATA_PATH)
    print(f"Read {len(data)} entries from {METADATA_PATH}.")
    image_vectors = np.load(IMAGE_VECTORS_PATH)
    text_vectors = np.load(TEXT_VECTORS_PATH)
    print(f"Image vectors shape: {image_vectors.shape}, Text vectors shape: {text_vectors.shape}")
    assert len(data) == len(image_vectors) == len(text_vectors), "Lengths do not match."
