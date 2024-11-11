"""
Check that the prepared dataset=(metadata parquet, images) is consistent.
"""
import pandas as pd
import os
import json
from src.common.constants import *


def read_saved_data(path: str) -> pd.DataFrame:
    """
    Return a dataset saved previously.
    """
    df = pd.read_parquet(path)
    print(f"Read {len(df)} entries from {path}.")
    return df


def verify_images(image_path: str, data: pd.DataFrame) -> bool:
    """
    Verify that the images at the image_path is consistent with the data frame.
    """
    # iterate over the URLS of the data frame
    for i, row in data.iterrows():
        shard = str(i // 10000).zfill(5)
        index = str(i % 10000).zfill(4)

        # check if image exists
        image_file = f"{image_path}/{shard}/{shard}{index}.jpg"
        if not os.path.exists(image_file):
            print(f"Image {image_file} does not exist for this row: {index, row}")
            return False

        # check if json exists
        json_file = f"{image_path}/{shard}/{shard}{index}.json"
        if not os.path.exists(json_file):
            print(f"Json {json_file} does not exist for this row: {index, row}")
            return False

        # compare the data URL with the json URL
        if row["URL"] != json.load(open(json_file))["url"]:
            print(f"Error: URL does not match for this index, row: {index, row}")
            print("Image file: ", image_file)
            print("Json file: ", image_file)
            return False

    print("All images and json files are verified.")
    return True


if __name__ == "__main__":
    # read metadata
    data = read_saved_data(METADATA_PATH)

    # verify that images match with it
    verify_images(IMAGES_PATH, data)
