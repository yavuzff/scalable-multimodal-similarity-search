"""
This script can be used to prepare a subset of the LAION dataset: extract valid images and save corresponding metadata.
The .parquet file used for this script (as well as the full dataset) can be found here: https://www.kaggle.com/datasets/romainbeaumont/laion400m

We use a subset of a single .parquet file of the LAION dataset (1.8GB):
[12933524 rows x 8 columns]
metadata:
SAMPLE_ID | URL | TEXT | LICENSE | NSFW | similarity | WIDTH | HEIGHT
"""

import os
import pandas as pd
import glob
import datetime
import shutil
import argparse
from img2dataset import download


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process the LAION dataset.")
    parser.add_argument(
        '--target_entity_count',
        type=int,
        default=100,
        help="Number of entities to process before filtering and downloading images (default: 100)"
    )
    parser.add_argument(
        '--full_laion_path',
        type=str,
        default="data/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
        help="Path to the full LAION dataset."
    )
    parser.add_argument(
        '--prep_dataset_base_path',
        type=str,
        default="data/",
        help="Base path for preparing the dataset."
    )
    return parser.parse_args()


def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path)
    else:
        print(f"Warning: {path} already exists!")


def read_safe_data(path, count):
    """Return non-NSFW entries from the full LAION dataset."""
    print(f"Reading {count} items from full LAION dataset...")
    df = pd.read_parquet(path)[:count]
    nsfw_removed_data = df[df["NSFW"] == "UNLIKELY"]
    print(f"Size after removing NSFW: {len(nsfw_removed_data)}")
    clean_url_data = nsfw_removed_data[~nsfw_removed_data["URL"].str.contains(",")]
    print(f"Size after removing URLs with commas: {len(clean_url_data)}")
    return clean_url_data


def write_urls(data, path):
    """Write the URLs found in the dataframe to a file."""
    with open(path, "w+") as f:
        for url in data["URL"]:
            f.write(url + "\n")
    print(f"Finished writing {len(data)} URLs to {path}")


def download_images(url_path, images_path):
    """Download images from a text file containing URLs."""
    if os.path.exists(images_path):
        print(
            f"Warning: {images_path} exists - renaming to {images_path}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}!")
        os.rename(images_path, images_path + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    download(
        processes_count=8,
        thread_count=32,
        url_list=url_path,
        image_size=256,
        output_folder=images_path,
    )


def get_valid_file_ids(path):
    """Return the IDs of valid image files in a directory."""
    files = glob.glob(os.path.join(path, "*/*.jpg"))
    print(f"Found {len(files)} files.")
    ids = [int(os.path.basename(file).split(".")[0]) for file in files]
    ids.sort()
    return ids


def move_files(images_path):
    """Rename and move files across shards for continuous indexing."""
    files = glob.glob(os.path.join(images_path, "*/*.jpg"))
    files.sort()
    for i, file_path in enumerate(files):
        shard = str(i // 10000).zfill(5)
        index = str(i % 10000).zfill(4)
        new_image_path = os.path.join(images_path, shard, f"{shard}{index}.jpg")
        new_json_path = new_image_path.replace(".jpg", ".json")
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        shutil.move(file_path, new_image_path)
        shutil.move(file_path.replace(".jpg", ".json"), new_json_path)


def process_dataset(args):
    """Main dataset processing workflow."""
    prep_dataset_base_path = os.path.join(args.prep_dataset_base_path, f"LAION-{args.target_entity_count}")
    images_path = os.path.join(prep_dataset_base_path, "images")
    urls_path = os.path.join(prep_dataset_base_path, "urls.txt")
    succeeded_urls_path = os.path.join(prep_dataset_base_path, "succeeded-urls.txt")
    metadata_path = os.path.join(prep_dataset_base_path, "metadata.parquet")

    create_directory(prep_dataset_base_path)

    data = read_safe_data(args.full_laion_path, args.target_entity_count)
    write_urls(data, urls_path)
    download_images(urls_path, images_path)

    valid_ids = get_valid_file_ids(images_path)
    data_with_images = data.iloc[valid_ids].reset_index(drop=True)
    write_urls(data_with_images, succeeded_urls_path)
    data_with_images.to_parquet(metadata_path)

    move_files(images_path)


def main():
    args = parse_arguments()
    process_dataset(args)


if __name__ == '__main__':
    main()
