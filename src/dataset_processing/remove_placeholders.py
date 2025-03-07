"""
Remove placeholder images from the dataset. Run this after you run find_duplicates.py, which computes duplicates/placeholder images.

Sample usage:
python3 -m src.dataset_processing.remove_placeholders --dataset_entity_count 150 --base_path /Users/yavuz/data
"""
import os
import argparse
import numpy as np
import pandas as pd

from src.dataset_processing.data_download import move_files


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Move duplicates identified to subfolders, and rename remaining entries.")
    parser.add_argument(
        '--dataset_entity_count',
        type=int,
        default=100,
        help="Number of entities to process before filtering and downloading images (default: 100)"
    )
    parser.add_argument(
        '--base_path',
        type=str,
        default="data/",
        help="Base path for dataset files."
    )
    return parser.parse_args()


def save_previous_state(metadata_path, text_vector_path, image_vector_path):
    """Save the previous state of the dataset."""
    # save metadata.parquet
    save_metadata_path = "/".join(metadata_path.split("/")[:-1]) + "/metadata_with_placeholders.parquet"
    save_text_vector_path = "/".join(text_vector_path.split("/")[:-1]) + "/text_vectors_with_placeholders.npy"
    save_image_vector_path = "/".join(image_vector_path.split("/")[:-1]) + "/image_vectors_with_placeholders.npy"

    # check if we have already removed placeholders
    if os.path.exists(save_metadata_path):
        raise ValueError(f"Error: this dataset has already had its placeholders removed: {save_metadata_path} exists.")
    if os.path.exists(save_text_vector_path):
        raise ValueError(f"Error: this dataset has already had its placeholders removed: {save_text_vector_path} exists.")
    if os.path.exists(save_image_vector_path):
        raise ValueError(f"Error: this dataset has already had its placeholders removed: {save_image_vector_path} exists.")

    os.rename(metadata_path, save_metadata_path)
    os.rename(text_vector_path, save_text_vector_path)
    os.rename(image_vector_path, save_image_vector_path)

    return save_metadata_path, save_text_vector_path, save_image_vector_path


def update_vectors(placeholder_ids, orig_text_vector_path, orig_image_vector_path,  text_vector_path, image_vector_path, ):
    """Update the vectors folder to remove placeholder images."""

    # load the vectors and remove the placeholder ids
    text_vectors = np.load(orig_text_vector_path)
    image_vectors = np.load(orig_image_vector_path)

    text_vectors = np.delete(text_vectors, placeholder_ids, axis=0)
    image_vectors = np.delete(image_vectors, placeholder_ids, axis=0)

    # save the updated vectors
    np.save(text_vector_path, text_vectors)
    np.save(image_vector_path, image_vectors)
    print(f"Removed placeholders from vectors and saved to {text_vector_path} and {image_vector_path}")


def update_metadata(placeholder_ids, orig_metadata_path, metadata_path):
    """Update the metadata file to remove placeholder images."""
    metadata = pd.read_parquet(orig_metadata_path)

    # remove the placeholder ids
    metadata = metadata.drop(placeholder_ids)
    metadata = metadata.reset_index(drop=True)

    # save the updated metadata
    metadata.to_parquet(metadata_path)
    print(f"Removed placeholders from metadata and saved to {metadata_path}")


def move_placeholders_to_subfolder(placeholder_ids, images_path):
    """Move placeholder images to a subfolder."""
    for i in placeholder_ids:
        # get name of the file for the placeholder image
        shard = str(i // 10000).zfill(5)
        index = str(i % 10000).zfill(4)
        image_path = os.path.join(images_path, shard, f"{shard}{index}.jpg")
        json_path = image_path.replace(".jpg", ".json")

        # assert the file exists
        assert os.path.exists(image_path), f"Placeholder image {image_path} does not exist."
        assert os.path.exists(json_path), f"Placeholder image {json_path} does not exist."

        # move to placeholder directory
        placeholder_directory = os.path.join(images_path, shard, "placeholder_images")
        os.makedirs(placeholder_directory, exist_ok=True)

        new_image_path = os.path.join(placeholder_directory, f"{shard}{index}.jpg")
        new_json_path = new_image_path.replace(".jpg", ".json")

        # assert that the new image path does not exist
        assert not os.path.exists(new_image_path), f"Placeholder image {new_image_path} already exists."
        assert not os.path.exists(new_json_path), f"Placeholder image {new_json_path} already exists."

        os.rename(image_path, new_image_path)
        os.rename(json_path, new_json_path)
    print("Moved placeholder images to placeholder_images/ subfolders")


def main():
    args = parse_arguments()
    dataset_base_path = os.path.join(args.base_path, f"LAION-{args.dataset_entity_count}")
    images_path = os.path.join(dataset_base_path, "images/")
    vector_folder_path = os.path.join(dataset_base_path, "vectors/")
    placeholder_images_path = os.path.join(vector_folder_path, "placeholder_images.npy")

    metadata_path = os.path.join(dataset_base_path, "metadata.parquet")
    text_vector_path = os.path.join(dataset_base_path, "vectors/text_vectors.npy")
    image_vector_path = os.path.join(dataset_base_path, "vectors/image_vectors.npy")

    # first load the placeholder ids - computed using find_duplicates.py
    placeholder_ids = np.load(placeholder_images_path)

    if len(placeholder_ids) == 0:
        raise ValueError(f"Error: no placeholders identified in {placeholder_images_path}")

    # check and save previous state
    save_metadata_path, save_text_vector_path, save_image_vector_path =\
        save_previous_state(metadata_path, text_vector_path, image_vector_path)

    # update vectors
    update_vectors(placeholder_ids, save_text_vector_path, save_image_vector_path, text_vector_path, image_vector_path)

    # update metadata
    update_metadata(placeholder_ids, save_metadata_path, metadata_path)

    # check and move placeholder images to subfolders within images
    move_placeholders_to_subfolder(placeholder_ids, images_path)
    # assert placeholder images file exists now
    assert os.path.exists(placeholder_images_path), f"Placeholder images file {placeholder_images_path} does not exist."
    # move every other image file so they are numbered sequentially
    print("Renaming image files to be sequential...")
    move_files(images_path)


if __name__ == "__main__":
    main()
