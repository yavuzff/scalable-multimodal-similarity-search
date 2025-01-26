"""
This script is used to generate and save vector embeddings for the text and image data downloaded using data_download.py.
The number of threads used to generate text/image embeddings can be amended in their corresponding function calls below.
"""

import glob
import numpy as np
import pandas as pd
import argparse
from src.embedding_generators.text_embeddings import SentenceTransformerEmbeddingGenerator
from src.embedding_generators.image_embeddings import HFImageEmbeddingGenerator
from src.common.logger import *


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process the LAION dataset.")
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
    parser.add_argument(
        '--num_vectors',
        type=int,
        default=None,
        help="Number of vectors to process from the total (default is None which means all)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help="Batch size for processing embeddings"
    )
    return parser.parse_args()


def create_directory(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_metadata(metadata_path):
    """Load metadata from a Parquet file."""
    return pd.read_parquet(metadata_path)


def generate_text_embeddings(df, model, vector_path, num_vectors=None, batch_size=128):
    """Generate and save text embeddings."""
    texts = list(df["TEXT"])
    if num_vectors is not None:
        texts = texts[:num_vectors]

    embedding_generator = SentenceTransformerEmbeddingGenerator(model)
    embeddings = embedding_generator.generate_text_embeddings(texts, normalize_embeddings=False, batch_size=batch_size)

    save_path = os.path.join(vector_path, str(num_vectors), "text_vectors.npy")
    print(f"Saving text embeddings to {save_path}")
    text_vector_path = save_embeddings(embeddings, save_path)
    return text_vector_path


def generate_image_embeddings(image_paths, model, vector_path, num_vectors=None, batch_size=128):
    """Generate and save image embeddings."""
    if num_vectors is not None:
        image_paths = image_paths[:num_vectors]

    embedding_generator = HFImageEmbeddingGenerator(model, batch_size=batch_size)
    embeddings = embedding_generator.batch_generate_image_embeddings(image_paths, normalize_embeddings=False)

    save_path = os.path.join(vector_path, str(num_vectors), "image_vectors.npy")
    print(f"Saving image embeddings to {save_path}")
    image_vector_path = save_embeddings(embeddings, save_path)
    return image_vector_path


def save_embeddings(embeddings, path):
    """Save embeddings to a file and handle conflicts."""
    if os.path.exists(path):
        new_path = path.replace(".npy", "") + datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S") + ".npy"
        print(f"Path {path} already exists. Saving to {new_path} instead.")
        path = new_path

    create_directory(os.path.dirname(path))
    np.save(path, embeddings)
    return path


def main():
    args = parse_arguments()

    # Paths and constants
    if args.num_vectors is None:
        args.num_vectors = args.dataset_entity_count

    batch_size = args.batch_size

    dataset_base_path = os.path.join(args.base_path, f"LAION-{args.dataset_entity_count}")
    metadata_path = os.path.join(dataset_base_path, "metadata.parquet")
    images_path = os.path.join(dataset_base_path, "images/")
    vector_path = os.path.join(dataset_base_path, "vectors/")
    create_directory(vector_path)

    # Load metadata
    df = load_metadata(metadata_path)

    # Generate text embeddings
    text_model = "BAAI/bge-small-en-v1.5"
    text_vector_path = generate_text_embeddings(df, text_model, vector_path, num_vectors=args.num_vectors,
                                                batch_size=batch_size)
    print("Text embeddings saved to", text_vector_path)

    # Generate image embeddings
    image_model = "google/vit-base-patch16-224-in21k"
    image_paths = glob.glob(os.path.join(images_path, "*/*.jpg"))
    image_paths.sort()
    image_vector_path = generate_image_embeddings(image_paths, image_model, vector_path, num_vectors=args.num_vectors,
                                                  batch_size=batch_size)
    print("Image embeddings saved to", image_vector_path)


if __name__ == '__main__':
    main()
