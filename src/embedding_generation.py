"""
This script is used to generate and save vector embeddings for the text and image data prepared in dataset_preparation.ipynb. Update the parameters in the cell below. Additionally, the number of threads used to generate text/image embeddings can be amended in their corresponding function calls below. The last part of this notebook identifies placeholder images by computing pairwise similarities across every generated image embedding.
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd
import argparse
from embedding_generation.text_embeddings import SentenceTransformerEmbeddingGenerator
from embedding_generation.image_embeddings import HFImageEmbeddingGenerator
from common.logger import *

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
    return parser.parse_args()

def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path)

def load_metadata(metadata_path):
    """Load metadata from a Parquet file."""
    return pd.read_parquet(metadata_path)

def generate_text_embeddings(df, model, vector_path, num_vectors=None):
    """Generate and save text embeddings."""
    texts = list(df["TEXT"])
    if num_vectors is not None:
        texts = texts[:num_vectors]

    embedding_generator = SentenceTransformerEmbeddingGenerator(model)
    embeddings = embedding_generator.generate_text_embeddings(texts, normalize_embeddings=False, batch_size=128)

    text_vector_path = save_embeddings(embeddings, vector_path + "text_vectors")
    return text_vector_path

def generate_image_embeddings(image_paths, model, vector_path, num_vectors=None):
    """Generate and save image embeddings."""
    if num_vectors is not None:
        image_paths = image_paths[:num_vectors]

    embedding_generator = HFImageEmbeddingGenerator(model)
    embeddings = embedding_generator.batch_generate_image_embeddings(image_paths, normalize_embeddings=False, batch_size=128)

    image_vector_path = save_embeddings(embeddings, vector_path + "image_vectors")
    return image_vector_path

def save_embeddings(embeddings, path):
    """Save embeddings to a file and handle conflicts."""
    if os.path.exists(path + ".npy"):
        new_path = path + datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
        print(f"Path {path} already exists. Saving to {new_path}.npy instead.")
        path = new_path
    np.save(path, embeddings)
    return path

def compute_similarity_matrix(embeddings):
    """Compute similarity matrix for embeddings."""
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(normalized_embeddings, normalized_embeddings.T)

def find_near_duplicates(similarity_matrix, threshold=0.99):
    """Find near-duplicate embeddings based on a similarity threshold."""
    near_duplicates = [
        (i, j, similarity_matrix[i][j])
        for i in range(len(similarity_matrix))
        for j in range(i + 1, len(similarity_matrix))
        if similarity_matrix[i][j] > threshold
    ]
    return sorted(near_duplicates, key=lambda x: x[2])

def save_near_duplicate_ids(near_duplicates, path):
    """Save IDs of near-duplicate embeddings."""
    near_duplicate_ids = {i for i, j, _ in near_duplicates}.union(j for i, j, _ in near_duplicates)
    np.save(path, np.array(list(near_duplicate_ids)))

def main():
    args = parse_arguments()

    # Paths and constants
    dataset_base_path = os.path.join(args.base_path, f"LAION-{args.dataset_entity_count}")
    metadata_path = os.path.join(dataset_base_path, "metadata.parquet")
    images_path = os.path.join(dataset_base_path, "images/")
    vector_path = os.path.join(dataset_base_path, "vectors/")
    create_directory(vector_path)

    # Load metadata
    df = load_metadata(metadata_path)

    # Generate text embeddings
    text_model = "BAAI/bge-small-en-v1.5"
    text_vector_path = generate_text_embeddings(df, text_model, vector_path)

    # Generate image embeddings
    image_model = "google/vit-base-patch16-224-in21k"
    image_paths = glob.glob(os.path.join(images_path, "*/*.jpg"))
    image_paths.sort()
    image_vector_path = generate_image_embeddings(image_paths, image_model, vector_path)

    # Compute similarity matrix
    image_embeddings = np.load(image_vector_path + ".npy")
    similarity_matrix = compute_similarity_matrix(image_embeddings)

    # Find and save near-duplicate IDs
    near_duplicates = find_near_duplicates(similarity_matrix)
    save_near_duplicate_ids(near_duplicates, vector_path + "placeholder_images.npy")

if __name__ == '__main__':
    main()
