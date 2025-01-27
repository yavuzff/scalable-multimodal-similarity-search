"""
Find near-duplicate embeddings in the LAION dataset.
"""
import numpy as np
import argparse
import os
from tqdm import tqdm


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


def batch_compute_all_duplicates(normalised_embeddings, batch_size=1000):
    """
    Compute all duplicates in batches.
    """
    near_duplicate_ids = set()
    for i in tqdm(range(0, len(normalised_embeddings), batch_size)):
        batch_embeddings = normalised_embeddings[i:i + batch_size]
        rest_embeddings = normalised_embeddings[i + batch_size:]
        similarity_matrix = np.dot(rest_embeddings, batch_embeddings.T)
        duplicate_indices = np.where(similarity_matrix > 0.99)

        near_duplicate_ids.update(duplicate_indices[0] + i + batch_size)
        near_duplicate_ids.update(duplicate_indices[1] + i)

    return near_duplicate_ids


def compute_duplicate_set_from_window(normalised_embeddings, placeholder_images_path, window_size=10000, threshold=0.99):
    """
    Compute a set of near-duplicate embedding ids.
    """
    near_duplicates = set()
    for i in tqdm(range(0, len(normalised_embeddings))):
        start = i + 1
        end = min(i + window_size, len(normalised_embeddings))
        if start < end:
            scores = np.dot(normalised_embeddings[i], normalised_embeddings[start:end].T)

            near_duplicate_indices = np.where(scores > threshold)[0] + start
            if len(near_duplicate_indices) > 0:
                near_duplicates.update([i] + list(near_duplicate_indices))

        # Save checkpoint every 50k iterations
        if i % 50000 == 0:
            np.save(placeholder_images_path + "_checkpoint", np.array(list(near_duplicates)))

    return near_duplicates


def normalise_embeddings(embeddings):
    """
    Normalise embeddings.
    """
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def compute_similarity_matrix(normalised_embeddings):
    """Compute similarity matrix for embeddings.
    Note: this should be used for small datasets only."""
    return np.dot(normalised_embeddings, normalised_embeddings.T)


def find_near_duplicates_with_distance(similarity_matrix, threshold=0.99):
    """Find near-duplicate embeddings from the similarity matrix based on a similarity threshold."""
    near_duplicates_with_distance = [
        (i, j, similarity_matrix[i][j])
        for i in range(len(similarity_matrix))
        for j in range(i + 1, len(similarity_matrix))
        if similarity_matrix[i][j] > threshold
    ]
    return sorted(near_duplicates_with_distance, key=lambda x: x[2])


def save_near_duplicate_ids(near_duplicate_ids: set, path: str):
    """Save IDs of near-duplicate embeddings."""
    np.save(path, np.array(list(near_duplicate_ids)))


def main():
    args = parse_arguments()
    dataset_base_path = os.path.join(args.base_path, f"LAION-{args.dataset_entity_count}")
    vector_path = os.path.join(dataset_base_path, "vectors/")
    image_vector_path = os.path.join(vector_path, "image_vectors.npy")

    image_embeddings = np.load(image_vector_path)
    normalised_image_embeddings = normalise_embeddings(image_embeddings)

    near_duplicates = set()
    window_size = 50000
    placeholder_images_path = os.path.join(vector_path, "placeholder_images" + str(window_size) + "_window")

    # below performs the all pairwise comparisons - this leads to around 3% duplicates identifed.
    # around a half of these are not placeholder images. So we do not perform this step.
    # near_duplicates.update(batch_compute_all_duplicates(normalised_image_embeddings, batch_size=1000))

    # below checks for duplicates within a sliding window of size 10k. This identifies around 1.5% duplicates.
    # most of these are placeholder images.
    near_duplicates.update(compute_duplicate_set_from_window(normalised_image_embeddings, placeholder_images_path, window_size=window_size))

    save_near_duplicate_ids(near_duplicates, placeholder_images_path)


if __name__ == "__main__":
    main()
