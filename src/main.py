from cppindex import ExactMultiIndex
import numpy as np

from src.common.constants import *
from src.evaluation import IndexEvaluator

def load_dataset():
    """
    Load the dataset from the prepared paths.
    """
    image_vectors = np.load(IMAGE_VECTORS_PATH)
    text_vectors = np.load(TEXT_VECTORS_PATH)
    print(f"Loaded image vectors shape: {image_vectors.shape}, Text vectors shape: {text_vectors.shape}")
    return text_vectors, image_vectors


def main():
    # load dataset
    text_vectors_all, image_vectors_all = load_dataset()

    # define subset for indexing and querying
    NUM_INDEXED_ENTITIES = 12000
    NUM_QUERY_ENTITIES = 500
    index_text_vectors = text_vectors_all[:NUM_INDEXED_ENTITIES]
    index_image_vectors = image_vectors_all[:NUM_INDEXED_ENTITIES]
    query_text_vectors = text_vectors_all[NUM_INDEXED_ENTITIES:NUM_INDEXED_ENTITIES + NUM_QUERY_ENTITIES]
    query_image_vectors = image_vectors_all[NUM_INDEXED_ENTITIES:NUM_INDEXED_ENTITIES + NUM_QUERY_ENTITIES]

    # define and build index that we will evaluate
    MODALITIES = 2
    DIMENSIONS = [text_vectors_all.shape[1], image_vectors_all.shape[1]]
    DISTANCE_METRICS = ["euclidean", "euclidean"]
    my_index = ExactMultiIndex(MODALITIES, dimensions=DIMENSIONS, distance_metrics=DISTANCE_METRICS)

    # search parameters
    K = 10

    # evaluate the index
    index_evaluator = IndexEvaluator(my_index)

    # evaluate inserting to the index
    entities = [index_text_vectors, index_image_vectors]
    insertion_time = index_evaluator.evaluate_add_entities(entities)
    print(f"Insertion took {insertion_time:.3f} seconds.")

    # evaluate search performance
    queries = [query_text_vectors, query_image_vectors]
    search_times, recall_scores = index_evaluator.evaluate_search(queries, K)

    print(f"Average search time: {np.mean(search_times):.3f} seconds. Variance: {np.var(search_times):.3f}. Min: {np.min(search_times):.3f}. Max: {np.max(search_times):.3f}.")
    print(f"Average recall: {np.mean(recall_scores):.3f}. Variance: {np.var(recall_scores):.3f}. Min: {np.min(recall_scores):.3f}. Max: {np.max(recall_scores):.3f}.")


if __name__ == "__main__":
    main()