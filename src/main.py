from multimodal_index import MultiHNSW
import numpy as np

from src.common.constants import *
from src.evaluation import IndexEvaluator

def load_dataset():
    """
    Load the dataset from the prepared paths.
    """
    text_vectors = np.load(TEXT_VECTORS_PATH)
    image_vectors = np.load(IMAGE_VECTORS32_PATH)

    #check type stored, should be float32
    if text_vectors.dtype != np.float32:
        # raise exception
        raise ValueError(f"Text vectors are not float32. They are {text_vectors.dtype}.")

    if image_vectors.dtype != np.float32:
        # raise exception
        # cast to float 32 and raise exception
        # image_vectors = image_vectors.astype(np.float32)
        # np.save(IMAGE_VECTORS32_PATH, image_vectors)
        raise ValueError(f"Image vectors are not float32. They are {image_vectors.dtype}.")

    print(f"Loaded 32-bit vectors. Text vectors shape: {text_vectors.shape}. Image vectors shape: {image_vectors.shape}.")
    return text_vectors, image_vectors


def main():
    # load dataset
    text_vectors_all, image_vectors_all = load_dataset()

    # define subset for indexing and querying
    NUM_INDEXED_ENTITIES = 1000
    NUM_QUERY_ENTITIES = 100
    index_text_vectors = text_vectors_all[:NUM_INDEXED_ENTITIES]
    index_image_vectors = image_vectors_all[:NUM_INDEXED_ENTITIES]
    query_text_vectors = text_vectors_all[-NUM_QUERY_ENTITIES:]
    query_image_vectors = image_vectors_all[-NUM_QUERY_ENTITIES:]

    # define and build index that we will evaluate
    MODALITIES = 2
    DIMENSIONS = [text_vectors_all.shape[1], image_vectors_all.shape[1]]
    DISTANCE_METRICS = ["manhattan", "manhattan"]
    my_index = MultiHNSW(MODALITIES, dimensions=DIMENSIONS, distance_metrics=DISTANCE_METRICS)

    # search parameters
    K = 10

    # evaluate the index
    index_evaluator = IndexEvaluator(my_index)

    # evaluate inserting to the index
    entities = [index_text_vectors, index_image_vectors]
    print(f"Inserting {NUM_INDEXED_ENTITIES} entities to the index.")
    insertion_time, memory_consumption = index_evaluator.evaluate_add_entities(entities)
    print(f"Insertion Time: {insertion_time:.3f} seconds.")
    print(f"Insertion Memory: {memory_consumption/1024/1024} MiB.")

    # evaluate search performance
    queries = [query_text_vectors, query_image_vectors]
    search_times, recall_scores, memory_consumption = index_evaluator.evaluate_search(queries, K)

    print(f"Average search time: {np.mean(search_times):.3f} seconds. Variance: {np.var(search_times):.3f}. Min: {np.min(search_times):.3f}. Max: {np.max(search_times):.3f}.")
    print(f"Average recall: {np.mean(recall_scores):.3f}. Variance: {np.var(recall_scores):.3f}. Min: {np.min(recall_scores):.3f}. Max: {np.max(recall_scores):.3f}.")
    print(f"Average memory consumption: {np.mean(memory_consumption):.3f} bytes. Variance: {np.var(memory_consumption):.3f}. Min: {np.min(memory_consumption):.3f}. Max: {np.max(memory_consumption):.3f}.")


if __name__ == "__main__":
    main()
