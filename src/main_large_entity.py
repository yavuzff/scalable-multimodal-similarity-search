from multivec_index import MultiVecHNSW
import numpy as np
import random
import time

from src.load_dataset import load_dataset
from src.evaluation import IndexEvaluator, compute_exact_results, evaluate_index_construction, evaluate_index_search
from src.evaluation import evaluate_hnsw_rerank_construction, evaluate_hnsw_rerank_search
from src.evaluation_params import Params, MultiVecHNSWConstructionParams, MultiVecHNSWSearchParams



LARGE_ENTITY_DATASET_ENTITY_COUNT = 150
LARGE_ENTITY_BASE_PATH = f"/Users/yavuz/data/LAION-{LARGE_ENTITY_DATASET_ENTITY_COUNT}-4-modalities/"

LARGE_ENTITY_METADATA_PATH = LARGE_ENTITY_BASE_PATH + "metadata-4-modalities.parquet"
LARGE_ENTITY_VECTOR_PATH = LARGE_ENTITY_BASE_PATH + "vectors-4-modalities/"

LARGE_ENTITY_TEXT_VECTORS_PATH = LARGE_ENTITY_VECTOR_PATH + "text_vectors.npy"
LARGE_ENTITY_IMAGE_VECTORS_PATH = LARGE_ENTITY_VECTOR_PATH + "image_vectors.npy"
LARGE_ENTITY_AUDIO_VECTORS_PATH = LARGE_ENTITY_VECTOR_PATH + "audio_vectors.npy"
LARGE_ENTITY_VIDEO_VECTORS_PATH = LARGE_ENTITY_VECTOR_PATH + "video_vectors.npy"


def get_index_and_query_vectors(all_vectors, num_indexed_entities, num_query_entities):
    """
    Given all vectors, returns the index and query vectors.
    Index vectors are the first num_indexed_entities vectors.
    Query vectors are the last num_query_entities vectors.
    """
    index_vectors = all_vectors[:num_indexed_entities]
    query_vectors = all_vectors[-num_query_entities:]
    return index_vectors, query_vectors


def main():
    # load dataset
    text_vectors_all = np.load(LARGE_ENTITY_TEXT_VECTORS_PATH)
    image_vectors_all = np.load(LARGE_ENTITY_IMAGE_VECTORS_PATH)
    audio_vectors_all = np.load(LARGE_ENTITY_AUDIO_VECTORS_PATH)
    video_vectors_all = np.load(LARGE_ENTITY_VIDEO_VECTORS_PATH)

    all_entities = [text_vectors_all, image_vectors_all, audio_vectors_all, video_vectors_all]
    map = {"text": 0, "image": 1, "audio": 2, "video": 3}

    # define modalities to index
    modalities = ["text", "image", "audio", "video"]

    # define subset for indexing and querying
    NUM_INDEXED_ENTITIES = 75
    NUM_QUERY_ENTITIES = 15

    # get index and query vectors
    index_and_query_vectors = [get_index_and_query_vectors(all_entities[map[modality]], NUM_INDEXED_ENTITIES, NUM_QUERY_ENTITIES) for modality in modalities]

    # define and build index that we will evaluate
    MODALITIES = 4
    DIMENSIONS = [all_entities[map[modality]].shape[1] for modality in modalities]
    DISTANCE_METRICS = ["cosine", "cosine", "cosine", "cosine"]
    WEIGHTS = [1 for _ in modalities]
    my_index = MultiVecHNSW(MODALITIES, dimensions=DIMENSIONS, distance_metrics=DISTANCE_METRICS, weights=WEIGHTS,
                         target_degree=32, max_degree=32, ef_construction=200, seed=1)
    my_index.set_ef_search(50)
    # search parameters
    k = 50

    # evaluate the index
    index_evaluator = IndexEvaluator(my_index)

    # evaluate inserting to the index
    index_entities = [index_and_query_vectors[map[modality]][0] for modality in modalities]
    insertion_time, memory_consumption = index_evaluator.evaluate_add_entities(index_entities)
    print(f"Insertion Time: {insertion_time:.3f} seconds.")
    print(f"Insertion Memory: {memory_consumption / 1024 / 1024} MiB.")

    # evaluate search performance
    queries = [index_and_query_vectors[map[modality]][1] for modality in modalities]
    search_times, recall_scores, memory_consumption = index_evaluator.evaluate_search(queries, k)

    print(
        f"Average search time: {np.mean(search_times)*1000:.3f}ms. Variance: {np.var(search_times)*1000*1000:.3f}ms^2. Min: {np.min(search_times)*1000:.3f}ms. Max: {np.max(search_times)*1000:.3f}ms.")
    print(
        f"Average recall: {np.mean(recall_scores):.3f}. Variance: {np.var(recall_scores):.3f}. Min: {np.min(recall_scores):.3f}. Max: {np.max(recall_scores):.3f}.")
    print(
        f"Average memory consumption: {np.mean(memory_consumption):.3f} bytes. Variance: {np.var(memory_consumption):.3f}. Min: {np.min(memory_consumption):.3f}. Max: {np.max(memory_consumption):.3f}.")


if __name__ == "__main__":
    main()
