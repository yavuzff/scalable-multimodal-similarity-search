from multimodal_index import MultiHNSW
import numpy as np
import random

from src.common.constants import *
from src.evaluation import IndexEvaluator, compute_exact_results, index_construction_evaluation, index_search_evaluation
from src.evaluation_params import Params, MultiHNSWConstructionParams, MultiHNSWSearchParams


def load_dataset():
    """
    Load the dataset from the prepared paths.
    """
    text_vectors = np.load(TEXT_VECTORS_PATH)
    image_vectors = np.load(IMAGE_VECTORS_PATH)

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
    NUM_INDEXED_ENTITIES = 1_000_000
    NUM_QUERY_ENTITIES = 100
    index_text_vectors = text_vectors_all[:NUM_INDEXED_ENTITIES]
    index_image_vectors = image_vectors_all[:NUM_INDEXED_ENTITIES]
    query_text_vectors = text_vectors_all[-NUM_QUERY_ENTITIES:]
    query_image_vectors = image_vectors_all[-NUM_QUERY_ENTITIES:]

    # define and build index that we will evaluate
    MODALITIES = 2
    DIMENSIONS = [text_vectors_all.shape[1], image_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine", "cosine"]
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


def get_params():
    text_vectors_all, image_vectors_all = load_dataset()
    dataset = [text_vectors_all, image_vectors_all]

    NUM_INDEXED_ENTITIES = 1000

    MODALITIES = 2
    DIMENSIONS = [text_vectors_all.shape[1], image_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine", "cosine"]
    WEIGHTS = [0.5, 0.5]

    K = 10

    NUM_QUERY_ENTITIES = 100

    # set query_ids to last NUM_QUERY_ENTITIES
    query_ids = list(range(len(dataset[0]) - NUM_QUERY_ENTITIES, len(dataset[0])))

    return Params(MODALITIES, DIMENSIONS, DISTANCE_METRICS, WEIGHTS, dataset, NUM_INDEXED_ENTITIES, K, query_ids)

def get_construction_params():
    TARGET_DEGREE = 32
    MAX_DEGREE = 64
    EF_CONSTRUCTION = 20
    SEED = 2

    return MultiHNSWConstructionParams(TARGET_DEGREE, MAX_DEGREE, EF_CONSTRUCTION, SEED)

def get_search_params(params: Params):
    EF_SEARCH = 20

    return MultiHNSWSearchParams(params.k, EF_SEARCH)

def run_exact_results():
    params = get_params()
    print(f"Searching {len(params.query_ids)} ids in exact index:")
    exact_results, times = compute_exact_results(params, cache=True)
    print(exact_results.shape)

def evaluate_construction():

    params = get_params()
    construction_params = get_construction_params()

    index = index_construction_evaluation(params, construction_params)

    # exact_results, times = compute_exact_results(params)
    #print(exact_results.shape)

def evaluate_search():

    params = get_params()
    construction_params = get_construction_params()
    search_params = get_search_params(params)


    exact_results, exact_times = compute_exact_results(params, cache=True)
    index, index_path = index_construction_evaluation(params, construction_params)

    results, search_times, recall_scores = index_search_evaluation(index, index_path, exact_results, params, search_params)

def evaluate_parameter_space():
    params = get_params()
    construction_params = get_construction_params()
    search_params = get_search_params(params)
    NUM_QUERY_ENTITIES = 100

    for index_size in [10_000, 25_000, 50_000, 75_000, 100_000]:
        for k in [10, 50, 100]:
            # set query_ids to last NUM_QUERY_ENTITIES
            query_ids = random.sample(range(params.index_size, len(params.dataset[0])), NUM_QUERY_ENTITIES)
            params.query_ids = query_ids
            params.index_size = index_size
            params.k = k
            search_params.k = k

            exact_results, exact_times = compute_exact_results(params, cache=True) # will cache these when possible

            # construct index
            for target_degree in [16, 32, 64, 128]:
                for max_degree in [target_degree, 2 * target_degree]:
                    for ef_construction in [50, 100, 150, 200]:
                        for seed in [1,2,3,4,5]:
                            construction_params.target_degree = target_degree
                            construction_params.max_degree = max_degree
                            construction_params.ef_construction = ef_construction
                            construction_params.seed = seed

                            index, index_path = index_construction_evaluation(params, construction_params)

                            for ef_search in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                                search_params.ef_search = ef_search
                                results, search_times, recall_scores = index_search_evaluation(index, index_path, exact_results, params, search_params)


if __name__ == "__main__":
    #main()
    #save_image_vectors_to_32(IMAGE_VECTORS_PATH.replace("image_vectors.npy", "image_vectors64.npy"))

    #run_exact_results()
    #evaluate_construction()
    evaluate_search()

    #evaluate_parameter_space()
