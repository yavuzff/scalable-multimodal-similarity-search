from multivec_index import MultiVecHNSW
import numpy as np
import random
import time

from src.load_dataset import load_dataset
from src.evaluation import IndexEvaluator, compute_exact_results, evaluate_index_construction, evaluate_index_search
from src.evaluation import evaluate_hnsw_rerank_construction, evaluate_hnsw_rerank_search
from src.evaluation_params import Params, MultiVecHNSWConstructionParams, MultiVecHNSWSearchParams


def main():
    # load dataset
    text_vectors_all, image_vectors_all = load_dataset()

    # define subset for indexing and querying
    NUM_INDEXED_ENTITIES = 10_000
    NUM_QUERY_ENTITIES = 1000
    index_text_vectors = text_vectors_all[:NUM_INDEXED_ENTITIES]
    index_image_vectors = image_vectors_all[:NUM_INDEXED_ENTITIES]
    query_text_vectors = text_vectors_all[-NUM_QUERY_ENTITIES:]
    query_image_vectors = image_vectors_all[-NUM_QUERY_ENTITIES:]

    # define and build index that we will evaluate
    MODALITIES = 2
    DIMENSIONS = [text_vectors_all.shape[1], image_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine", "cosine"]
    my_index = MultiVecHNSW(MODALITIES, dimensions=DIMENSIONS, distance_metrics=DISTANCE_METRICS, weights=[0.5, 0.5],
                         target_degree=32, max_degree=32, ef_construction=200, seed=1)
    my_index.set_ef_search(50)
    # search parameters
    K = 50

    # evaluate the index
    index_evaluator = IndexEvaluator(my_index)

    # evaluate inserting to the index
    entities = [index_text_vectors, index_image_vectors]
    insertion_time, memory_consumption = index_evaluator.evaluate_add_entities(entities)
    print(f"Insertion Time: {insertion_time:.3f} seconds.")
    print(f"Insertion Memory: {memory_consumption / 1024 / 1024} MiB.")

    # evaluate search performance
    queries = [query_text_vectors, query_image_vectors]
    search_times, recall_scores, memory_consumption = index_evaluator.evaluate_search(queries, K)

    print(
        f"Average search time: {np.mean(search_times)*1000:.3f}ms. Variance: {np.var(search_times)*1000*1000:.3f}ms^2. Min: {np.min(search_times)*1000:.3f}ms. Max: {np.max(search_times)*1000:.3f}ms.")
    print(
        f"Average recall: {np.mean(recall_scores):.3f}. Variance: {np.var(recall_scores):.3f}. Min: {np.min(recall_scores):.3f}. Max: {np.max(recall_scores):.3f}.")
    print(
        f"Average memory consumption: {np.mean(memory_consumption):.3f} bytes. Variance: {np.var(memory_consumption):.3f}. Min: {np.min(memory_consumption):.3f}. Max: {np.max(memory_consumption):.3f}.")


def get_params():
    """
    Get the parameters for the exact index setup.
    """
    text_vectors_all, image_vectors_all = load_dataset()
    dataset = [text_vectors_all, image_vectors_all]

    INDEX_SIZE = 1000

    MODALITIES = 2
    DIMENSIONS = [text_vectors_all.shape[1], image_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine", "cosine"]
    WEIGHTS = [0.5, 0.5]

    K = 50

    NUM_QUERY_ENTITIES = 100

    # set query_ids to last NUM_QUERY_ENTITIES
    query_ids = list(range(len(dataset[0]) - NUM_QUERY_ENTITIES, len(dataset[0])))

    return Params(MODALITIES, DIMENSIONS, DISTANCE_METRICS, WEIGHTS, dataset, INDEX_SIZE, K, query_ids)


def get_construction_params():
    """
    Get the parameters for the HNSW index construction.
    """
    TARGET_DEGREE = 32
    MAX_DEGREE = 64
    EF_CONSTRUCTION = 20
    SEED = 2

    return MultiVecHNSWConstructionParams(TARGET_DEGREE, MAX_DEGREE, EF_CONSTRUCTION, SEED)


def get_search_params(params: Params):
    """
    Get the parameters for the HNSW index search.
    """
    EF_SEARCH = 20

    return MultiVecHNSWSearchParams(params.k, EF_SEARCH)


def run_exact_results():
    """
    Evaluate the exact results search.
    """
    params = get_params()
    params.k = 50
    params.index_size = 10000
    params.metrics = ["euclidean", "euclidean"]
    print(f"Searching {len(params.query_ids)} ids in exact index:")
    exact_results, times = compute_exact_results(params, cache=True)
    print("Searched for query ids", params.query_ids)
    print("Exact results are:", exact_results)


def evaluate_construction():
    """
    Evaluate the MultiVecHSNW index construction.
    """
    params = get_params()
    construction_params = get_construction_params()
    evaluate_index_construction(params, construction_params)


def evaluate_search():
    """
    Evaluate the MultiVecHSNW index search.
    """
    params = get_params()
    construction_params = get_construction_params()
    search_params = get_search_params(params)

    exact_results, exact_times = compute_exact_results(params, cache=True)
    print("Exact results are:", exact_results)
    print("Exact times are:", exact_times)

    index, index_path = evaluate_index_construction(params, construction_params)

    results, search_times, recall_scores = evaluate_index_search(index, index_path, exact_results, params,
                                                                 search_params)


def evaluate_parameter_space(index_sizes):
    """
    Evaluate a range of values in the parameter space for the MultiVecHSNW index construction and search.
    """
    params = get_params()
    construction_params = get_construction_params()
    search_params = get_search_params(params)
    NUM_QUERY_ENTITIES = 1000

    experiment_seed = 1

    #for index_size in [10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 250_000, 375_000, 500_000, 625_000, 750_000, 875_000, 1_000_000]:
    #for index_size in reversed([10_000, 25_000, 50_000, 75_000, 100_000, 250_000, 375_000, 500_000, 625_000, 750_000, 875_000, 1_000_000]):
    #for index_size in [10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 250_000, 375_000]:
    #for index_size in [1000]:
    for index_size in index_sizes:
        params.index_size = index_size

        for k in [50]:
            params.k = k
            search_params.k = k

            # set query_ids to a random sample of NUM_QUERY_ENTITIES unused entities
            query_ids = random.sample(range(params.index_size, len(params.dataset[0])), NUM_QUERY_ENTITIES)
            params.query_ids = query_ids
            print("Query_ids:", query_ids)
            exact_results, exact_times = compute_exact_results(params, cache=True)  # will cache these when possible
            print(
                f"Average time for k={k} exact search on index_size {index_size} is {round(np.mean(exact_times) * 1000, 3)} ms.")

            # construct index
            # for target_degree in [16]:
            #     for max_degree in [target_degree]:
            #         for ef_construction in [100]:
            #             #for seed in [1,2,3,4,5]:
            #             for seed in [1]:
            for target_degree, max_degree, ef_construction, seed in [(16, 16, 100, experiment_seed), (32, 32, 200, experiment_seed)]:
                            construction_params.target_degree = target_degree
                            construction_params.max_degree = max_degree
                            construction_params.ef_construction = ef_construction
                            construction_params.seed = seed

                            index, index_path = evaluate_index_construction(params, construction_params)

                            # try 20 values for ef_search, starting from ef_search=k-10
                            for ef_search in range(k-10, k + 200, 10):
                                search_params.ef_search = ef_search
                                results, search_times, recall_scores = evaluate_index_search(index, index_path,
                                                                                             exact_results, params,
                                                                                             search_params)

        time.sleep(index_size/5000) # sleep to avoid overloading the system


def evaluate_rerank_hnsw(index_sizes):
    """
    Evaluate a range of values in the parameter space for the MultiVecHSNW index construction and search.
    """
    from src.evaluation import EXACT_RESULTS_DIR, sanitise_path_string
    import os

    params = get_params()
    construction_params = get_construction_params()
    search_params = get_search_params(params)
    NUM_QUERY_ENTITIES = 1000

    experiment_seed = 1

    #for index_size in reversed([10_000, 25_000, 50_000, 75_000, 100_000, 250_000, 375_000, 500_000, 625_000, 750_000, 875_000, 1_000_000]):
    #for index_size in [10_000, 25_000, 50_000, 75_000, 100_000, 250_000, 375_000, 500_000, 625_000, 750_000, 875_000, 1_000_000]:
    #for index_size in [1000, 10_000]:
    for index_size in index_sizes:
        params.index_size = index_size
        for k in [50]:
            params.k = k
            search_params.k = k

            save_folder = EXACT_RESULTS_DIR + sanitise_path_string(
                f"{params.modalities}_{params.dimensions}_{params.metrics}_{params.weights}_{params.index_size}_{params.k}/")

            # read save_folder and get the folder with the date/time that is the latest
            sub_folders = [f for f in os.listdir(save_folder) if os.path.isdir(save_folder + f)]
            last_sub_folder = sorted(sub_folders)[-1]
            print("Reading query ids and results from folder:", save_folder+ last_sub_folder)
            # read query_ids.npy from last_sub_folder
            params.query_ids = np.load(save_folder + last_sub_folder + "/query_ids.npy")
            exact_results = np.load(save_folder + last_sub_folder + "/results.npz")["results"]
            print("Query_ids:", params.query_ids)

            # construct indexes
            for target_degree, max_degree, ef_construction, seed in [(16, 16, 100, experiment_seed), (32, 32, 200, experiment_seed)]:
                construction_params.target_degree = target_degree
                construction_params.max_degree = max_degree
                construction_params.ef_construction = ef_construction
                construction_params.seed = seed

                rerank_hnsw_indexes, index_path = evaluate_hnsw_rerank_construction(params, construction_params)

                # try values for ef_search=k', starting from ef_search=k
                for ef_search in range(k-10, k + 600, 10):
                #for ef_search in range(300, 500, 10):
                    search_params.ef_search = ef_search
                    evaluate_hnsw_rerank_search(rerank_hnsw_indexes, index_path, exact_results, params, search_params)

        time.sleep(index_size/5000) # sleep to avoid overloading the system

if __name__ == "__main__":
    main()
    #save_image_vectors_to_32(IMAGE_VECTORS_PATH.replace("image_vectors.npy", "image_vectors64.npy"))

    #run_exact_results()
    #evaluate_construction()
    #evaluate_search()


#    all_index_sizes = [10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 250_000, 375_000, 500_000, 625_000, 750_000, 875_000, 1_000_000]
#    print("Starting parameter space evaluation...")
#    evaluate_parameter_space([1_000_000])
#    print("Starting rerank evaluation for the previous parameter space...")
#    evaluate_rerank_hnsw(all_index_sizes)
