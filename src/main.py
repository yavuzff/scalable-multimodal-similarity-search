from multivec_index import MultiVecHNSW
import numpy as np
import random
import time

from src.common.load_dataset import load_dataset
from src.evaluation.evaluation import IndexEvaluator, compute_exact_results, evaluate_index_construction, evaluate_index_search
from src.evaluation.evaluation import evaluate_hnsw_rerank_construction, evaluate_hnsw_rerank_search
from src.evaluation.evaluation_params import Params, MultiVecHNSWConstructionParams, MultiVecHNSWSearchParams
from src.evaluation.evaluation_params import get_params, get_construction_params, get_search_params

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

    # swap out the index for loaded version
    test_save_load = False
    if test_save_load:
        path = "index.dat"
        index_evaluator.index.save(path)
        loaded_index = MultiVecHNSW(1, [1])
        loaded_index.load(path)
        index_evaluator.index = loaded_index

    # evaluate search performance
    queries = [query_text_vectors, query_image_vectors]
    search_times, recall_scores, memory_consumption = index_evaluator.evaluate_search(queries, K)

    print(
        f"Average search time: {np.mean(search_times)*1000:.3f}ms. Variance: {np.var(search_times)*1000*1000:.3f}ms^2. Min: {np.min(search_times)*1000:.3f}ms. Max: {np.max(search_times)*1000:.3f}ms.")
    print(
        f"Average recall: {np.mean(recall_scores):.6f}. Variance: {np.var(recall_scores):.6f}. Min: {np.min(recall_scores):.6f}. Max: {np.max(recall_scores):.6f}.")
    print(
        f"Average memory consumption: {np.mean(memory_consumption):.3f} bytes. Variance: {np.var(memory_consumption):.3f}. Min: {np.min(memory_consumption):.3f}. Max: {np.max(memory_consumption):.3f}.")


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


def evaluate_parameter_space(index_sizes, experiment_seed, experiment_construction_params):
    """
    Evaluate a range of values in the parameter space for the MultiVecHSNW index construction and search.
    """
    params = get_params()
    construction_params = get_construction_params()
    search_params = get_search_params(params)
    NUM_QUERY_ENTITIES = 1000


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
            for target_degree, max_degree, ef_construction, seed in experiment_construction_params:
                            construction_params.target_degree = target_degree
                            construction_params.max_degree = max_degree
                            construction_params.ef_construction = ef_construction
                            construction_params.seed = seed

                            index, index_path = evaluate_index_construction(params, construction_params, save_index=False)

                            # try 20 values for ef_search, starting from ef_search=k-10
                            for ef_search in range(k-10, k + 200, 10):
                                search_params.ef_search = ef_search
                                results, search_times, recall_scores = evaluate_index_search(index, index_path,
                                                                                             exact_results, params,
                                                                                             search_params)

        time.sleep(index_size/5000) # sleep to avoid overloading the system


def evaluate_rerank_hnsw(index_sizes, experiment_seed, experiment_construction_params):
    """
    Evaluate a range of values in the parameter space for the MultiVecHSNW index construction and search.
    """
    from src.evaluation.evaluation import EXACT_RESULTS_DIR, SEARCH_DIR, sanitise_path_string
    import os

    params = get_params()
    construction_params = get_construction_params()
    search_params = get_search_params(params)
    NUM_QUERY_ENTITIES = 1000


    #for index_size in reversed([10_000, 25_000, 50_000, 75_000, 100_000, 250_000, 375_000, 500_000, 625_000, 750_000, 875_000, 1_000_000]):
    #for index_size in [10_000, 25_000, 50_000, 75_000, 100_000, 250_000, 375_000, 500_000, 625_000, 750_000, 875_000, 1_000_000]:
    #for index_size in [1000, 10_000]:
    for index_size in index_sizes:
        params.index_size = index_size
        for k in [50]:
            params.k = k
            search_params.k = k

            # construct indexes
            #for target_degree, max_degree, ef_construction, seed in [(16, 16, 100, experiment_seed), (32, 32, 200, experiment_seed)]:
            for target_degree, max_degree, ef_construction, seed in experiment_construction_params:
                construction_params.target_degree = target_degree
                construction_params.max_degree = max_degree
                construction_params.ef_construction = ef_construction
                construction_params.seed = seed


                # read query ids from the last subfolder for multivechnsw
                multivechnsw_search_folder = SEARCH_DIR + sanitise_path_string(
                    f"{params.modalities}_{params.dimensions}_{params.metrics}_{params.weights}_{params.index_size}/") + \
                                             f"{construction_params.target_degree}_{construction_params.max_degree}_{construction_params.ef_construction}_{construction_params.seed}/"
                search_sub_folders = [f for f in os.listdir(multivechnsw_search_folder) if os.path.isdir(multivechnsw_search_folder + f)]
                last_search_sub_folder = sorted(search_sub_folders)[-1] + "/"

                # get any folder in last_search_sub_folder and then get "/query_ids.npy"
                ef_subfolders = [ef for ef in os.listdir(multivechnsw_search_folder + last_search_sub_folder) if os.path.isdir(multivechnsw_search_folder + last_search_sub_folder + ef)]

                print("Reading query ids and results from folder:", multivechnsw_search_folder + last_search_sub_folder + ef_subfolders[0])
                params.query_ids = np.load(multivechnsw_search_folder + last_search_sub_folder + ef_subfolders[0] + "/query_ids.npy")
                print("Query_ids:", params.query_ids)

                # read exact results for these query ids
                exact_results, _ = compute_exact_results(params, cache=True, recompute=False)  # will cache these when possible


                # construct index
                rerank_hnsw_indexes, index_path = evaluate_hnsw_rerank_construction(params, construction_params)

                # try values for ef_search=k', starting from ef_search=k
                for ef_search in range(k-10, k + 600, 10):
                #for ef_search in range(300, 500, 10):
                    search_params.ef_search = ef_search
                    evaluate_hnsw_rerank_search(rerank_hnsw_indexes, index_path, exact_results, params, search_params)

        time.sleep(index_size/5000) # sleep to avoid overloading the system

def evaluate_weighted_and_tracked_index_building(params, index_size, seeds, save_index=True, shuffle=False):
    text_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    params.index_size = index_size
    construction_params = MultiVecHNSWConstructionParams(target_degree=16, max_degree=16, ef_construction=100, seed=60)

    for seed in seeds:
        construction_params.seed = seed
        for text_weight in text_weights:
            params.weights = [text_weight, round(1 - text_weight, 5)]
            print(f"Running weights experiments for dataset size, seed, search weights: {params.index_size}, {construction_params.seed} {params.weights}")

            index, index_path = evaluate_index_construction(params, construction_params, save_index=save_index, shuffle=shuffle)
            time.sleep(index_size/10000)


if __name__ == "__main__":
    #main()
    #save_image_vectors_to_32(IMAGE_VECTORS_PATH.replace("image_vectors.npy", "image_vectors64.npy"))

    #run_exact_results()
    #evaluate_construction()
    #evaluate_search()

    #all_index_sizes = [10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 250_000, 375_000, 500_000, 625_000, 750_000, 875_000, 1_000_000]
    #print("Starting parameter space evaluation...")
    #experiment_construction_params = [(32, 32, 200, 9)]
    #evaluate_parameter_space(all_index_sizes, 9, experiment_construction_params)
    #print("Starting rerank evaluation for the previous parameter space...")
    #evaluate_rerank_hnsw(all_index_sizes, 9, experiment_construction_params)

    # weighted index construction experiments
    index_size = 1_000_000
    #seeds = [60, 61]
    seeds = [60]
    params = get_params()
    params.index_size = index_size

    params.metrics = ["manhattan", "euclidean"] # next: manhattan, euclidean. euclidean, cosine.
    evaluate_weighted_and_tracked_index_building(params, index_size, seeds, save_index=False)

    params.metrics = ["euclidean", "cosine"] # next: manhattan, euclidean. euclidean, cosine.
    evaluate_weighted_and_tracked_index_building(params, index_size, seeds, save_index=False)

    #random shuffle tests: params.metrics = ["cosine", "cosine"], and ["cosine", "euclidean"]
    #evaluate_weighted_and_tracked_index_building(params, index_size, [71], save_index=False, shuffle=True)
