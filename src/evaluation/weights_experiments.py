import random
from datetime import datetime
import os
import time
import numpy as np

from multivec_index import MultiVecHNSW

from src.evaluation.evaluation import compute_exact_results, load_multivec_index_from_params, sanitise_path_string, \
    compute_recall, SEARCH_WEIGHTS_DIR
from src.evaluation.evaluation_params import get_construction_params, get_params, get_search_params, \
    MultiVecHNSWSearchParams, Params, MultiVecHNSWConstructionParams


def evaluate_weights_experiments(params, construction_params, experiment_dataset_sizes, ef_search_range):
    search_params = get_search_params(params)
    NUM_QUERY_ENTITIES = 100

    experiment_search_weights = [0.1 * i for i in range(11)][::-1]  # [1.0, 0.9, 0.8, ..., 0.1, 0.0]
    #experiment_search_weights = list(np.arange(0.4, 0.5, 0.01)) # [1.0, 0.9, 0.8, ..., 0.1, 0.0]
    k = params.k
    for dataset_size in experiment_dataset_sizes:
        params.index_size = dataset_size
        search_params.k = k
        multivec_index, index_file = load_multivec_index_from_params(params, construction_params)

        print(f"Loaded index with weights {multivec_index.weights} and metrics {multivec_index.distance_metrics}")

        # set query_ids to last NUM_QUERY_ENTITIES
        query_ids = random.sample(range(params.index_size, len(params.dataset[0])), NUM_QUERY_ENTITIES)
        params.query_ids = query_ids
        print(f"Query_ids for dataset_size {dataset_size}:", query_ids)

        index_file_name = index_file.split('/')[-1].split('.')[0]
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond
        save_folder = SEARCH_WEIGHTS_DIR + \
                      sanitise_path_string(f"{params.modalities}_{params.dimensions}_{params.metrics}_{params.weights}_{params.index_size}/") + \
                      f"{construction_params.target_degree}_{construction_params.max_degree}_{construction_params.ef_construction}_{construction_params.seed}/" + \
                      f"{index_file_name}/" + \
                      current_time + "/"

        for text_search_weight in experiment_search_weights:
            search_params.weights = [text_search_weight, 1 - text_search_weight]
            print(f"Running weights experiments for dataset size, search weights: {dataset_size}, {search_params.weights}")

            exact_params = Params(params.modalities, params.dimensions, params.metrics, search_params.weights,
                                  dataset=params.dataset, index_size=params.index_size, k=params.k,
                                  query_ids=query_ids)
            exact_results, exact_times = compute_exact_results(exact_params, cache=True)  # will cache these when possible

            # try values for ef_search=k', starting from ef_search=k
            for ef_search in ef_search_range:
                #for ef_search in range(300, 500, 10):
                search_params.ef_search = ef_search
                # search the index
                evaluate_weighted_search_on_index(multivec_index, exact_results, params, search_params, efs_folder=save_folder)


def evaluate_weighted_search_on_index(index: MultiVecHNSW, exact_results, params: Params,
                          search_params: MultiVecHNSWSearchParams, efs_folder):
    """
    Search the index with the given parameters and save the ANN results and search times to a file.
    """
    assert params.k == search_params.k

    rounded_weights = [round(w, 5) for w in search_params.weights]
    save_folder = efs_folder + sanitise_path_string(f"{search_params.k}_{rounded_weights}/")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond

    save_folder += sanitise_path_string(f"{search_params.k}_{search_params.ef_search}_") + current_time + "/"
    os.makedirs(os.path.dirname(save_folder))

    index.set_ef_search(ef_search=search_params.ef_search)

    #search_times = []
    #results = []
    recall_scores = []
    for i, query_id in enumerate(params.query_ids):
        query = [modality[query_id] for modality in params.dataset]
        #start_time = time.perf_counter()
        result = index.search(query, search_params.k, search_params.weights)
        #end_time = time.perf_counter()
        if len(result) < params.k:
            print(f"WARNING: Search returned less than k={50} results. Returned {len(result)} results for {i}th query id: {query_id}. Padding with -1.")
            result = np.array(list(result) + [-1] * (params.k-len(result)))
        #search_times.append(end_time - start_time)
        #results.append(result)
        recall_scores.append(compute_recall(result, exact_results[i]))

    # save query_ids, results and search_times at save_folder
    np.save(save_folder + "query_ids.npy", params.query_ids)
    np.savez(save_folder + "results.npz", #results=results, search_times=search_times,
             recall_scores=recall_scores, ef_search=search_params.ef_search, weights=params.weights)

    #print(f"Search time, efSearch, recall: {sum(search_times) / len(search_times) * 1000:.3f}, {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")
    print(f"efSearch, recall: {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")
    #print(f"Saved ef={search_params.ef_search} to {save_folder}")


def main():
    params = get_params()
    construction_params = get_construction_params()

    construction_params.target_degree = 16
    construction_params.max_degree = 16
    construction_params.ef_construction = 100
    construction_params.seed = 3

    params.k = 50
    ef_search_range = range(50, 410, 10)
    dataset_sizes = [1_000_000]
    evaluate_weights_experiments(params, construction_params, dataset_sizes, ef_search_range)

if __name__ == "__main__":
    main()
