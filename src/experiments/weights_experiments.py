import random
from datetime import datetime
import os
import time
import numpy as np

from multivec_index import MultiVecHNSW

from src.experiments.evaluation import compute_exact_results, load_multivec_index_from_params, sanitise_path_string, \
    compute_recall, SEARCH_WEIGHTS_DIR
from src.experiments.evaluation_params import get_construction_params, get_params, get_search_params, \
    MultiVecHNSWSearchParams, Params, MultiVecHNSWConstructionParams


def evaluate_weights_experiments(params, construction_params, experiment_dataset_sizes, ef_search_range, normalised_dataset=False, num_query_entities=100, only_index_weights=False):
    """
    Evaluate the effect of different search weights on the recall of the index.
    :param only_index_weights: if true, search based on only the weights the index was constructed with. If false, search with all weights from 0.0 to 1.0 (step 0.1)
    """
    search_params = get_search_params(params)

    if only_index_weights:
        experiment_search_weights = [params.weights[0]]
    else:
        experiment_search_weights = [0.1 * i for i in range(11)][::-1]  # [1.0, 0.9, 0.8, ..., 0.1, 0.0]

    k = params.k
    for dataset_size in experiment_dataset_sizes:
        params.index_size = dataset_size
        search_params.k = k
        if normalised_dataset:
            multivec_index, index_file = load_multivec_index_from_params(params, construction_params, subdir_name="normalised/")
        else:
            multivec_index, index_file = load_multivec_index_from_params(params, construction_params)

        print(f"Loaded index with weights {multivec_index.weights} and metrics {multivec_index.distance_metrics}")

        # set query_ids to last NUM_QUERY_ENTITIES
        query_ids = random.sample(range(params.index_size, len(params.dataset[0])), num_query_entities)
        params.query_ids = query_ids
        print(f"Query_ids for dataset_size {dataset_size}:", query_ids)

        index_file_name = index_file.split('/')[-1].split('.')[0]
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond

        save_folder = SEARCH_WEIGHTS_DIR
        if normalised_dataset:
            save_folder += "normalised/"
        save_folder += sanitise_path_string(f"{params.modalities}_{params.dimensions}_{params.metrics}_{params.weights}_{params.index_size}/") + \
                      f"{construction_params.target_degree}_{construction_params.max_degree}_{construction_params.ef_construction}_{construction_params.seed}/" + \
                      f"{index_file_name}/" + \
                      current_time + "/"

        for text_search_weight in experiment_search_weights:
            search_params.weights = [text_search_weight, 1 - text_search_weight]
            print(f"Running weights experiments for dataset size, search weights: {dataset_size}, {search_params.weights}. (index weights: {params.weights})")

            exact_params = Params(params.modalities, params.dimensions, params.metrics, search_params.weights,
                                  dataset=params.dataset, index_size=params.index_size, k=params.k,
                                  query_ids=query_ids)
            if normalised_dataset:
                exact_results, exact_times = compute_exact_results(exact_params, cache=True, subdir_name="normalised/")
            else:
                exact_results, exact_times = compute_exact_results(exact_params, cache=True)

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
    num_compute_distance_calls = []
    num_lazy_distance_calls = []
    num_lazy_distance_cutoffs = []
    num_vectors_skipped_due_to_cutoffs = []
    for i, query_id in enumerate(params.query_ids):
        query = [modality[query_id] for modality in params.dataset]
        index.reset_stats()
        #start_time = time.perf_counter()
        result = index.search(query, search_params.k, search_params.weights)
        #end_time = time.perf_counter()
        if len(result) < params.k:
            print(f"WARNING: Search returned less than k={50} results. Returned {len(result)} results for {i}th query id: {query_id}. Padding with -1.")
            result = np.array(list(result) + [-1] * (params.k-len(result)))
        #search_times.append(end_time - start_time)
        #results.append(result)
        num_compute_distance_calls.append(index.num_compute_distance_calls)
        num_lazy_distance_calls.append(index.num_lazy_distance_calls)
        num_lazy_distance_cutoffs.append(index.num_lazy_distance_cutoff)
        num_vectors_skipped_due_to_cutoffs.append(index.num_vectors_skipped_due_to_cutoff)
        recall_scores.append(compute_recall(result, exact_results[i]))

    # save query_ids, results and search_times at save_folder
    np.save(save_folder + "query_ids.npy", params.query_ids)
    np.savez(save_folder + "results.npz", #results=results, search_times=search_times,
             recall_scores=recall_scores, ef_search=search_params.ef_search, weights=params.weights,
             num_compute_distance_calls=num_compute_distance_calls,
             num_lazy_distance_calls=num_lazy_distance_calls,
             num_lazy_distance_cutoffs=num_lazy_distance_cutoffs,
             num_vectors_skipped_due_to_cutoffs=num_vectors_skipped_due_to_cutoffs)

    #print(f"Search time, efSearch, recall: {sum(search_times) / len(search_times) * 1000:.3f}, {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")
    print(f"efSearch, recall: {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")
    #print(f"Saved ef={search_params.ef_search} to {save_folder}")

def normalise_dataset(dataset):
    for i in range(len(dataset)):
        dataset[i] = dataset[i] / np.linalg.norm(dataset[i], axis=1, keepdims=True)
    return dataset

def evaluate_index_weights_recall():
    index_text_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    normalised_dataset = True
    metrics = ["manhattan", "euclidean"]
    num_query_entities = 1000

    for index_text_weight in index_text_weights:
        index_image_weight = round(1 - index_text_weight, 5)
        print(f"Testing index weights: {index_text_weight}, {index_image_weight}")
        params = get_params()
        construction_params = get_construction_params()
        params.weights = [index_text_weight, index_image_weight]
        params.metrics = metrics
        if normalised_dataset: normalise_dataset(params.dataset)
        construction_params.target_degree = 16
        construction_params.max_degree = 16
        construction_params.ef_construction = 100
        construction_params.seed = 60

        params.k = 50
        ef_search_range = range(50, 260, 10)
        dataset_sizes = [1_000_000]
        evaluate_weights_experiments(params, construction_params, dataset_sizes, ef_search_range, normalised_dataset, num_query_entities, only_index_weights=True)


def main():
    index_text_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    normalised_dataset = True
    for metrics in [["cosine","euclidean"],["manhattan","euclidean"]]:
        for index_text_weight in index_text_weights:
            index_image_weight = round(1 - index_text_weight, 5)
            print(f"Testing index weights: {index_text_weight}, {index_image_weight}")
            params = get_params()
            construction_params = get_construction_params()
            params.weights = [index_text_weight, index_image_weight]
            params.metrics = metrics
            if normalised_dataset: normalise_dataset(params.dataset)

            construction_params.target_degree = 16
            construction_params.max_degree = 16
            construction_params.ef_construction = 100
            construction_params.seed = 60

            params.k = 50
            ef_search_range = range(50, 410, 10)
            dataset_sizes = [1_000_000]
            evaluate_weights_experiments(params, construction_params, dataset_sizes, ef_search_range, normalised_dataset)

if __name__ == "__main__":
    #main()
    evaluate_index_weights_recall()
