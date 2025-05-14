"""
This file contains utility functions for analysing and plotting experiment results.
"""

from src.experiments.evaluation_params import Params, MultiVecHNSWConstructionParams, MultiVecHNSWSearchParams
from src.experiments.evaluation import sanitise_path_string
import os
import numpy as np
from scipy import stats
from collections import defaultdict


def get_latest_experiment_file(folder, prev_experiment_index=1):
    """ Get the latest experiment file in the folder."""
    files = os.listdir(folder)
    # get the latest experiment file (file name is time)
    files.sort()
    data_file = files[-prev_experiment_index]
    return data_file


def get_matching_experiment_folder(folder, query_ids):
    """Given a folder and query_ids, return the subfolder (for ef experiments) with the matching query_ids."""
    subfolders = os.listdir(folder)
    subfolders.sort(reverse=True)  # start search from latest
    for subfolder in subfolders:
        if subfolder.startswith("."):
            continue

        # get files in the subfolder
        ef_folders = os.listdir(folder + "/" + subfolder)

        # get an ef folder in the subfolder
        for ef_folder in ef_folders:
            if ef_folder.startswith("."):
                continue
            # load query_ids.npy file
            query_ids_file = os.path.join(folder, subfolder, ef_folder, "query_ids.npy")
            loaded_query_ids = np.load(query_ids_file)

            # check if queries are the same
            if np.array_equal(query_ids, loaded_query_ids):
                return folder + "/" + subfolder

    assert False, f"Could not find matching experiment folder for {folder}"


def find_results_at_target_recall(ef_folders_root, construction_params):
    if construction_params.target_degree == 32:
        target_recall = 0.95
    else:
        target_recall = 0.9

    ef_folders = os.listdir(ef_folders_root)
    # remove any folder that starts with "."
    ef_folders = [ef_folder for ef_folder in ef_folders if not ef_folder.startswith(".")]
    # sort the folders by ef value
    ef_folders.sort(key=lambda x: int(x.split("_")[1]))

    for ef_folder in ef_folders:
        # load results.npz file
        results = np.load(os.path.join(ef_folders_root, ef_folder, "results.npz"))
        # results contains search_times, recall_scores, ef_search, results
        assert ef_folder.split("_")[1] == str(results["ef_search"])
        # get recall scores
        recall_scores = results["recall_scores"]

        # check if we have the target recall
        if np.mean(recall_scores) >= target_recall:
            return results["ef_search"], results["search_times"]

    assert False, f"Could not find results at target recall {target_recall} in {ef_folders_root}"


def read_search_times(params, construction_params, num_entities, query_ids_dict, search_base_folder,
                      prev_experiment_index=1):
    search_times = {}
    recall_achieving_ef = {}
    for num_entity in num_entities:
        params.index_size = num_entity
        query_ids = query_ids_dict[num_entity]

        search_folder = search_base_folder + get_construction_folder(params) + get_hnsw_construction_params_folder(
            construction_params)

        data_folder = get_matching_experiment_folder(search_folder, query_ids)

        recall_achieving_ef[num_entity], search_times[num_entity] = find_results_at_target_recall(data_folder, construction_params)

    return search_times, recall_achieving_ef


def get_query_id_from_search_folder(folder):
    """Get the query id from the search folder."""
    # get the latest experiment folder
    subfolders = os.listdir(folder)
    subfolders.sort(reverse=True)  # start search from latest
    for subfolder in subfolders:
        if subfolder.startswith("."):
            continue
        # load query_ids.npy file
        query_ids_file = os.path.join(folder, subfolder, "query_ids.npy")
        loaded_query_ids = np.load(query_ids_file)
        return loaded_query_ids
    assert False, f"Could not find query ids in {folder}"


def get_search_times_and_query_ids(params, construction_params, num_entities, search_base_folder,
                                   prev_experiment_index=1):
    """For a given parameter setting, get the latest experiment folder and read the search times and query ids."""
    search_times = {}
    query_ids_dict = {}
    recall_achieving_ef = {}

    for num_entity in num_entities:
        params.index_size = num_entity
        search_folder = search_base_folder + get_construction_folder(params) + get_hnsw_construction_params_folder(
            construction_params)

        data_folder = os.path.join(search_folder, get_latest_experiment_folder(search_folder,
                                                                               prev_experiment_index=prev_experiment_index))

        query_ids = get_query_id_from_search_folder(data_folder)
        query_ids_dict[num_entity] = query_ids

        # load results.npz file
        recall_achieving_ef[num_entity], search_times[num_entity] = find_results_at_target_recall(data_folder, construction_params)

    return search_times, recall_achieving_ef, query_ids_dict


def read_construction_times_with_baseline(params, specific_construction_params, num_entities, construction_dir, rerank_construction_dir):
    hnsw_construction_times = defaultdict(list)
    rerank_construction_times = defaultdict(list)

    for num_entity in num_entities:
        params.index_size = num_entity

        # MultiVecHNSW
        construction_folder = construction_dir + get_construction_folder(params) + get_hnsw_construction_params_folder(
            specific_construction_params)

        data_file = get_latest_experiment_file(construction_folder)

        # load the .npz file
        data = np.load(os.path.join(construction_folder, data_file))["time"]

        # get the time
        hnsw_construction_times[num_entity].append(data[0])

        # HNSWRerank
        rerank_construction_folder = rerank_construction_dir + get_construction_folder(
            params) + get_hnsw_construction_params_folder(specific_construction_params)

        rerank_data_file = get_latest_experiment_file(rerank_construction_folder)

        rerank_indexes_times = np.load(os.path.join(rerank_construction_folder, rerank_data_file))["time"]
        rerank_construction_times[num_entity].append(sum(rerank_indexes_times))

    return hnsw_construction_times, rerank_construction_times


def get_gb_size_given_num_vectors(params, num_entity):
    total_dimensions = sum(params.dimensions)
    single_entity_size = total_dimensions * 4 # 4 bytes per vector
    return single_entity_size * num_entity / 10**9

def convert_to_GB(sizes_dict):
    sizes_dict_new = {}
    for num_entity in sizes_dict:
        sizes_dict_new[num_entity] = [x / 10**9 for x in sizes_dict[num_entity]]
    return sizes_dict_new

def get_index_sizes(params, specific_construction_params, num_entities, saved_index_dir):
    index_sizes = defaultdict(list)
    for num_entity in num_entities:
        params.index_size = num_entity

        index_folder = saved_index_dir + get_construction_folder(params) + get_hnsw_construction_params_folder(
            specific_construction_params)

        data_file = get_latest_experiment_file(index_folder)

        index_file = os.path.join(index_folder, data_file)
        index_size = os.path.getsize(index_file)
        index_sizes[num_entity].append(index_size)

    return index_sizes


def get_ef_exp_data(params, construction_params, search_base_folder):
    search_folder = search_base_folder + get_construction_folder(params) + get_hnsw_construction_params_folder(
        construction_params)

    selected_folder = get_latest_experiment_file(search_folder)
    ef_folders = os.listdir(search_folder + "/" + selected_folder)
    ef_folders.sort()

    recall_scores_per_ef = {}
    search_times_per_ef = {}
    for ef_folder in ef_folders:
        if ef_folder.startswith("."):
            continue
        stats = ef_folder.split("_")
        k = int(stats[0])
        ef = int(stats[1])

        if ef >= k:
            # load results.npz file
            results = np.load(os.path.join(search_folder, selected_folder, ef_folder, "results.npz"))
            # results contains search_times, recall_scores, ef_search, results
            assert ef == results["ef_search"]
            assert ef not in recall_scores_per_ef
            recall_scores_per_ef[ef] = results["recall_scores"]
            search_times_per_ef[ef] = results["search_times"]

    return recall_scores_per_ef, search_times_per_ef


def get_construction_folder(params: Params, path_connector_symbol=":"):
    save_folder = sanitise_path_string(
        f"{params.modalities}_{params.dimensions}_{params.metrics}_{params.weights}_{params.index_size}/",
        path_connector_symbol)
    return save_folder

def get_hnsw_construction_params_folder(specific_params: MultiVecHNSWConstructionParams, path_connector_symbol=":"):
    save_folder = sanitise_path_string(
        f"{specific_params.target_degree}_{specific_params.max_degree}_{specific_params.ef_construction}_{specific_params.seed}/",
                path_connector_symbol)
    return save_folder

def get_exact_results_folder(params: Params, path_connector_symbol=":"):
    save_folder = sanitise_path_string(
        f"{params.modalities}_{params.dimensions}_{params.metrics}_{params.weights}_{params.index_size}_{params.k}/",
        path_connector_symbol)
    return save_folder


def compute_mean_and_ci_stats(data, confidence=0.95):
    # data is a list of numbers
    mean = np.mean(data)
    sem = stats.sem(data)
    conf_bound = (1. + confidence) / 2. # e.g. 0.995 for 99% CI
    ci = sem * stats.t.ppf(conf_bound, len(data) - 1)
    return mean, ci

def compute_means_and_cis_from_dict_of_list(data, confidence=0.95):
    # data is a dict of lists
    mean_list = []
    ci_list = []
    for key in data.keys():
        mean, ci = compute_mean_and_ci_stats(data[key], confidence)
        mean_list.append(mean)
        ci_list.append(ci)
    return np.array(mean_list), np.array(ci_list)

def metrics_to_str(metrics):
    # from a list of strings, returns string [Metric1, Metric2, ...], where each metric is capitalised
    metrics = [metric.capitalize() for metric in metrics]
    return "[" + ", ".join(metrics) + "]"

def format_xaxis(x, pos):
    if x >= 1_000_000:
        return f'{x / 1_000_000:.0f}M'
    elif 1000 <= x < 1_000_000:
        return f'{x / 1000:.0f}K'
    else:
        return f'{x:.0f}'


def get_latest_experiment_folder(folder, prev_experiment_index=1):
    """ Get the latest experiment folder in the folder."""
    subfolders = os.listdir(folder)
    # get the latest experiment folder (folder name is time)
    subfolders.sort()
    data_folder = subfolders[-prev_experiment_index]
    return data_folder


def get_search_weights_data(params, construction_params, base_folder, prev_experiment_folder=1, bracket_split_char="-", modalities=2, prev_index_folder=1):
    folder = base_folder + get_construction_folder(params, bracket_split_char) + get_hnsw_construction_params_folder(construction_params, bracket_split_char)
    index_folder = get_latest_experiment_folder(folder, prev_index_folder)
    print(f"Loaded {index_folder}")

    exps_folder = os.path.join(folder, index_folder)
    exp_folder = os.path.join(folder, index_folder, get_latest_experiment_folder(exps_folder, prev_experiment_folder))

    search_weights_folders = os.listdir(exp_folder)

    search_weights_data = defaultdict(lambda: defaultdict(list)) # text_weight -> ef -> recall
    # or if 4 modalities then (w1, w2, w3, w4) -> ef -> recall

    for search_weights_folder in search_weights_folders:
        if search_weights_folder.startswith("."):
            continue

        search_weights = search_weights_folder.split(bracket_split_char)[1]
        if modalities == 2:
            text_weight = float(search_weights.split(",")[0])
        else:
            weights = tuple(float(w) for w in search_weights.split(","))

        for ef_folder in os.listdir(exp_folder + "/" + search_weights_folder):
            if ef_folder.startswith("."):
                continue
            stats = ef_folder.split("_")
            k = int(stats[0])
            ef = int(stats[1])

            if ef >=k:
                # load results.npz file
                results = np.load(os.path.join(exp_folder, search_weights_folder, ef_folder, "results.npz"))
                # results contains recall_scores, ef_search
                assert ef == results["ef_search"]
                if modalities == 2:
                    search_weights_data[text_weight][ef].append(results["recall_scores"])
                else:
                    search_weights_data[weights][ef].append(results["recall_scores"])

    if modalities == 2:
        print(f"Read values for k={k} for dataset size {params.index_size} for {len(search_weights_data[text_weight])} ef values")
    else:
        print(f"Read values for k={k} for dataset size {params.index_size} for {len(search_weights_data[weights])} ef values")
    return search_weights_data, k


def get_construction_stats(params, construction_params, base_folder, bracket_split_char="-", normalised=""):
    folder = base_folder + "construction/" + normalised + get_construction_folder(params, bracket_split_char) + get_hnsw_construction_params_folder(construction_params, bracket_split_char)
    data_file = get_latest_experiment_folder(folder)
    # load data file .npz
    data = np.load(os.path.join(folder, data_file))
    return data

def get_construction_data_per_metric_and_seed(metrics, seed=60, normalised="", base_folder=None):
    times = {}
    num_compute_distance_calls = {}
    num_lazy_distance_calls = {}
    total_distance_calls = {}
    num_lazy_distance_cutoffs = {}  # = num_vectors_skipped_due_to_cutoff, since modality is 1
    for text_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        params = Params(modalities=2, dimensions=[384, 768], metrics=metrics, weights=[text_weight, round(1-text_weight, 5)], index_size=1_000_000)
        construction_params = MultiVecHNSWConstructionParams(target_degree=16, max_degree=16, ef_construction=100, seed=seed)

        data = get_construction_stats(params, construction_params, normalised=normalised, base_folder=base_folder)
        times[text_weight] = data["time"]
        num_compute_distance_calls[text_weight] = data["num_compute_distance_calls"]
        num_lazy_distance_calls[text_weight] = data["num_lazy_distance_calls"]
        num_lazy_distance_cutoffs[text_weight] = data["num_vectors_skipped_due_to_cutoff"]
        total_distance_calls[text_weight] = num_compute_distance_calls[text_weight] + num_lazy_distance_calls[text_weight]

    # for weights 0.0 and 1.0, every distance call is cutoff, but these aren't captured in the stats above so we manually set them
    num_lazy_distance_cutoffs[1.0] = total_distance_calls[1.0]
    num_lazy_distance_cutoffs[0.0] = total_distance_calls[0.0]

    return params, times, num_compute_distance_calls, num_lazy_distance_calls, total_distance_calls, num_lazy_distance_cutoffs


def get_metrics_data(all_metrics, base_folder, seed=60, normalised=""):
    metric_to_total_distances = {}
    metric_to_cutoffs = {}
    for metrics in all_metrics:
        # convert tuple to list
        total_distance_calls = defaultdict(list)
        num_lazy_distance_cutoffs = defaultdict(list)
        for seed in [60]:
            instance_normalised = normalised
            if metrics[1] == "cosine":
                instance_normalised = ""
            params, _, _, _, total_distance_calls_temp, num_lazy_distance_cutoffs_temp = get_construction_data_per_metric_and_seed([i for i in metrics], seed=seed, normalised=instance_normalised, base_folder=base_folder)
            keys = list(total_distance_calls_temp.keys())
            for key in keys:
                total_distance_calls[key].append(total_distance_calls_temp[key])
                num_lazy_distance_cutoffs[key].append(num_lazy_distance_cutoffs_temp[key])

        metric_to_total_distances[metrics] = total_distance_calls
        metric_to_cutoffs[metrics] = num_lazy_distance_cutoffs

    return metric_to_total_distances, metric_to_cutoffs

def get_recall_at_indexweights_data(metrics, construction_params, base_folder, prev_experiment_folder=1, bracket_split_char="-", normalised="normalised/"):
    assert len(metrics) == 2
    recall_data = defaultdict(lambda: defaultdict(list)) # index_text_weight -> ef -> recall
    base_folder += normalised
    for index_text_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(index_text_weight)
        index_image_weight = round(1-index_text_weight, 4)

        params = Params(modalities=2, dimensions=[384, 768], metrics=metrics, weights=[index_text_weight, index_image_weight], index_size=1_000_000)

        folder = base_folder + get_construction_folder(params, bracket_split_char) + get_hnsw_construction_params_folder(construction_params, bracket_split_char)
        index_folder = get_latest_experiment_folder(folder)
        exps_folder = os.path.join(folder, index_folder)
        exp_folder = os.path.join(folder, index_folder, get_latest_experiment_folder(exps_folder, prev_experiment_folder))
        search_weights_folders = os.listdir(exp_folder)

        for search_weights_folder in search_weights_folders:
            if search_weights_folder.startswith("."):
                continue

            search_weights = search_weights_folder.split(bracket_split_char)[1]
            text_weight = float(search_weights.split(",")[0])

            if text_weight != index_text_weight:
                continue
            for ef_folder in os.listdir(exp_folder + "/" + search_weights_folder):
                if ef_folder.startswith("."):
                    continue
                stats = ef_folder.split("_")
                k = int(stats[0])
                ef = int(stats[1])

                if ef >=k:
                    # load results.npz file
                    results = np.load(os.path.join(exp_folder, search_weights_folder, ef_folder, "results.npz"))
                    # results contains recall_scores, ef_search
                    assert ef == results["ef_search"]
                    recall_data[text_weight][ef].append(results["recall_scores"])

    return recall_data, k