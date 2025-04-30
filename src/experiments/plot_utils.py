from src.experiments.evaluation_params import Params, MultiVecHNSWConstructionParams, MultiVecHNSWSearchParams
from src.experiments.evaluation import sanitise_path_string
import os
import numpy as np
from scipy import stats
from collections import defaultdict


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
