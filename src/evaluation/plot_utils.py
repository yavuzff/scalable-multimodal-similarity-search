from src.evaluation.evaluation_params import Params, MultiVecHNSWConstructionParams, MultiVecHNSWSearchParams
from src.evaluation.evaluation import sanitise_path_string
import os
import numpy as np
from scipy import stats


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