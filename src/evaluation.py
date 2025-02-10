import time
import numpy as np
import psutil
import os
from datetime import datetime

from multimodal_index import ExactMultiIndex, MultiHNSW
from src.evaluation_params import Params, MultiHNSWConstructionParams, MultiHNSWSearchParams
from src.load_dataset import load_dataset

EXPERIMENTS_DIR = "experiments/"
EXACT_RESULTS_DIR = EXPERIMENTS_DIR + "exact_results/"
CONSTRUCTION_DIR = EXPERIMENTS_DIR + "construction/"
SEARCH_DIR = EXPERIMENTS_DIR + "search/"


def compute_exact_results(p: Params, cache=True):
    """
    Return exact results for this set of inputs, caching if needed.
    """
    assert p.modalities == len(p.dataset)

    save_folder = EXACT_RESULTS_DIR + sanitise_path_string(
        f"{p.modalities}_{p.dimensions}_{p.metrics}_{p.weights}_{p.index_size}_{p.k}/")

    if cache and os.path.exists(save_folder):
        # iterate over every folder in save_folder
        for folder in os.listdir(save_folder):
            cached_query_ids = np.load(save_folder + folder + "/query_ids.npy")
            if np.array_equal(p.query_ids, cached_query_ids):
                print(f"Loading cached results from {save_folder + folder}")
                data = np.load(save_folder + folder + "/results.npz")
                return data["results"], data["search_times"]

    # create directory where we will save our results
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond
    save_folder += current_time + "/"
    os.makedirs(os.path.dirname(save_folder))

    search_times = []
    results = []

    exact_index = ExactMultiIndex(p.modalities, p.dimensions, p.metrics, p.weights)

    entities_to_insert = [modality[:p.index_size] for modality in p.dataset]
    exact_index.add_entities(entities_to_insert)
    print(f"Inserted {len(entities_to_insert[0])} entities to the exact index.")

    for query_id in p.query_ids:
        query = [modality[query_id] for modality in p.dataset]
        start_time = time.perf_counter()
        result = exact_index.search(query, p.k)
        end_time = time.perf_counter()
        search_times.append(end_time - start_time)
        results.append(result)

    # save query_ids, results and search_times at save_folder
    np.save(save_folder + "query_ids.npy", p.query_ids)
    np.savez(save_folder + "results.npz", results=results, search_times=search_times)

    print(f"Saved results.npz to {save_folder}")

    return np.array(results), np.array(search_times)


def index_construction_evaluation(p: Params, specific_params: MultiHNSWConstructionParams):
    """
    Construct an index with the given parameters and save the time it took to construct it to a file.
    """
    save_folder = CONSTRUCTION_DIR + \
                  sanitise_path_string(f"{p.modalities}_{p.dimensions}_{p.metrics}_{p.weights}_{p.index_size}/") + \
                  f"{specific_params.target_degree}_{specific_params.max_degree}_{specific_params.ef_construction}_{specific_params.seed}/"

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond

    entities_to_insert = [modality[:p.index_size] for modality in p.dataset]

    start_time = time.perf_counter()
    multi_hnsw = MultiHNSW(p.modalities, p.dimensions, p.metrics, weights=p.weights,
                           target_degree=specific_params.target_degree,
                           max_degree=specific_params.max_degree,
                           ef_construction=specific_params.ef_construction,
                           seed=specific_params.seed)
    multi_hnsw.add_entities(entities_to_insert)
    total_time = time.perf_counter() - start_time

    os.makedirs(os.path.dirname(save_folder), exist_ok=True)

    save_file = save_folder + current_time + ".npz"
    np.savez(save_file, time=[total_time])

    print(f"Constructed index in {total_time} seconds. Saved to {save_file}")

    return multi_hnsw, save_file


def index_search_evaluation(index: MultiHNSW, index_path: str, exact_results, params: Params,
                            search_params: MultiHNSWSearchParams):
    """
    Search the index with the given parameters and save the ANN results and search times to a file.
    """
    assert params.k == search_params.k

    index_path_components = index_path.split('/')
    index_path_components[-1] = index_path_components[-1].replace(".npz", "")  # update the index path to be a folder
    rest_path = '/'.join(
        index_path_components[2:])  # do not include the starting part of the index path (exp/const/...)
    save_folder = SEARCH_DIR + rest_path + '/'

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond
    save_folder += f"{search_params.k}_{search_params.ef_search}_" + current_time + "/"
    os.makedirs(os.path.dirname(save_folder))

    index.set_ef_search(ef_search=search_params.ef_search)

    search_times = []
    results = []
    recall_scores = []
    for i, query_id in enumerate(params.query_ids):
        query = [modality[query_id] for modality in params.dataset]
        start_time = time.perf_counter()
        result = index.search(query, params.k)
        end_time = time.perf_counter()
        search_times.append(end_time - start_time)
        results.append(result)
        recall_scores.append(compute_recall(result, exact_results[i]))

    # save query_ids, results and search_times at save_folder
    np.save(save_folder + "query_ids.npy", params.query_ids)
    np.savez(save_folder + "results.npz", results=results, search_times=search_times,
             recall_scores=recall_scores, ef_search=search_params.ef_search)

    print(f"Search time (ms), efSearch, recall: {sum(search_times) / len(search_times) * 1000:.3f}, {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")
    print(f"Saved query_ids.npy and results.npz for efSearch={search_params.ef_search} to {save_folder}")
    return results, search_times, recall_scores


def evaluate_single_modality():
    """
    Evaluate the MultiHSNW index construction and search for a single modality.
    """
    import time

    text_vectors_all, image_vectors_all = load_dataset()

    MODALITIES = 1
    DIMENSIONS = [text_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine"]
    WEIGHTS = [1]

    entities_to_insert = [text_vectors_all[:100_000]]

    start_time = time.perf_counter()
    multi_hnsw = MultiHNSW(MODALITIES, DIMENSIONS, DISTANCE_METRICS, weights=WEIGHTS,
                           target_degree=16,
                           max_degree=16,
                           ef_construction=200,
                           seed=10)
    multi_hnsw.add_entities(entities_to_insert)
    total_time = time.perf_counter() - start_time

    print(f"Index construction time: {total_time}")

def sanitise_path_string(path):
    """
    Replace invalid characters in a path string.
    """
    return path.replace(" ", "").replace("'", "").replace("[", ":").replace("]", ":")


class IndexEvaluator:
    """
    Class for evaluating the performance of an index, by computing exact index results and comparing them to evaluated index results.
    """
    def __init__(self, index: MultiHNSW):
        self.index = index

        # create an exact index for recall calculation
        self.exact_index = ExactMultiIndex(
            num_modalities=index.num_modalities,
            dimensions=index.dimensions,
            distance_metrics=index.distance_metrics,
            weights=index.weights
        )

        self.process = psutil.Process(os.getpid())

    def evaluate_add_entities(self, entities: list[np.ndarray]) -> tuple[float, int]:
        """
        Evaluates the time it takes to add entities to the index.
        :arg entities: A list of numpy arrays, each containing the entity vectors for a modality.
        :returns: The total time it took to add the entities to the index and the max memory usage in bytes.
        """
        mem_before = self.process.memory_info().rss
        start_time = time.perf_counter()
        self.index.add_entities(entities)
        total_time = time.perf_counter() - start_time
        mem_after = self.process.memory_info().rss
        mem_usage = mem_after - mem_before

        # also add it to exact index
        mem_before = self.process.memory_info().rss
        exact_index_start_time = time.perf_counter()
        self.exact_index.add_entities(entities)
        exact_index_total_time = time.perf_counter() - exact_index_start_time
        mem_after = self.process.memory_info().rss
        print(f"Exact index insertion time: {exact_index_total_time:.3f} seconds.")
        print(f"Exact index memory consumption: {(mem_after - mem_before) / 1024 / 1024} MiB.")

        return total_time, mem_usage

    def evaluate_search(self, queries: list[np.ndarray], k: int):
        """
        Evaluates the search performance of the index.
        :arg queries: A list of numpy arrays, each containing the query vectors for a modality.
        :arg k: The number of nearest-neighbors to search for.
        :returns: A tuple containing the search times and recall scores.
        """
        num_queries = len(queries[0])
        search_times = []
        recall_scores = []
        memory_consumptions = []
        for i in range(num_queries):
            query = [modality[i] for modality in queries]

            mem_before = self.process.memory_info().rss
            start_time = time.perf_counter()
            results = self.index.search(query, k)
            end_time = time.perf_counter()
            mem_after = self.process.memory_info().rss

            search_times.append(end_time - start_time)
            memory_consumptions.append(mem_after - mem_before)

            # calculate recall
            exact_results = self.exact_index.search(query, k)
            recall_scores.append(compute_recall(results, exact_results))

        return search_times, recall_scores, memory_consumptions


def compute_recall(results, exact_results):
    """
    Compute the recall of the results given the exact results.
    :arg results: The results obtained from the index.
    :arg exact_results: The exact results for the query.
    :returns: The recall of the results.
    """
    return len(set(results).intersection(set(exact_results))) / len(exact_results)
