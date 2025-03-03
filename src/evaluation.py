import time
import numpy as np
import psutil
import os
from datetime import datetime

from multivec_index import ExactMultiVecIndex, MultiVecHNSW
from src.evaluation_params import Params, MultiVecHNSWConstructionParams, MultiVecHNSWSearchParams
from src.load_dataset import load_dataset

EXPERIMENTS_DIR = "experiments/clean/"
EXACT_RESULTS_DIR = EXPERIMENTS_DIR + "exact_results/"
CONSTRUCTION_DIR = EXPERIMENTS_DIR + "construction/"
SEARCH_DIR = EXPERIMENTS_DIR + "search/"

RERANK_CONSTRUCTION_DIR = EXPERIMENTS_DIR + "rerank_construction/"
RERANK_SEARCH_DIR = EXPERIMENTS_DIR + "rerank_search/"

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

    exact_index = ExactMultiVecIndex(p.modalities, p.dimensions, p.metrics, p.weights)

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

    print(f"Saved exact results.npz and query_ids.npy to {save_folder}")

    return np.array(results), np.array(search_times)


def evaluate_index_construction(p: Params, specific_params: MultiVecHNSWConstructionParams):
    """
    Construct an index with the given parameters and save the time it took to construct it to a file.
    """
    save_folder = CONSTRUCTION_DIR + \
                  sanitise_path_string(f"{p.modalities}_{p.dimensions}_{p.metrics}_{p.weights}_{p.index_size}/") + \
                  f"{specific_params.target_degree}_{specific_params.max_degree}_{specific_params.ef_construction}_{specific_params.seed}/"

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond

    entities_to_insert = [modality[:p.index_size] for modality in p.dataset]

    start_time = time.perf_counter()
    multivec_hnsw = MultiVecHNSW(p.modalities, p.dimensions, p.metrics, weights=p.weights,
                           target_degree=specific_params.target_degree,
                           max_degree=specific_params.max_degree,
                           ef_construction=specific_params.ef_construction,
                           seed=specific_params.seed)
    multivec_hnsw.add_entities(entities_to_insert)
    total_time = time.perf_counter() - start_time

    os.makedirs(os.path.dirname(save_folder), exist_ok=True)

    save_file = save_folder + current_time + ".npz"
    np.savez(save_file, time=[total_time])

    print(f"Constructed index in {total_time} seconds. Saved to {save_file}")

    return multivec_hnsw, save_file


def evaluate_index_search(index: MultiVecHNSW, index_path: str, exact_results, params: Params,
                          search_params: MultiVecHNSWSearchParams):
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

    print(f"Search time, efSearch, recall: {sum(search_times) / len(search_times) * 1000:.3f}, {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")
    print(f"Saved efSearch={search_params.ef_search} results to {save_folder}")
    return results, search_times, recall_scores


def evaluate_single_modality():
    """
    Evaluate the MultiVecHSNW index construction and search for a single modality.
    """
    import time

    text_vectors_all, image_vectors_all = load_dataset()

    MODALITIES = 1
    DIMENSIONS = [text_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine"]
    WEIGHTS = [1]

    entities_to_insert = [text_vectors_all[:100_000]]

    start_time = time.perf_counter()
    multivec_hnsw = MultiVecHNSW(MODALITIES, DIMENSIONS, DISTANCE_METRICS, weights=WEIGHTS,
                           target_degree=16,
                           max_degree=16,
                           ef_construction=200,
                           seed=10)
    multivec_hnsw.add_entities(entities_to_insert)
    total_time = time.perf_counter() - start_time

    print(f"Index construction time: {total_time}")


def evaluate_hnsw_rerank_construction(p: Params, specific_params: MultiVecHNSWConstructionParams):
    """
    Evaluate the construction and search of a HNSW index with reranking.
    """
    print("Starting rerank construction at ", datetime.now(), " for ", p.dimensions, " and ", p.metrics)

    save_folder = RERANK_CONSTRUCTION_DIR + \
                  sanitise_path_string(f"{p.modalities}_{p.dimensions}_{p.metrics}_{p.weights}_{p.index_size}/") + \
                  f"{specific_params.target_degree}_{specific_params.max_degree}_{specific_params.ef_construction}_{specific_params.seed}/"

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond

    indexes = []
    construction_times = []
    for i in range(p.modalities):
        vectors_to_insert = [p.dataset[i][:p.index_size]]
        start_time = time.perf_counter()
        index = MultiVecHNSW(1, [p.dimensions[i]], [p.metrics[i]], weights=[1],
                               target_degree=specific_params.target_degree,
                               max_degree=specific_params.max_degree,
                               ef_construction=specific_params.ef_construction,
                               seed=specific_params.seed)
        index.add_entities(vectors_to_insert)
        total_time = time.perf_counter() - start_time
        construction_times.append(total_time)
        indexes.append(index)

    print(f"Constructed indexes in: {construction_times} with total time {sum(construction_times)}")
    print(f"Index params were {specific_params.target_degree}, {specific_params.max_degree}, {specific_params.ef_construction}, {specific_params.seed}")

    # save construction times
    os.makedirs(os.path.dirname(save_folder), exist_ok=True)
    save_file = save_folder + current_time + ".npz"
    np.savez(save_file, time=construction_times)
    print(f"Saved construction times to {save_file}")

    return indexes, save_file


def evaluate_hnsw_rerank_search(indexes, index_path: str, exact_results, params: Params, search_params: MultiVecHNSWSearchParams):
    """
    Evaluate the search performance of a HNSW index with reranking. We use ef search as the k for each search.
    """
    index_path_components = index_path.split('/')
    index_path_components[-1] = index_path_components[-1].replace(".npz", "")  # update the index path to be a folder
    rest_path = '/'.join(
        index_path_components[2:])  # do not include the starting part of the index path (exp/construction/...)
    save_folder = RERANK_SEARCH_DIR + rest_path + '/'

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond
    save_folder += f"{search_params.k}_{search_params.ef_search}_" + current_time + "/"
    os.makedirs(os.path.dirname(save_folder))

    for index in indexes:
        index.set_ef_search(search_params.ef_search)

    # note that we do not keep track of the results returned, just the recall
    search_times = []
    recall_scores = []
    for i, query_id in enumerate(params.query_ids):
        start_time = time.perf_counter()
        ids = []
        for modality in range(0, len(indexes)):
            result = indexes[modality].search([params.dataset[modality][query_id]], search_params.ef_search)
            ids.append(result)
        # flatten the ids into a set containing unique elements
        result = set([item for sublist in ids for item in sublist])
        end_time = time.perf_counter()

        search_times.append(end_time - start_time)
        recall_scores.append(compute_recall(result, exact_results[i]))

    # save query_ids, recall_scores and search_times at save_folder
    np.save(save_folder + "query_ids.npy", params.query_ids)
    np.savez(save_folder + "results.npz", search_times=search_times,
             recall_scores=recall_scores, ef_search=search_params.ef_search)

    print(f"Rerank: Search time, efSearch=k', recall: {sum(search_times) / len(search_times) * 1000:.3f}, {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")


def sanitise_path_string(path):
    """
    Replace invalid characters in a path string.
    """
    return path.replace(" ", "").replace("'", "").replace("[", ":").replace("]", ":")


class IndexEvaluator:
    """
    Class for evaluating the performance of an index, by computing exact index results and comparing them to evaluated index results.
    """
    def __init__(self, index: MultiVecHNSW):
        self.index = index

        # create an exact index for recall calculation
        self.exact_index = ExactMultiVecIndex(
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
