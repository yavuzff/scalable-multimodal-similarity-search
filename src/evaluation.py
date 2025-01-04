import time
from multimodal_index import ExactMultiIndex
import numpy as np
import psutil
import os


class IndexEvaluator:
    def __init__(self, index: ExactMultiIndex):
        self.index = index

        # create an exact index for recall calculation
        self.exact_index = ExactMultiIndex(
            num_modalities=index.num_modalities,
            dimensions=index.dimensions,
            distance_metrics=index.distance_metrics,
            weights = index.weights
        )

        self.process = psutil.Process(os.getpid())

    def evaluate_add_entities(self, entities: list[np.ndarray]) -> tuple[float, int]:
        """
        Evaluates the time it takes to add entities to the index.
        :arg entities: A list of numpy arrays, each containing the entity vectors for a modality.
        :returns: The total time it took to add the entities to the index and the max memory usage in bytes.
        """
        mem_before = self.process.memory_info().rss
        start_time = time.time()
        self.index.add_entities(entities)
        mem_after = self.process.memory_info().rss

        total_time = time.time() - start_time
        mem_usage = mem_after - mem_before

        # also add it to exact index
        mem_before = self.process.memory_info().rss
        exact_index_start_time = time.time()
        self.exact_index.add_entities(entities)
        exact_index_total_time = time.time() - exact_index_start_time
        mem_after = self.process.memory_info().rss
        print(f"Exact index insertion time: {exact_index_total_time:.3f} seconds.")
        print(f"Exact index memory consumption: {(mem_after - mem_before)/1024/1024} MiB.")

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
            start_time = time.time()
            results = self.index.search(query, k)
            end_time = time.time()
            mem_after = self.process.memory_info().rss

            search_times.append(end_time-start_time)
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
