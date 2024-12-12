import time
from cppindex import ExactMultiIndex
import numpy as np

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

    def evaluate_add_entities(self, entities: list[np.ndarray]) -> float:
        """
        Evaluates the time it takes to add entities to the index.
        :arg entities: A list of numpy arrays, each containing the entity vectors for a modality.
        :returns: The total time it took to add the entities to the index.
        """
        start_time = time.time()
        self.index.add_entities(entities)
        total_time = time.time() - start_time

        # also add it to exact index
        self.exact_index.add_entities(entities)
        return total_time

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
        for i in range(num_queries):
            query = [modality[i] for modality in queries]

            start_time = time.time()
            results = self.index.search(query, k)
            search_time = time.time() - start_time
            search_times.append(search_time)

            # calculate recall
            exact_results = self.exact_index.search(query, k)
            recall_scores.append(compute_recall(results, exact_results))

        return search_times, recall_scores

def compute_recall(results, exact_results):
    """
    Compute the recall of the results given the exact results.
    :arg results: The results obtained from the index.
    :arg exact_results: The exact results for the query.
    :returns: The recall of the results.
    """
    return len(set(results).intersection(set(exact_results))) / len(exact_results)
