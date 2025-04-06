import numpy as np


class Params:
    def __init__(self, modalities: int, dimensions: list[int], metrics: list[str], weights: list[float],
                 dataset: list[np.ndarray] = None, index_size: int = None, k: int = None, query_ids: list[int] = None):
        self.modalities = modalities
        self.dimensions = dimensions
        self.metrics = metrics
        self.weights = weights
        self.dataset = dataset
        self.index_size = index_size
        if dataset is not None and self.index_size > len(dataset[0]):
            print(
                f"Warning: requested index size {self.index_size} < dataset size ({len(dataset[0])}. Setting index_size to {len(dataset[0])}")
            self.index_size = len(dataset[0])
        self.k = k
        self.query_ids = query_ids


class MultiVecHNSWConstructionParams:
    def __init__(self, target_degree: int, max_degree: int, ef_construction: int, seed: int):
        # distributionScaleFactor ignored - set to default - 1/ln(M)
        self.target_degree = target_degree
        self.max_degree = max_degree
        self.ef_construction = ef_construction
        self.seed = seed


class MultiVecHNSWSearchParams:
    def __init__(self, k: int, ef_search: int):
        self.k = k
        self.ef_search = ef_search
