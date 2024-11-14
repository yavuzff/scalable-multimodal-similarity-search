"""
MultimodalHNSWIndex implements MultimodalIndex using an HNSW-based approach.
Calls C++ code to achieve this.
"""

from src.index.multimodal_index_interface import MultimodalIndex


class MultimodalHNSWIndex(MultimodalIndex):
    def __init__(self, modalities: int, dims: list[int], distance_metrics: list[str], weights: list[float] = [],
                 ef_construction: int = 200, M: int = 16, M_max: int = 100,
                 random_seed: int = 0, dist_scale_factor: float = 1.1):
        super().__init__(modalities, dims, distance_metrics, weights)
        self._ef_construction = ef_construction
        self._M = M
        self._M_max = M_max
        self._random_seed = random_seed
        self._dist_scale_factor = dist_scale_factor

    def add(self, entities: list[list[list[float]]], ids: list[int] = []):
        pass

    def search(self, query: list[list[float]], k: int, weights: list[float] = [], ef_search: int = 100):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    @property
    def ef_construction(self):
        return self._ef_construction

    @property
    def M(self):
        return self._M

    @property
    def M_max(self):
        return self._M_max

    @property
    def random_seed(self):
        return self._random_seed

    @property
    def dist_scale_factor(self):
        return self._dist_scale_factor
