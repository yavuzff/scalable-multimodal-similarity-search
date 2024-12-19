"""
ExactMultimodalIndex implements MultimodalIndex using an exact linear search approach.
Calls C++ code to achieve this.
"""
from src.index.multimodal_index_interface import MultimodalIndex


class ExactMultimodalIndex(MultimodalIndex):
    def __init__(self, modalities: int, dims: list[int], distance_metrics: list[str], weights: list[float] = []):
        super().__init__(modalities, dims, distance_metrics, weights)

    def add(self, entities: list[list[list[float]]], ids: list[int] = []):
        pass

    def search(self, query: list[list[float]], k: int, weights: list[float] = []):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
