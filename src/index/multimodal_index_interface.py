"""
This is the interface for the multimodal search index.
This will be used as an interface between C++ indexes and Python user code.
"""
from abc import ABC, abstractmethod


class MultimodalIndex(ABC):
    """
    Abstract class representing the basic operations which every multimodal k-nn index should support.
    """

    def __init__(self, modalities: int, dims: list[int], distance_metrics: list[str], weights: list[float] = []):
        """
        Initialize the multimodal index.
        :param modalities: number of modalities/vectors to represent each entity
        :param dims: list containing the dimension of vectors for each modality
        :param distance_metrics: list containing the distance_metrics for each modality ('Euclidean, 'cosine', 'l2')
        :param weights: (optional) list containing the weights for each modality for index construction
        """
        self._modalities = modalities
        self._dims = dims
        self._distance_metrics = distance_metrics
        self._weights = weights

    @abstractmethod
    def add(self, entities: list[list[list[float]]], ids: list[int] = []) -> None:
        """Add vectors to the index.
        :param entities: list of entities, each entity being list of vectors for each modality
        :param ids: (optional) list of ids for each new entity
        """
        pass

    @abstractmethod
    def search(self, query: list[list[float]], k: int, weights: list[float] = []) -> tuple[list[list[list[float]]], list[int]]:
        """Search for the k nearest neighbors of the query entity.
        :param query: list of vectors for each modality of the query entity
        :param k: number of nearest neighbors to return
        :param weights: (optional) list containing the weights for each modality for search
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to storage"""
        pass

    def load(self, path: str) -> None:
        """Load the index from storage."""
        pass

    @property
    def dims(self):
        """Get the dimensions of the modality vector spaces."""
        return self._dims

    @property
    def distance_metrics(self):
        """Get the distance metrics of the modality vector spaces."""
        return self._distance_metrics

    @property
    def weights(self):
        """Get the weights of the modalities."""
        return self._weights

    @property
    def modalities(self):
        """Get the number of modalities."""
        return self._modalities
