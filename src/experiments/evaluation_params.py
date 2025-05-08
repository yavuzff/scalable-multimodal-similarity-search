import numpy as np
from src.common.load_dataset import load_dataset, load_4_modality_dataset, load_3_modality_dataset

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
    def __init__(self, k: int, ef_search: int, weights: list[float] = None):
        self.k = k
        self.ef_search = ef_search
        self.weights = weights

def get_params():
    """
    Get the parameters for the exact index setup.
    """
    text_vectors_all, image_vectors_all = load_dataset()
    dataset = [text_vectors_all, image_vectors_all]

    INDEX_SIZE = 1000

    MODALITIES = 2
    DIMENSIONS = [text_vectors_all.shape[1], image_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine", "cosine"]
    WEIGHTS = [0.5, 0.5]

    K = 50

    NUM_QUERY_ENTITIES = 100

    # set query_ids to last NUM_QUERY_ENTITIES
    query_ids = list(range(len(dataset[0]) - NUM_QUERY_ENTITIES, len(dataset[0])))

    return Params(MODALITIES, DIMENSIONS, DISTANCE_METRICS, WEIGHTS, dataset, INDEX_SIZE, K, query_ids)

def get_params_4_modality():
    """
    Get the parameters for the exact index setup.
    """
    text_vectors_all, image_vectors_all, audio_vectors_all, video_vectors_all = load_4_modality_dataset()

    dataset = [text_vectors_all, image_vectors_all, audio_vectors_all, video_vectors_all]

    INDEX_SIZE = 9000

    MODALITIES = 4
    DIMENSIONS = [text_vectors_all.shape[1], image_vectors_all.shape[1], audio_vectors_all.shape[1], video_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine", "cosine", "cosine", "cosine"]
    WEIGHTS = [0.25, 0.25, 0.25, 0.25]

    K = 10

    NUM_QUERY_ENTITIES = 100

    # set query_ids to last NUM_QUERY_ENTITIES
    query_ids = list(range(len(dataset[0]) - NUM_QUERY_ENTITIES, len(dataset[0])))

    return Params(MODALITIES, DIMENSIONS, DISTANCE_METRICS, WEIGHTS, dataset, INDEX_SIZE, K, query_ids)

def get_params_3_modality():
    """
    Get the parameters for the exact index setup.
    """
    image_vectors_all, audio_vectors_all, video_vectors_all = load_3_modality_dataset()

    dataset = [image_vectors_all, audio_vectors_all, video_vectors_all]

    INDEX_SIZE = 9000

    MODALITIES = 3
    DIMENSIONS = [image_vectors_all.shape[1], audio_vectors_all.shape[1], video_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine", "cosine", "cosine"]
    WEIGHTS = [1./3.]*3

    K = 10

    NUM_QUERY_ENTITIES = 100

    # set query_ids to last NUM_QUERY_ENTITIES
    query_ids = list(range(len(dataset[0]) - NUM_QUERY_ENTITIES, len(dataset[0])))

    return Params(MODALITIES, DIMENSIONS, DISTANCE_METRICS, WEIGHTS, dataset, INDEX_SIZE, K, query_ids)


def get_search_params(params: Params):
    """
    Get the parameters for the HNSW index search.
    """
    EF_SEARCH = 20

    return MultiVecHNSWSearchParams(params.k, EF_SEARCH, params.weights)

def get_construction_params():
    """
    Get the parameters for the HNSW index construction.
    """
    TARGET_DEGREE = 16
    MAX_DEGREE = 16
    EF_CONSTRUCTION = 100
    SEED = 2

    return MultiVecHNSWConstructionParams(TARGET_DEGREE, MAX_DEGREE, EF_CONSTRUCTION, SEED)