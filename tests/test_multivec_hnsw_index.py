import pytest
import numpy as np
from multivec_index import MultiVecHNSW
import os

def eq(array1, array2):
    return np.all(np.equal(array1, array2))
# eq for float arrays
def eq2(array1, array2):
    return np.all(np.isclose(array1, array2, atol=1e-6))

def test_multivec_hnsw_index_initialisation():
    index = MultiVecHNSW(2, dimensions=[1, 2], distance_metrics=["euclidean", "euclidean"], weights=[0.3, 0.7],
                      ef_construction=100, distribution_scale_factor=1.0)

    assert index.num_modalities == 2
    assert index.dimensions == [1, 2]
    assert index.distance_metrics == ["euclidean", "euclidean"]
    assert eq2(index.weights, [0.3, 0.7])
    assert index.ef_construction == 100
    assert index.num_entities == 0
    assert index.distribution_scale_factor == 1.0
    assert index.target_degree == 32
    assert index.max_degree == 32
    assert index.ef_search == 50
    assert index.seed == 42


def test_searching_multivec_hnsw_single_dim_single_modality():
    # test single modality, single vector
    index = MultiVecHNSW(1, dimensions=np.array([1]), distance_metrics=["euclidean"], weights=[1])

    # add 1 modality, 8 vectors each of 1 dimension
    index.add_entities([[[-2],[-1],[0],[1],[2],[3],[4],[5]]])
    assert index.num_entities == 8

    # query has 1 modality which is 1 dimensional vector
    query1 = [np.array([0])]
    assert index.search(query1, 1) == [2]

    # query within inner location being list
    query2 = [[1]]
    assert index.search(query2, 1) == [3]

    query3 = [[-1.1]]
    assert eq(index.search(query3, 1), [1])
    assert eq(index.search(query3, 2), [1,0])
    assert eq(index.search(query3, 3), [1,0,2])

def test_searching_multivec_hnsw_single_modality_multiple_dim():
    index = MultiVecHNSW(1, dimensions=np.array([3]), distance_metrics=["euclidean"], weights=[1])

    # adding with 1 modality, 4 entities with 3 dimensions for the modality
    index.add_entities([[  #single modality so we have 1 outer array
        [1.0,2.1,3.1], [4,5.0,6], [7.1,8,9], [10,11,12] # modality 1: we are adding 3 vectors, each of dimension 4
    ]])

    assert index.num_entities == 4

    query1 = [[1,2,3]]
    assert eq(index.search(query1, 1), [0])
    assert eq(index.search(query1, 2), [0,1])
    assert eq(index.search(query1, 4), [0,1,2,3])

    query2 = [[-1,-2,-3]]
    assert eq(index.search(query2, 1), [0])
    assert eq(index.search(query2, 2), [0,1])

    query3 = [[6.2,7.1,7.9]]
    assert eq(index.search(query3, 1), [2])
    assert eq(index.search(query3, 2), [2,1])


def test_searching_multivec_hnsw_multiple_modalities_multiple_dim():
    index = MultiVecHNSW(2, dimensions=np.array([2, 3]), distance_metrics=["euclidean", "euclidean"], weights=[0.5,0.5])

    # add 2 modality, 4 entities with dimensions 2, 3 for the modalities
    index.add_entities([
        [[1,1], [2,2], [3,3], [4,4]], #modality 1: shape (4, 2)
        [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]  #modality 2: shape (4, 3)
    ])

    assert index.num_entities == 4

    query1 = [[1,1],[1,1,1]]
    assert eq(index.search(query1, 1), [0])
    assert eq(index.search(query1, 2), [0,1])

    query2 = [[1.9,2],[3,3,3]]
    assert eq(index.search(query2, 1), [2])
    assert eq(index.search(query2, 2), [2,1])

    assert eq(index.search(query2, 1, query_weights=[0.8,0.2]), [1])
    assert eq(index.search(query2, 2, query_weights=[1,0]), [1,0])

    query3 = [[1,3],[3,2,4]]
    # distance to entity 1: sqrt(4) + sqrt(14)
    # distance to entity 2: sqrt(2) + sqrt(5)
    # distance to entity 3: sqrt(4) + sqrt(2)
    # distance to entity 4: sqrt(10) + sqrt(5)
    assert eq(index.search(query3, 1, [0.5, 0.5]), [2])
    assert eq(index.search(query3, 2, [0.5, 0.5]), [2,1])
    assert eq(index.search(query3, 4, [0.5, 0.5]), [2,1,3,0])
    assert eq(index.search(query3, 4, [0.6, 0.4]), [1,2,0,3])
    assert eq(index.search(query3, 4, [0.4, 0.6]), [2,1,3,0])


    # testing if normalised search weights
    query4 = [[1,3],[3,2,4]]
    assert eq(index.search(query4, 1, [5, 5]), [2])
    assert eq(index.search(query4, 2, [0.15, 0.15]), [2,1])
    assert eq(index.search(query4, 4, [300, 300]), [2,1,3,0])
    assert eq(index.search(query4, 4, [1.2, 0.8]), [1,2,0,3])
    assert eq(index.search(query4, 4, [1.2, 1.8]), [2,1,3,0])
    assert eq(index.search(query4, 10000, [1.2, 1.8]), [2,1,3,0])

def test_invalid_multivec_hnsw_searches():
    index = MultiVecHNSW(2, dimensions=np.array([2, 3]), distance_metrics=["euclidean", "euclidean"], weights=[0.5,0.5])

    # add 2 modality, 4 entities with dimensions 2, 3 for the modalities
    index.add_entities([
        [[1,1], [2,2], [3,3], [4,4]], #modality 1: shape (4, 2)
        [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]  #modality 2: shape (4, 3)
    ])

    assert index.num_entities == 4

    # invalid query shape
    with pytest.raises(RuntimeError, match="Input must be a list or tuple of numpy arrays"):
        index.search(np.array([1,2]), 1)

    # invalid query shape
    with pytest.raises(RuntimeError, match="Input must be a 1D array"):
        index.search([1,2], 1)

    # trying to search 4 entities
    with pytest.raises(RuntimeError, match="Input must be a 1D array"):
        index.search([[[1,1], [2,2], [3,3]], #modality 1: shape (4, 2)
                      [[1,1,1], [2,2,2], [3,3,3]]  #modality 2: shape (4, 3)
                      ], 1)

    # invalid query shape
    with pytest.raises(ValueError, match="Entity must have the same number of modalities as the index"):
        index.search([[1]], 1)

    # invalid query shape for modality 2
    with pytest.raises(ValueError, match="Modality 1 has incorrect data size: 2 is not a multiple of the expected dimension 3"):
        index.search([[1,2],[1,2]], 1)

    # invalid query shape for modality 1
    with pytest.raises(ValueError, match="Modality 0 has incorrect data size: 3 is not a multiple of the expected dimension 2"):
        index.search([[1,2,3],[1,2,3]], 1)

    # invalid weights length
    with pytest.raises(ValueError, match="Number of weights must match number of modalities"):
        index.search([[1,2],[1,2,3]], 1, query_weights=[1])

    # invalid weights magnitude
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        index.search([[1,2],[1,2,3]], 1, query_weights=[1,-1])

    # invalid weights sum to 0
    with pytest.raises(ValueError, match="Weights must not be all zero"):
        index.search([[1,2],[1,2,3]], 1, query_weights=[0,0])

    # invalid k
    with pytest.raises(ValueError, match="k must be at least 1"):
        index.search([[1,2],[1,2,3]], 0)

    # invalid k
    with pytest.raises(ValueError, match="k must be at least 1"):
        index.search([[1,2],[1,2,3]], -1)

def test_distance_metrics_single_modality():
    euclidean_index = MultiVecHNSW(1, dimensions=np.array([3]), distance_metrics=["euclidean"])
    manhattan_index = MultiVecHNSW(1, dimensions=np.array([3]), distance_metrics=["manhattan"])
    cosine_index = MultiVecHNSW(1, dimensions=np.array([3]), distance_metrics=["cosine"])

    # adding with 1 modality, 3 entities with 3 dimensions for the modality
    entities = [[  #single modality so we have 1 outer array
        [1.99,3,4], [3,1,5], [1,4,2] # modality 1: we are adding 3 vectors, each of dimension 3
    ]]
    euclidean_index.add_entities(entities)
    manhattan_index.add_entities(entities)
    cosine_index.add_entities(entities)

    query1 = [[3,2,3]]
    # euclidean distances: [sqrt(>3), sqrt(5), sqrt(9)]
    # manhattan distances: [>3, 3, 5]
    # cosine similarities: [24/(sqrt(22*29), 26/(sqrt(22*36)), 17/(sqrt(22*21))]
    # cosine distance: 1 - cosine similarity, order: [  0,1,2
    assert eq(euclidean_index.search(query1, 3), [0,1,2])
    assert eq(manhattan_index.search(query1, 3), [1,0,2])
    assert eq(cosine_index.search(query1, 3), [0,1,2])


    euclidean_index = MultiVecHNSW(1, dimensions=np.array([3]), distance_metrics=["euclidean"])
    manhattan_index = MultiVecHNSW(1, dimensions=np.array([3]), distance_metrics=["manhattan"])
    cosine_index = MultiVecHNSW(1, dimensions=np.array([3]), distance_metrics=["cosine"])

    # adding with 1 modality, 3 entities with 3 dimensions for the modality
    entities = [[  #single modality so we have 1 outer array
        [2.9999,1,4], [1,5,2], [4,2,1] # modality 1: we are adding 3 vectors, each of dimension 3
    ]]

    euclidean_index.add_entities(entities)
    manhattan_index.add_entities(entities)
    cosine_index.add_entities(entities)

    query2 = [[2,3,3]]
    # euclidean distances: [sqrt(<6), sqrt(6), sqrt(9)]
    # manhattan distances: [<4, 4, 5]
    # cosine similarities: [21/(sqrt(22*26), 23/(sqrt(22*30)), 17/(sqrt(22*21))] prop to [4.118,4.199,3.7097]
    # cosine distance: 1 - cosine similarity, order: [1,0,2]
    assert eq(euclidean_index.search(query2, 3), [0,1,2])
    assert eq(manhattan_index.search(query2, 3), [0,1,2])
    assert eq(cosine_index.search(query2, 3), [1,0,2])

def test_combined_metric_multivec_hnsw_search():
    index = MultiVecHNSW(2, dimensions=np.array([3, 3]), distance_metrics=["euclidean", "manhattan"], weights=[0.5,0.5])

    # add 2 modality, 2 entities with dimensions 3, 3 for the modalities
    entities = [
        [[3,1,5], [1.99,3,4]], #modality 1: shape (2, 3)
        [[3,1,5], [1.99,3,4]]  #modality 2: shape (2, 3)
    ]

    index.add_entities(entities)

    #from previous test: euc([3,2,3], [1.99,3,4]) = sqrt(3.0201)
    #from previous test: man([3,2,3], [1.99,3,4]) = 3.01
    #from previous test: euc([3,2,3], [3,1,5]) = sqrt(5)
    #from previous test: man([3,2,3], [3,1,5]) = 3

    query1 = [[3,2,3],[3,2,3]]
    # distances to entity 1: sqrt(5), 3
    # distances to entity 2: sqrt(>3), >3
    assert eq(index.search(query1, 1, [1,0]), [1]) # (sqrt(5), sqrt(3.0201))
    assert eq(index.search(query1, 1), [1]) # even weights
    assert eq(index.search(query1, 1, [0.05,0.995]), [1]) # (2.9618033988749892, 2.946392174561349)
    assert eq(index.search(query1, 1, [0.02,0.98]), [1])  #(2.984721359549996, 2.9845568698245395)
    assert eq(index.search(query1, 1, [0.019,0.981]), [0])  #(2.985485291572496, 2.9858290263333127)
    assert eq(index.search(query1, 1, [0,1]), [0]) # (3,3.01)

def test_manhattan_cosine_multivec_hnsw_search():
    index = MultiVecHNSW(2, dimensions=np.array([3, 3]), distance_metrics=["cosine", "manhattan"], weights=[0.5,0.5])

    # add 2 modality, 2 entities with dimensions 3, 3 for the modalities
    entities = [
        [[3,1,5], [1.99,3,4]], #modality 1: shape (2, 3)
        [[3,1,5], [1.99,3,4]]  #modality 2: shape (2, 3)
    ]

    index.add_entities(entities)

    #from previous test: man([3,2,3], [1.99,3,4]) = 3.01
    #from previous test: man([3,2,3], [3,1,5]) = 3
    #cosine dist([3,2,3], [3,1,5]) = 0.063025
    #cosine dist([3,2,3], [1.99,3,4]) = 0.050365

    query1 = [[3,2,3],[3,2,3]]
    # distances to entity 1: 0.063025, 3
    # distances to entity 2: 0.050365, 3.01
    assert eq(index.search(query1, 1, [1,0]), [1]) # (0.063025,  0.050365)
    assert eq(index.search(query1, 1), [1]) # even weights - (1.5315125, 1.530182)
    assert eq(index.search(query1, 1, [0.45,0.55]), [1]) # (1.67836125, 1.6781638)
    assert eq(index.search(query1, 1, [0.44,0.56]), [0])  #(1.707731, 1.707760)
    assert eq(index.search(query1, 1, [0,1]), [0]) # (3,3.01)

def test_saving_and_loading_index():
    index = MultiVecHNSW(2, dimensions=np.array([2, 3]), distance_metrics=["euclidean", "euclidean"], weights=[0.5,0.5])

    # add 2 modality, 4 entities with dimensions 2, 3 for the modalities
    index.add_entities([
        [[1,1], [2,2], [3,3], [4,4]], #modality 1: shape (4, 2)
        [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]  #modality 2: shape (4, 3)
    ])

    # save index to a file
    index.save("test_index.bin")

    # load index from the file
    loaded_index = MultiVecHNSW(1, dimensions=np.array([2]))
    loaded_index.load("test_index.bin")

    # delete the file after loading
    os.remove("test_index.bin")

    # check if the loaded index has the same number of entities
    assert loaded_index.num_entities == 4
    assert loaded_index.dimensions == index.dimensions
    assert loaded_index.distance_metrics == index.distance_metrics
    assert eq(loaded_index.weights, index.weights)
    assert loaded_index.ef_construction == index.ef_construction
    assert loaded_index.num_entities == index.num_entities
    assert loaded_index.distribution_scale_factor == index.distribution_scale_factor
    assert loaded_index.target_degree == index.target_degree
    assert loaded_index.max_degree == index.max_degree
    assert loaded_index.ef_search == index.ef_search
    assert loaded_index.seed == index.seed

    # check search results are the same
    query = [[1,1],[1,1,1]]
    assert eq(index.search(query, 1), loaded_index.search(query, 1))

# potentially add tests covering efConstruction, efSearch, maxDegree, Degree, Seed etc.