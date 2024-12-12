import pytest
import numpy as np
import cppindex

def test_simple_index():
    my_index = cppindex.ExactIndex()

    # given
    my_index.add(np.array([3, 4]))
    my_index.add(np.array([1, 2]))
    my_index.add(np.array([0, 0]))
    my_index.add(np.array([-1, -2]))
    my_index.add(np.array([-3, -4]))

    # when
    result_1 = my_index.search(np.array([0, 0]), 2)
    result_2 = my_index.search(np.array([0, 0]), 3)
    result_3 = my_index.search(np.array([5, 6]), 3)
    result_4 = my_index.search(np.array([15, 16]), 5)

    assert len(result_1) == 2
    assert len(result_2) == 3
    assert len(result_3) == 3
    assert len(result_4) <= 5  # If fewer than 5 points exist, the result should not exceed them

def test_valid_exact_multi_index_initialisation_with_arguments():
    # should succeed with np.array and normal lists
    index1 = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["euclidean", "euclidean"], weights=[0.3, 0.7])

    assert index1.modalities == 2
    assert index1.dimensions == [1,2]
    assert index1.distance_metrics == ["euclidean", "euclidean"]
    assert np.allclose(index1.weights,[0.3, 0.7])

    index2 = cppindex.ExactMultiIndex(1, dims=np.array([1]), distance_metrics=["euclidean"], weights=[0.3])

    assert index2.modalities == 1
    assert index2.dimensions == [1]
    assert index2.distance_metrics == ["euclidean"]
    assert np.allclose(index2.weights,[0.3])

    # should succeed with different orders
    index3 = cppindex.ExactMultiIndex(weights=[0.3, 0.7], modalities=2, dims=np.array([1, 2]), distance_metrics=["euclidean", "euclidean"])

    assert index3.modalities == 2
    assert index3.dimensions == [1,2]
    assert index3.distance_metrics == ["euclidean", "euclidean"]
    assert np.allclose(index3.weights,[0.3, 0.7])

    # should succeed without passing in weights
    index4 = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["euclidean", "euclidean"])

    assert index4.modalities == 2
    assert index4.dimensions == [1,2]
    assert index4.distance_metrics == ["euclidean", "euclidean"]
    assert np.allclose(index4.weights,[0.5, 0.5])


    index5 = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["euclidean", "euclidean"], weights = None)

    assert index5.modalities == 2
    assert index5.dimensions == [1,2]
    assert index5.distance_metrics == ["euclidean", "euclidean"]
    assert np.allclose(index5.weights,[0.5, 0.5])

def test_immutable_exact_multi_index_attributes():
    index = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["euclidean", "euclidean"], weights=[0.3, 0.7])

    with pytest.raises(AttributeError):
        index.modalities = 3

    with pytest.raises(AttributeError):
        index.dimensions = [1, 3]

    with pytest.raises(AttributeError):
        index.distance_metrics = ["euclidean", "euclidean", "c"]

    with pytest.raises(AttributeError):
        index.weights = [0.3, 0.7, 0.9]

def test_invalid_exact_multi_index_initialization():

    with pytest.raises(TypeError):  # pybind11 raises TypeError for type mismatches
        cppindex.ExactMultiIndex(
            2,
            weights=np.array([1, 2]),
            distance_metrics=["euclidean", "euclidean"],
            dims=[0.5, 0.5]  # incorrect type, expected to be list of ints
        )

    with pytest.raises(TypeError):  # pybind11 raises TypeError for type mismatches
        cppindex.ExactMultiIndex(
            modalities = 2,
            dims=np.array([0.7, 1.2]), # incorrect type, expected to be list of ints
            distance_metrics=["euclidean", "euclidean"],
            weights=[1.0, 1.0]
        )

def test_invalid_item_type_added_to_exact_index():
    index = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["euclidean", "euclidean"], weights=[0.3, 0.7])

    with pytest.raises(RuntimeError, match="Input must be a list or tuple of numpy arrays"):
        index.add_entities(np.array([1,2]))

    with pytest.raises(RuntimeError, match="Subarray must be a 1D or 2D array-like object"):
        index.add_entities((1,2))

    # test adding incorrect number of entities
    with pytest.raises(ValueError):
        index.add_entities([[1,2],[1,2]])


    # test adding incorrect shape
    index = cppindex.ExactMultiIndex(1, dims=np.array([3]), distance_metrics=["euclidean"], weights=[1])
    with pytest.raises(RuntimeError, match="Modality 0 has incorrect data size: 4 is not equal to the expected dimension 3"):
        # adding 3 entities with 4 dimensions (rather than 4 entities with 3 dimensions)
        index.add_entities([[  #single modality so we have 1 outer array
            [1,2,3,4], [5, 6, 7, 8], [9,10,11,12] # modality 1: we are adding 3 vectors, each of dimension 4
        ]])

def test_adding_items_to_index():
    index = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["euclidean", "euclidean"], weights=[0.3, 0.7])

    # add single entity as list of items
    index.add_entities([[1],[1,2]])

    assert index.num_entities == 1

    # adding 2 entities
    index.add_entities([[[1], [2]],
                        [[2,3], [4,5]]])

    assert index.num_entities == 3

    # adding 3 entities
    index.add_entities([
        [[1], [2], [5]],
        [[2,3], [4,5], [6,7]],
    ])

    assert index.num_entities == 6

    # adding 3 entities as np.array
    index.add_entities([
        np.array([[1], [2], [5]]), # modality 1 vectors
        np.array([[2,3], [4,5], [6,7]]), # modality 2 vectors
    ])

    assert index.num_entities == 9

def eq(array1,array2):
    return np.all(np.equal(array1,array2))

def test_searching_exact_multi_index_single_dim_single_modality():

    # test single modality, single vector
    index = cppindex.ExactMultiIndex(1, dims=np.array([1]), distance_metrics=["euclidean"], weights=[1])

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

def test_searching_exact_multi_index_multiple_modalities_single_dim():
    index = cppindex.ExactMultiIndex(1, dims=np.array([3]), distance_metrics=["euclidean"], weights=[1])


    # adding wtih 1 modality, 4 entities with 3 dimensions for the modality
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


def test_searching_exact_multi_index_multiple_modalities_multiple_dim():
    index = cppindex.ExactMultiIndex(2, dims=np.array([2, 3]), distance_metrics=["euclidean", "euclidean"], weights=[0.5,0.5])

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

