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
    index1 = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["a", "b"], weights=[0.3, 0.7])

    assert index1.modalities == 2
    assert index1.dimensions == [1,2]
    assert index1.distance_metrics == ["a", "b"]
    assert np.allclose(index1.weights,[0.3, 0.7])

    index2 = cppindex.ExactMultiIndex(1, dims=np.array([1]), distance_metrics=["a"], weights=[0.3])

    assert index2.modalities == 1
    assert index2.dimensions == [1]
    assert index2.distance_metrics == ["a"]
    assert np.allclose(index2.weights,[0.3])

    # should succeed with different orders
    index3 = cppindex.ExactMultiIndex(weights=[0.3, 0.7], modalities=2, dims=np.array([1, 2]), distance_metrics=["a", "b"])

    assert index3.modalities == 2
    assert index3.dimensions == [1,2]
    assert index3.distance_metrics == ["a", "b"]
    assert np.allclose(index3.weights,[0.3, 0.7])

    # should succeed without passing in weights
    index4 = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["a", "b"])

    assert index4.modalities == 2
    assert index4.dimensions == [1,2]
    assert index4.distance_metrics == ["a", "b"]
    assert np.allclose(index4.weights,[1.0, 1.0])


    index5 = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["a", "b"], weights = None)

    assert index5.modalities == 2
    assert index5.dimensions == [1,2]
    assert index5.distance_metrics == ["a", "b"]
    assert np.allclose(index5.weights,[1.0, 1.0])

def test_immutable_exact_multi_index_attributes():
    index = cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["a", "b"], weights=[0.3, 0.7])

    with pytest.raises(AttributeError):
        index.modalities = 3

    with pytest.raises(AttributeError):
        index.dimensions = [1, 3]

    with pytest.raises(AttributeError):
        index.distance_metrics = ["a", "b", "c"]

    with pytest.raises(AttributeError):
        index.weights = [0.3, 0.7, 0.9]

def test_invalid_exact_multi_index_initialization():

    with pytest.raises(TypeError):  # pybind11 raises TypeError for type mismatches
        cppindex.ExactMultiIndex(
            2,
            weights=np.array([1, 2]),
            distance_metrics=["a", "b"],
            dims=[0.5, 0.5]  # incorrect type, expected to be list of ints
        )

    with pytest.raises(TypeError):  # pybind11 raises TypeError for type mismatches
        cppindex.ExactMultiIndex(
            modalities = 2,
            dims=np.array([0.7, 1.2]), # incorrect type, expected to be list of ints
            distance_metrics=["a", "b"],
            weights=[1.0, 1.0]
        )
