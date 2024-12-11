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
    cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["a", "b"], weights=[0.3, 0.7])

    cppindex.ExactMultiIndex(1, dims=np.array([1]), distance_metrics=["a"], weights=[0.3])

    # should succeed with different orders
    cppindex.ExactMultiIndex(weights=[0.3, 0.7], modalities=2, dims=np.array([1, 2]), distance_metrics=["a", "b"])

    # should succeed without passing in weights
    cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["a", "b"])

    cppindex.ExactMultiIndex(2, dims=np.array([1, 2]), distance_metrics=["a", "b"], weights = None)

    assert True

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
