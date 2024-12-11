import cppindex
import numpy as np

def form_index():
    my_index = cppindex.ExactIndex()
    my_index.add(np.array([3,4]))
    my_index.add(np.array([1,2]))
    my_index.add(np.array([0,0]))
    my_index.add(np.array([-1,-2]))
    my_index.add(np.array([-3,-4]))

    print(my_index.search(np.array([0,0]),2))
    print(my_index.search(np.array([0,0]),3))
    print(my_index.search(np.array([5,6]),3))
    print(my_index.search(np.array([15,16]),5))


def form_exact_multi_index():
    index1 = cppindex.ExactMultiIndex()
    index2 = cppindex.ExactMultiIndex(2, dims=np.array([1,2]), distance_metrics=["a","b"], weights=[0.5,0.5])

    index2.save("mypath")

form_index()
form_exact_multi_index()