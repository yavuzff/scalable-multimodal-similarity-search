import cppindex
import numpy as np

def form_index():
    my_index = cppindex.ExactIndex()
    my_index.add(np.array([3,4]))
    my_index.add(np.array([1,2]))
    my_index.add(np.array([0,0]))
    my_index.add(np.array([-1,-2]))
    my_index.add(np.array([-3,-4]))
    return my_index

index = form_index()

print(index.search(np.array([0,0]),2))
print(index.search(np.array([0,0]),3))
print(index.search(np.array([5,6]),3))
print(index.search(np.array([15,16]),5))
