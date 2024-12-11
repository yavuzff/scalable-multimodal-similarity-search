#include <iostream>

#include "index/simple-knn/ExactKNNIndex.hpp"
#include "index/ExactMultiIndex.hpp"

void exact_demo() {
    ExactKNNIndex index;

    // Add some vectors
    index.add({1.0f, 2.0f});
    index.add({3.0f, 4.0f});
    index.add({5.0f, 6.0f});
    index.add({1.0f, 1.0f});

    std::vector<int> neighbours = index.search({1.0f, 1.0f}, 2);

    std::cout << "Indices of nearest neighbors: ";
    for (int idx : neighbours) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
}

void multi_exact_demo() {
    // initialise an index
    size_t modalities = 2;
    std::vector<size_t> dims = {1, 2};
    std::vector<std::string> distance_metrics = {"euclidean", "euclidean"};
    std::vector<float> weights = {0.5f, 0.5f};
    ExactMultiIndex index(modalities, dims, distance_metrics);

    // add some entities
    index.add({{1.0f}, {2.0f, 3.0f}});
    index.add({{3.0f}, {4.0f, 5.0f}});
    index.add({{5.0f}, {6.0f, 7.0f}});
    index.add({{1.0f}, {1.0f, 1.0f}});

    // search for nearest neighbors
    //std::vector<size_t> neighbours = index.search({{1.0f}, {1.0f, 1.0f}}, 2,  {0.5f, 0.5f});
    std::vector<size_t> neighbours = index.search({{1.0f}, {1.0f, 1.0f}}, 2);

    std::cout << "Indices of nearest neighbors: ";
    for (size_t idx : neighbours) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
}

int main() {

    multi_exact_demo();

    return 0;
}