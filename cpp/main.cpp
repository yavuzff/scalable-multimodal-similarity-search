#include <iostream>

#include "index/ExactKNNIndex.hpp"

int main() {
    ExactKNNIndex index;

    // Add some vectors
    index.add({1.0f, 2.0f});
    index.add({3.0f, 4.0f});
    index.add({5.0f, 6.0f});
    index.add({1.0f, 1.0f});

    // Perform a search
    std::vector<int> neighbours = index.search({1.0f, 1.0f}, 2);

    // Print the results
    std::cout << "Indices of nearest neighbors: ";
    for (int idx : neighbours) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    return 0;
}