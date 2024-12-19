#include "../../include/simple-knn/ExactKNNIndex.hpp"
#include "../../common.hpp"
#include "../../include/DistanceMetrics.hpp"

#include <cassert>
#include <queue>

ExactKNNIndex::~ExactKNNIndex() = default;

void ExactKNNIndex::add(const std::vector<float>& vector) {
    debug_printf("Adding vector of size %lu\n", vector.size());
    data.push_back(vector);
}

// linear search using priority queue
std::vector<size_t> ExactKNNIndex::search(const std::vector<float>& query, size_t k) const {
    assert(k <= data.size());

    // max-heap priority queue to store the k-nearest neighbors. Pair: <distance, index>.
    std::priority_queue<std::pair<float, size_t>> maxHeap;

    for (size_t i = 0; i < data.size(); i++) {
        float dist = euclideanDistance(query, data[i]);

        // add directly if the heap isn't full, otherwise replace largest distance item
        if (maxHeap.size() < k) {
            maxHeap.emplace(dist, i);
        } else if (dist < maxHeap.top().first) {
            maxHeap.pop();
            maxHeap.emplace(dist, i);
        }
    }

    // extract indices from priority queue
    std::vector<size_t> result;
    while (!maxHeap.empty()) {
        result.push_back(maxHeap.top().second);
        maxHeap.pop();
    }

    // return in increasing distance order
    std::reverse(result.begin(), result.end());
    return result;
}