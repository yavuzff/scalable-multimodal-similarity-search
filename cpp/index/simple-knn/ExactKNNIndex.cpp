#include "ExactKNNIndex.hpp"
#include "../../common.hpp"
#include "../DistanceMetrics.hpp"

#include <cassert>
#include <queue>

ExactKNNIndex::~ExactKNNIndex() = default;

void ExactKNNIndex::add(const std::vector<float>& vector) {
    debug_printf("Adding vector of size %lu\n", vector.size());
    data.push_back(vector);
}

// linear search using priority queue
std::vector<int> ExactKNNIndex::search(const std::vector<float>& query, size_t k) const {
    assert(k <= data.size());

    // max-heap priority queue to store the k-nearest neighbors. Pair: <distance, index>.
    std::priority_queue<std::pair<float, int>> maxHeap;

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
    std::vector<int> result;
    while (!maxHeap.empty()) {
        result.push_back(maxHeap.top().second);
        maxHeap.pop();
    }

    // return in increasing distance order
    std::ranges::reverse(result);
    return result;
}