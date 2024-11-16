//
// Created by Yavuz on 16/11/2024.
//

#include "ExactKNNIndex.hpp"

// #ifndef DEBUG
// #define DEBUG 0
// #endif
// #define debug_printf(fmt, ...) \
// do { if (DEBUG) { fprintf(stderr, fmt, __VA_ARGS__); \
// fflush(stderr); } } while (0)


#include <cassert>
#include <queue>

// Function to calculate the Euclidean distance between two vectors
static float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

//calculate the dot product of two vectors
static float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i<a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}


ExactKNNIndex::~ExactKNNIndex() = default;

// Add a vector to the index
void ExactKNNIndex::add(const std::vector<float>& vector) {
    data.push_back(vector);
}


// Perform a k-nearest neighbor search using a priority queue
std::vector<int> ExactKNNIndex::search(const std::vector<float>& query, size_t k) const {
    assert(k <= data.size());

    // A max-heap priority queue to store the k-nearest neighbors. Pair: <distance, index>.
    std::priority_queue<std::pair<float, int>> maxHeap;

    for (size_t i = 0; i < data.size(); i++) {
        float dist = euclideanDistance(query, data[i]);

        if (maxHeap.size() < k) {
            // Add directly if the heap isn't full
            maxHeap.emplace(dist, i);
        } else if (dist < maxHeap.top().first) {
            // Replace the largest distance if the new distance is smaller
            maxHeap.pop();
            maxHeap.emplace(dist, i);
        }
    }

    // Extract the indices from the heap
    std::vector<int> result;
    while (!maxHeap.empty()) {
        result.push_back(maxHeap.top().second);
        maxHeap.pop();
    }

    // The indices will be in reverse order due to the max-heap, so reverse them
    std::reverse(result.begin(), result.end());
    return result;
}