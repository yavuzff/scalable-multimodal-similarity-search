#include "ExactMultiIndex.hpp"
#include "DistanceMetrics.hpp"

#include <iostream>
#include <queue>

//TODO:
// - integrate with pybind
// - add distance metric selection
// - validate weights length etc. in constructor, have optional weights
// - add optional weights usage in search (so would default to index weights)
// Lower priority:
// - add multiple entity addition? check with pybind
// - return vectors from search? Or add method to retrieve vector with certain index?
// - add save and load index


//private function to validate an input entity
void ExactMultiIndex::validateEntity(const std::vector<std::vector<float>>& entity) const {
    if (entity.size() != modalities) {
        throw std::invalid_argument("Entity must have the same number of modalities as the index");
    }
    for (size_t i = 0; i < modalities; ++i) {
        if (entity[i].size() != dimensions[i]) {
            throw std::invalid_argument("Each vector in the entity must have the correct dimension");
        }
    }
}

void ExactMultiIndex::add(const std::vector<std::vector<float>>& entity) {
    validateEntity(entity);
    entities.push_back(entity);
}

std::vector<size_t> ExactMultiIndex::search(const std::vector<std::vector<float>>& query, const size_t k,
                        const std::vector<float>& query_weights) {
    validateEntity(query);

    std::priority_queue<std::pair<float, size_t>> maxHeap;

    for (size_t i = 0; i < entities.size(); i++) {
        float dist = weightedEuclideanDistance(query, entities[i], query_weights);

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
    std::ranges::reverse(result);
    return result;
}

std::vector<size_t> ExactMultiIndex::search(const std::vector<std::vector<float>>& query, size_t k) {
    return search(query, k, weights);
}

void ExactMultiIndex::save(const std::string& path) const {
    std::cout << "Saving index to " << path << std::endl;
}

void ExactMultiIndex::load(const std::string& path) {
    std::cout << "Loading index from " << path << std::endl;
}