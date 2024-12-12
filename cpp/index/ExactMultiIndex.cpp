#include "../include/ExactMultiIndex.hpp"
#include "../include/DistanceMetrics.hpp"

#include <iostream>
#include <queue>

//TODO:
// - add distance metric selection


ExactMultiIndex::ExactMultiIndex(const size_t numModalities,
                                 std::vector<size_t> dims,
                                 std::vector<std::string> distance_metrics,
                                 std::vector<float> weights)
    : AbstractMultiIndex(numModalities, std::move(dims), std::move(distance_metrics), std::move(weights)) {
    storedEntities.resize(numModalities);
}

void ExactMultiIndex::addEntities(const std::vector<std::vector<float>> &entities) {
    size_t numNewEntities = validateEntities(entities);
    numEntities += numNewEntities;
    std::cout << "Adding " << numNewEntities << " entities!" << std::endl;

    // add each modality's data to the corresponding storedEntities modality vector
    // note that we copy the input into savedEntities
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];
        storedEntities[i].insert(storedEntities[i].end(), modalityVectors.begin(), modalityVectors.end());
    }
}

std::vector<size_t> ExactMultiIndex::search(const std::vector<std::vector<float>>& query, const size_t k,
                        const std::vector<float>& query_weights) {
    // validate the inputs
    size_t numNewEntities = validateEntities(query);
    if (numNewEntities != 1) {
        throw std::invalid_argument("Query must contain exactly one entity, but got " + std::to_string(numNewEntities));
    }

    // copy weights as we will normalise them
    auto normalised_query_weights = std::vector(query_weights);
    validateAndNormaliseWeights(normalised_query_weights, numModalities);

    // iterate over entities through modality vectors
    std::priority_queue<std::pair<float, size_t>> maxHeap;
    for (size_t i = 0; i < numEntities; ++i) {
        float dist = 0.0f;
        for (size_t modality = 0; modality < numModalities; ++modality) {
            const size_t vectorStart = i * dimensions[modality];
            const size_t vectorEnd = vectorStart + dimensions[modality];

            // compute Euclidean distance between storedEntities[modality][vectorStart:vectorEnd] and query[modality]
            const float modalityDistance = computeEuclideanDistanceFromSlice(storedEntities[modality], vectorStart, vectorEnd, query[modality], 0);
            dist += normalised_query_weights[modality] * modalityDistance;
        }

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
    std::cout << "Index properties: " << numModalities << ", Num Entities: " << numEntities << std::endl;
}

void ExactMultiIndex::load(const std::string& path) {
    std::cout << "Loading index from " << path << std::endl;
}

//private function to validate an input entity and return the number of entities
size_t ExactMultiIndex::validateEntities(const std::vector<std::vector<float>>& entities) const {
    if (entities.size() != numModalities) {
        throw std::invalid_argument("Entity must have the same number of modalities as the index");
    }

    std::optional<size_t> numNewEntities;
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];

        // check that modality vectors is a multiple of the dimension count
        if (modalityVectors.size() % dimensions[i] != 0) {
            throw std::invalid_argument(
                "Modality " + std::to_string(i) + " has incorrect data size: " +
                std::to_string(modalityVectors.size()) + " is not a multiple of the expected dimension " + std::to_string(dimensions[i])
                );
        }

        // check that modality vectors contains the same number of entities
        size_t numEntitiesThisModality = modalityVectors.size() / dimensions[i];
        if (numNewEntities.has_value()) {
            if (numEntitiesThisModality != numNewEntities.value()) {
                throw std::invalid_argument("Modality " + std::to_string(i) + " has a different number of entities than the other modalities, expected " + std::to_string(numNewEntities.value()) + " but got " + std::to_string(numEntitiesThisModality));
            }
        } else {
            // this is the first modality, so set the number of entities
            numNewEntities = numEntitiesThisModality;
        }
    }
    return numNewEntities.value();
}
