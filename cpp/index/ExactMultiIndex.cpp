#include "../include/ExactMultiIndex.hpp"

#include <iostream>
#include <queue>
#include <span>


ExactMultiIndex::ExactMultiIndex(const size_t numModalities,
                                 std::vector<size_t> dims,
                                 std::vector<std::string> distanceMetrics,
                                 std::vector<float> weights)
    : AbstractMultiIndex(numModalities, std::move(dims), std::move(distanceMetrics), std::move(weights)) {
    storedEntities.resize(numModalities);
}

void ExactMultiIndex::addEntities(const std::vector<std::vector<float>> &entities) {
    addEntities(getSpanViewOfVectors(entities));
}

void ExactMultiIndex::addEntities(const std::vector<std::span<const float>>& entities) {
    const size_t numNewEntities = validateEntities(entities);
    numEntities += numNewEntities;
    std::cout << "Adding " << numNewEntities << " entities!" << std::endl;

    // add each modality's data to the corresponding storedEntities modality vector
    // note that we copy the input into savedEntities
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];
        storedEntities[i].insert(storedEntities[i].end(), modalityVectors.begin(), modalityVectors.end());
    }
}

std::vector<size_t> ExactMultiIndex::search(const std::vector<std::span<const float>>& query, const size_t k,
                        const std::vector<float>& queryWeights) {
    validateQuery(query, k);

    // copy weights as we will normalise them
    auto normalisedQueryWeights = std::vector(queryWeights);
    validateAndNormaliseWeights(normalisedQueryWeights, numModalities);

    return internalSearch(query, k, normalisedQueryWeights);
}

std::vector<size_t> ExactMultiIndex::search(const std::vector<std::span<const float>>& query, const size_t k) {
    validateQuery(query, k);
    return internalSearch(query, k, weights);
}


std::vector<size_t> ExactMultiIndex::search(const std::vector<std::vector<float>>& query, const size_t k,
                        const std::vector<float>& queryWeights) {
    return search(getSpanViewOfVectors(query), k, queryWeights);
}

std::vector<size_t> ExactMultiIndex::search(const std::vector<std::vector<float>>& query, const size_t k) {
    return search(getSpanViewOfVectors(query), k);
}

std::vector<size_t> ExactMultiIndex::internalSearch(const std::vector<std::span<const float>>& query, const size_t k,
                        const std::vector<float>& normalisedWeights) const {
    // iterate over entities through modality vectors
    std::priority_queue<std::pair<float, size_t>> maxHeap;
    for (size_t i = 0; i < numEntities; ++i) {
        float dist = 0.0f;
        for (size_t modality = 0; modality < numModalities; ++modality) {
            const size_t vectorStart = i * dimensions[modality];
            const size_t vectorEnd = vectorStart + dimensions[modality];

            float modalityDistance;
            // compute distance based on distance_metric for this modality
            // distance is computed between storedEntities[modality][vectorStart:vectorEnd] and query[modality]
            switch(distanceMetrics[modality]){
                case DistanceMetric::Euclidean:
                    modalityDistance = computeEuclideanDistanceFromSlice(storedEntities[modality], vectorStart, vectorEnd, query[modality], 0);
                    break;
                case DistanceMetric::Manhattan:
                    modalityDistance = computeManhattanDistanceFromSlice(storedEntities[modality], vectorStart, vectorEnd, query[modality], 0);
                    break;
                case DistanceMetric::Cosine:
                    modalityDistance = computeCosineDistanceFromSlice(storedEntities[modality], vectorStart, vectorEnd, query[modality], 0);
                    break;
                default:
                    throw std::invalid_argument("Invalid distance metric. You should not be seeing this message.");
            }

            // aggregate distance by summing modalityDistance*weight
            dist += normalisedWeights[modality] * modalityDistance;
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
    std::reverse(result.begin(), result.end());
    return result;
}

void ExactMultiIndex::save(const std::string& path) const {
    std::cout << "Saving index to " << path << std::endl;
    std::cout << "Index properties: " << numModalities << ", Num Entities: " << numEntities << std::endl;
}

void ExactMultiIndex::load(const std::string& path) {
    std::cout << "Loading index from " << path << std::endl;
}

