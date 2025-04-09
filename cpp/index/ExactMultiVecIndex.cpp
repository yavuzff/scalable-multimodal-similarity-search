#include "../include/ExactMultiVecIndex.hpp"
#include "../include/utils.hpp"
#include "../include/common.hpp"

#include <iostream>
#include <queue>
#include <span>


ExactMultiVecIndex::ExactMultiVecIndex(const size_t numModalities,
                                 std::vector<size_t> dims,
                                 std::vector<std::string> distanceMetrics,
                                 std::vector<float> weights)
    : AbstractMultiVecIndex(numModalities, std::move(dims), std::move(distanceMetrics), std::move(weights)) {
    storedEntities.resize(numModalities);
}

void ExactMultiVecIndex::addEntities(const std::vector<std::vector<float>> &entities) {
    addEntities(getSpanViewOfVectors(entities));
}

void ExactMultiVecIndex::addEntities(const std::vector<std::span<const float>>& entities) {
    const size_t numNewEntities = validateEntities(entities, dimensions);
    numEntities += numNewEntities;
    debug_printf("Adding %zu entities!\n", numNewEntities);

    // add each modality's data to the corresponding storedEntities modality vector
    // note that we copy the input into savedEntities
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];
        storedEntities[i].insert(storedEntities[i].end(), modalityVectors.begin(), modalityVectors.end());

        // if the modality uses cosine distance, normalise the inserted vectors for this modality
        if (distanceMetrics[i] == DistanceMetric::Cosine) {
            debug_printf("ExactMultiVecIndex: Storing normalised vectors for modality %zu to efficiently compute cosine distance\n", i);
            for (size_t j = 0; j < numNewEntities; ++j) {
                l2NormalizeVector(std::span(storedEntities[i]).subspan(j * dimensions[i], dimensions[i]));
            }
        }
    }
}

std::vector<size_t> ExactMultiVecIndex::search(const std::vector<std::span<const float>>& query, const size_t k,
                        const std::vector<float>& queryWeights) {
    validateQuery(query, k, dimensions);
    // copy weights as we will normalise them
    auto normalisedQueryWeights = std::vector(queryWeights);
    validateAndNormaliseWeights(normalisedQueryWeights, numModalities);

    return internalSearch(query, k, normalisedQueryWeights);
}

std::vector<size_t> ExactMultiVecIndex::search(const std::vector<std::span<const float>>& query, const size_t k) {
    validateQuery(query, k, dimensions);
    return internalSearch(query, k, indexWeights);
}

std::vector<size_t> ExactMultiVecIndex::search(const std::vector<std::vector<float>>& query, const size_t k,
                        const std::vector<float>& queryWeights) {
    return search(getSpanViewOfVectors(query), k, queryWeights);
}

std::vector<size_t> ExactMultiVecIndex::search(const std::vector<std::vector<float>>& query, const size_t k) {
    return search(getSpanViewOfVectors(query), k);
}

std::vector<size_t> ExactMultiVecIndex::internalSearch(const std::vector<std::span<const float>>& userQuery, const size_t k,
                        const std::vector<float>& normalisedWeights) const {

    std::vector<std::span<const float>> query = userQuery;

    // storage for normalized query vectors to ensure valid spans
    // this storage must be defined outside the if block to ensure the data is in scope throughout the search
    std::vector<std::vector<float>> normalisedVectors;

    if (!toNormalise.empty()) {
        std::vector<std::span<const float>> normalisedQuery;
        for (size_t i = 0; i < numModalities; ++i) {
            if (distanceMetrics[i] == DistanceMetric::Cosine) {
                debug_printf("Normalising query for modality %zu to efficiently compute cosine distance\n", i);
                // copy and normalize vector
                normalisedVectors.emplace_back(query[i].begin(), query[i].end());
                l2NormalizeVector(std::span(normalisedVectors.back()));

                normalisedQuery.emplace_back(std::span(normalisedVectors.back()));
            } else {
                normalisedQuery.push_back(query[i]);
            }
        }
        query = std::move(normalisedQuery);
    }

    // iterate over entities through modality vectors
    std::priority_queue<std::pair<float, size_t>> maxHeap;
    for (size_t i = 0; i < numEntities; ++i) {
        float dist = 0.0f;
        for (size_t modality = 0; modality < numModalities; ++modality) {
            // create spans for the stored entity and the query
            std::span<const float> storedEntitySlice = std::span(storedEntities[modality]).subspan( i * dimensions[modality], dimensions[modality]);
            std::span<const float> querySlice = std::span(query[modality]);

            float modalityDistance;
            // compute distance based on distance_metric for this modality
            // distance is computed between storedEntities[modality][vectorStart:vectorEnd] and query[modality]
            switch(distanceMetrics[modality]){
                case DistanceMetric::Euclidean:
                    modalityDistance = euclidean(storedEntitySlice, querySlice);
                    break;
                case DistanceMetric::Manhattan:
                    modalityDistance = manhattan(storedEntitySlice, querySlice);
                    break;
                case DistanceMetric::Cosine:
                    // vectors are already normalised
                    modalityDistance = cosine(storedEntitySlice, querySlice);
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

void ExactMultiVecIndex::save(const std::string& path) const {
    throw std::runtime_error("ExactMultiVecIndex::save not implemented. You can simply rebuild the index from the data, instead of saving to " + path);
}

void ExactMultiVecIndex::load(const std::string& path) {
    throw std::runtime_error("ExactMultiVecIndex::load not implemented. You can simply rebuild the index from the data, instead of loading from " + path);
}

void ExactMultiVecIndex::outputEntities() const {
    // print entities one by one
    for (size_t i = 0; i < numEntities; ++i) {
        std::cout << "Entity " << i << ": ";
        for (size_t j = 0; j < numModalities; ++j) {
            std::cout << "Modality " << j << ": ";
            for (size_t k = 0; k < dimensions[j]; ++k) {
                std::cout << storedEntities[j][i * dimensions[j] + k] << " ";
            }
            std::cout << std::endl;
        }
    }
}