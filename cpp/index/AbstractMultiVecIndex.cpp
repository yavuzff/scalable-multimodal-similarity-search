#include "../include/AbstractMultiVecIndex.hpp"
#include "../include/utils.hpp"

#include <iostream>
#include <numeric>
#include <optional>

#include "../include/common.hpp"

// Constructor: take parameters in by value to gain ownership
AbstractMultiVecIndex::AbstractMultiVecIndex(size_t theModalities,
        std::vector<size_t> dims,
        std::vector<std::string> distMetrics,
        std::vector<float> ws)
        : numModalities(theModalities), dimensions(std::move(dims)), strDistanceMetrics(distMetrics), indexWeights(std::move(ws)) {
    if (numModalities == 0) {
        throw std::invalid_argument("Number of modalities must be positive");
    }
    if (dimensions.size() != numModalities) {
        throw std::invalid_argument("Number of dimensions must match number of modalities");
    }

    // initialise distance metrics if not provided, otherwise validate and convert to enum
    if (distMetrics.empty()) {
        distanceMetrics.resize(numModalities, DistanceMetric::Euclidean);
        strDistanceMetrics.resize(numModalities, "euclidean");
    } else if (distMetrics.size() != numModalities) {
        throw std::invalid_argument("Number of distance metrics must match number of modalities");
    } else {
        std::transform(distMetrics.begin(), distMetrics.end(), std::back_inserter(distanceMetrics), &stringToDistanceMetric);
    }

    // we will normalise vectors for cosine distance
    for (size_t i = 0; i < numModalities; ++i) {
        if (distanceMetrics[i] == DistanceMetric::Cosine) {
            toNormalise.push_back(i);
        }
    }

    // initialise weights if not provided
    if (indexWeights.empty()) {
        indexWeights.resize(numModalities, 1.0f / numModalities);
    } else {
        validateAndNormaliseWeights(indexWeights, numModalities);
    }

    // calculate total dimensions
    totalDimensions = std::accumulate(dimensions.begin(), dimensions.end(), 0);

    // print out what we just initialised:
    debug_printf("Created MultiVecIndex with %zu modalities\n", numModalities);
    for (size_t i = 0; i < numModalities; ++i) {
        debug_printf("Modality %zu has dimension %zu, distance metric %s and weight %f\n", i, dimensions[i], distanceMetricToString(distanceMetrics[i]).c_str(), indexWeights[i]);
    }
    }

bool AbstractMultiVecIndex::operator==(const AbstractMultiVecIndex& other) const {
    return numModalities == other.numModalities &&
           dimensions == other.dimensions &&
           distanceMetrics == other.distanceMetrics &&
           strDistanceMetrics == other.strDistanceMetrics &&
           indexWeights == other.indexWeights &&
           numEntities == other.numEntities &&
           toNormalise == other.toNormalise &&
           totalDimensions == other.totalDimensions;
}

//private function to validate input entities and return the number of entities
size_t AbstractMultiVecIndex::validateEntities(const std::vector<std::span<const float>>& entities, const std::vector<size_t>& expectedDimensions) const {
    if (entities.size() != numModalities) {
        throw std::invalid_argument("Entity must have the same number of modalities as the index");
    }

    std::optional<size_t> numNewEntities;
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];

        // check that modality vectors is a multiple of the dimension count
        if (modalityVectors.size() % expectedDimensions[i] != 0) {
            throw std::invalid_argument(
                "Modality " + std::to_string(i) + " has incorrect data size: " +
                std::to_string(modalityVectors.size()) + " is not a multiple of the expected dimension " + std::to_string(expectedDimensions[i])
                );
        }

        // check that modality vectors contains the same number of entities
        size_t numEntitiesThisModality = modalityVectors.size() / expectedDimensions[i];
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

void AbstractMultiVecIndex::validateQuery(const std::vector<std::span<const float>> &query, size_t k, const std::vector<size_t>& expectedDimensions) const {
    // validate the query entity and k
    if (k < 1) {
        throw std::invalid_argument("k must be at least 1");
    }
    size_t numNewEntities = validateEntities(query, expectedDimensions);
    if (numNewEntities != 1) {
        throw std::invalid_argument("Query must contain exactly one entity, but got " + std::to_string(numNewEntities));
    }
}

    // getter implementations
    [[nodiscard]] size_t AbstractMultiVecIndex::getNumModalities() const {
        return numModalities;
    }

    [[nodiscard]] const std::vector<size_t>& AbstractMultiVecIndex::getDimensions() const {
        return dimensions;
    }

    [[nodiscard]] const std::vector<std::string>& AbstractMultiVecIndex::getDistanceMetrics() const {
        return strDistanceMetrics;
    }

    [[nodiscard]] const std::vector<float>& AbstractMultiVecIndex::getWeights() const {
        return indexWeights;
    }

    [[nodiscard]] size_t AbstractMultiVecIndex::getNumEntities() const {
        return numEntities;
    }
