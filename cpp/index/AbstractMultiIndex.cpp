#include "../include/AbstractMultiIndex.hpp"
#include "../include/utils.hpp"

#include <iostream>
#include <optional>

// Constructor: take parameters in by value to gain ownership
AbstractMultiIndex::AbstractMultiIndex(size_t theModalities,
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

        // print out what we just initialised:
        std::cout << "Created MultiIndex with " << numModalities << " modalities" << std::endl;
        for (size_t i = 0; i < numModalities; ++i) {
            std::cout << "Modality " << i << " has dimension " << dimensions[i] << ", distance metric " << distanceMetricToString(distanceMetrics[i]) << " and weight " << indexWeights[i] << std::endl;
        }
    }

//private function to validate input entities and return the number of entities
size_t AbstractMultiIndex::validateEntities(const std::vector<std::span<const float>>& entities) const {
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

void AbstractMultiIndex::validateQuery(const std::vector<std::span<const float>> &query, size_t k) const {
    // validate the query entity and k
    if (k < 1) {
        throw std::invalid_argument("k must be at least 1");
    }
    size_t numNewEntities = validateEntities(query);
    if (numNewEntities != 1) {
        throw std::invalid_argument("Query must contain exactly one entity, but got " + std::to_string(numNewEntities));
    }
}

    // getter implementations
    [[nodiscard]] size_t AbstractMultiIndex::getNumModalities() const {
        return numModalities;
    }

    [[nodiscard]] const std::vector<size_t>& AbstractMultiIndex::getDimensions() const {
        return dimensions;
    }

    [[nodiscard]] const std::vector<std::string>& AbstractMultiIndex::getDistanceMetrics() const {
        return strDistanceMetrics;
    }

    [[nodiscard]] const std::vector<float>& AbstractMultiIndex::getWeights() const {
        return indexWeights;
    }

    [[nodiscard]] size_t AbstractMultiIndex::getNumEntities() const {
        return numEntities;
    }
