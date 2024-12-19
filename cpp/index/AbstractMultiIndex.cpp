#include "../include/AbstractMultiIndex.hpp"

#include <iostream>
#include <numeric>
#include <algorithm>

// function to validate and normalise weights
void validateAndNormaliseWeights(std::vector<float>& ws, const size_t numModalities) {
    if (ws.size() != numModalities) {
        throw std::invalid_argument("Number of weights must match number of modalities");
    }
    // check weights are non-negative
    if (std::ranges::any_of(ws, [](float weight) { return weight < 0; })) {
        throw std::invalid_argument("Weights must be non-negative");
    }
    // normalise weights
    float sum = std::accumulate(ws.begin(), ws.end(), 0.0f);
    if (sum == 0) {
        throw std::invalid_argument("Weights must not be all zero");
    }
    for (size_t i = 0; i < numModalities; ++i) {
        ws[i] /= sum;
    }
}


// Constructor: take parameters in by value to gain ownership
AbstractMultiIndex::AbstractMultiIndex(size_t the_modalities,
        std::vector<size_t> dims,
        std::vector<std::string> dist_metrics,
        std::vector<float> ws)
        : numModalities(the_modalities), dimensions(std::move(dims)), str_distance_metrics(dist_metrics), weights(std::move(ws)) {
        if (numModalities == 0) {
            throw std::invalid_argument("Number of modalities must be positive");
        }
        if (dimensions.size() != numModalities) {
            throw std::invalid_argument("Number of dimensions must match number of modalities");
        }

        // initialise distance metrics if not provided, otherwise validate and convert to enum
        if (dist_metrics.empty()) {
            distance_metrics.resize(numModalities, DistanceMetric::Euclidean);
            str_distance_metrics.resize(numModalities, "euclidean");
        } else if (dist_metrics.size() != numModalities) {
            throw std::invalid_argument("Number of distance metrics must match number of modalities");
        } else {
            std::transform(dist_metrics.begin(), dist_metrics.end(), std::back_inserter(distance_metrics), &stringToDistanceMetric);
        }

        // initialise weights if not provided
        if (weights.empty()) {
            weights.resize(numModalities, 1.0f / numModalities);
        } else {
            validateAndNormaliseWeights(weights, numModalities);
        }

        // print out what we just initialised:
        std::cout << "Created MultiIndex with " << numModalities << " modalities" << std::endl;
        for (size_t i = 0; i < numModalities; ++i) {
            std::cout << "Modality " << i << " has dimension " << dimensions[i] << ", distance metric " << distanceMetricToString(distance_metrics[i]) << " and weight " << weights[i] << std::endl;
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
        return str_distance_metrics;
    }

    [[nodiscard]] const std::vector<float>& AbstractMultiIndex::getWeights() const {
        return weights;
    }

    [[nodiscard]] size_t AbstractMultiIndex::getNumEntities() const {
        return numEntities;
    }
