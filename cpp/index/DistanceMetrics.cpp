#include "../include/DistanceMetrics.hpp"

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <span>

DistanceMetric stringToDistanceMetric(const std::string& str) {
    // take lowercase of the string
    std::string strLower = str;
    std::transform(strLower.begin(), strLower.end(), strLower.begin(), ::tolower);

    if (strLower == "euclidean" or strLower == "l2") {
        return DistanceMetric::Euclidean;
    }
    if (strLower == "manhattan" or strLower == "l1") {
        return DistanceMetric::Manhattan;
    }
    if (strLower == "cosine") {
        return DistanceMetric::Cosine;
    }
    throw std::invalid_argument("Invalid distance metric: " + str + ". Must be one of 'euclidean', 'manhattan' or 'cosine'");
}

std::string distanceMetricToString(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::Euclidean:
            return "euclidean";
        case DistanceMetric::Manhattan:
            return "manhattan";
        case DistanceMetric::Cosine:
            return "cosine";
    }
    throw std::invalid_argument("Invalid distance metric");
}

float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

//calculate the dot product of two vectors
float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i<a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// compute the Euclidean distance from slices of two vectors
float computeEuclideanDistanceFromSlice(const std::span<const float>& storedEntity, const size_t startIdx, const size_t endIdx,
                               const std::span<const float>& queryEntity, const size_t queryStartIdx) {
    float sum = 0.0f;
    for (size_t idx = startIdx, queryIdx = queryStartIdx; idx < endIdx; ++idx, ++queryIdx) {
        float diff = storedEntity[idx] - queryEntity[queryIdx];
        sum += diff * diff; // Add squared difference
    }
    return std::sqrt(sum);
}

float computeDotProductFromSlice(const std::span<const float> &storedEntity, size_t startIdx, size_t endIdx, const std::span<const float> &queryEntity, size_t queryStartIdx) {
    float sum = 0.0f;
    for (size_t idx = startIdx, queryIdx = queryStartIdx; idx < endIdx; ++idx, ++queryIdx) {
        sum += storedEntity[idx] * queryEntity[queryIdx];
    }
    return sum;
}

// cosine distance ranges from 0 to 2, where 0 is identical and 2 is opposite direction
// we assume it is not 0
float computeCosineDistanceFromSlice(const std::span<const float> &storedEntity, size_t startIdx, size_t endIdx, const std::span<const float> &queryEntity, size_t queryStartIdx, const bool normalised) {
    const float dot_product = computeDotProductFromSlice(storedEntity, startIdx, endIdx, queryEntity, queryStartIdx);
    if (normalised) {
        return 1.0f - dot_product;
    }
    float stored_norm = 0.0f;
    float query_norm = 0.0f;
    for (size_t idx = startIdx, queryIdx = queryStartIdx; idx < endIdx; ++idx, ++queryIdx) {
        stored_norm += storedEntity[idx] * storedEntity[idx];
        query_norm += queryEntity[queryIdx] * queryEntity[queryIdx];
    }
    stored_norm = std::sqrt(stored_norm);
    query_norm = std::sqrt(query_norm);
    return 1.0f - dot_product / (stored_norm * query_norm);
}

float computeManhattanDistanceFromSlice(const std::span<const float>& storedEntity, size_t startIdx, size_t endIdx,
                               const std::span<const float>& queryEntity, size_t queryStartIdx) {
    float sum = 0.0f;
    for (size_t idx = startIdx, queryIdx = queryStartIdx; idx < endIdx; ++idx, ++queryIdx) {
        sum += std::abs(storedEntity[idx] - queryEntity[queryIdx]);
    }
    return sum;
}