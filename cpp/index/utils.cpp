#include "include/utils.hpp"

#include <numeric>
#include <algorithm>
#include <cmath> //for sqrt

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

// get span view of vectors
std::vector<std::span<const float>> getSpanViewOfVectors(const std::vector<std::vector<float>> &vectors) {
    // convert std::vector<std::vector<float>> to std::vector<std::span<float>>
    std::vector<std::span<const float>> entitiesAsSpans;
    for (const auto& modality : vectors) {
        // Use std::span<const float> because modality is const
        std::span<const float> span(modality.data(), modality.size());
        entitiesAsSpans.push_back(span);
    }
    return entitiesAsSpans;
}

// normalise vector in-place so that the l2-norm is 1
void l2NormalizeVector(std::span<float> vector) {
    float sum = 0;
    for (float val : vector) {
        sum += val * val;
    }
    float norm = std::sqrt(sum);

    if (norm == 0) {
        throw std::invalid_argument("One or both input vectors have zero magnitude");
    }
    for (float& val : vector) {
        val /= norm;
    }
}
