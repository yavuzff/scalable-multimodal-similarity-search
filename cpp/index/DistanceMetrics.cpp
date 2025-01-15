#include "../include/DistanceMetrics.hpp"

#include <cassert>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <span>
#include <Eigen/Dense>

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


float computeEuclideanPureEigen(const Eigen::VectorXf& vector1, const Eigen::VectorXf& vector2) {
    assert(vector1.size() == vector2.size());
    return (vector1 - vector2).norm();
}

float computeEuclideanEigenMap(const std::span<const float> &vector1, const std::span<const float> &vector2) {
    assert(vector1.size() == vector2.size());
    Eigen::Map<const Eigen::VectorXf> v1(vector1.data(), vector1.size());
    Eigen::Map<const Eigen::VectorXf> v2(vector2.data(), vector2.size());
    return (v1 - v2).norm();
}


#include <cstddef>
float computeEuclideanDistancePragma(const std::span<const float> &vectorSlice1, const std::span<const float> &vectorSlice2) {
    assert(vectorSlice1.size() == vectorSlice2.size());
    float sum = 0.0f;
    // OpenMP SIMD pragma for vectorization
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < vectorSlice1.size(); ++i) {
        const float diff = vectorSlice1[i] - vectorSlice2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}


#include <execution>
#include <numeric>
float computeEuclideanDistanceParUnseq(const std::span<const float>& vectorSlice1, const std::span<const float> &vectorSlice2) {
    assert(vectorSlice1.size() == vectorSlice2.size());

    // compute sum of squared differences in parallel
    float squaredSum = std::transform_reduce(
        //std::execution::par_unseq,
        vectorSlice1.begin(), vectorSlice1.end(),
        vectorSlice2.begin(),
        0.0f,                         // initial value for the reduction
        std::plus<>(),
        [](float a, float b) {
            return (a - b) * (a - b);
        }
    );
    return std::sqrt(squaredSum);
}

// compute the Euclidean distance from slices of two vectors
float computeEuclideanDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2) {
    assert(vectorSlice1.size() == vectorSlice2.size());
    float sum = 0.0f;
    for (size_t i = 0; i< vectorSlice1.size(); i++) {
        float diff = vectorSlice1[i] - vectorSlice2[i];
        sum += diff * diff; // Add squared difference
    }
    return std::sqrt(sum);
}

float computeDotProduct(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2) {
    float sum = 0.0f;
    for (size_t i = 0; i< vectorSlice1.size(); i++) {
        sum += vectorSlice1[i] * vectorSlice2[i];
    }
    return sum;
}

// cosine distance ranges from 0 to 2, where 0 is identical and 2 is opposite direction
// we assume it is not 0
float computeCosineDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2, bool normalised) {
    assert(vectorSlice1.size() == vectorSlice2.size());
    const float dot_product = computeDotProduct(vectorSlice1, vectorSlice2);
    if (normalised) {
        return 1.0f - dot_product;
    }
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    for (size_t i = 0; i< vectorSlice1.size(); i++) {
        norm1 += vectorSlice1[i] * vectorSlice1[i];
        norm2 += vectorSlice2[i] * vectorSlice2[i];
    }
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    // Check for zero vectors to avoid division by zero
    if (norm1 == 0.0f || norm2 == 0.0f) {
        throw std::invalid_argument("One or both input vectors have zero magnitude");
    }
    return 1.0f - dot_product / (norm1 * norm2);
}

float computeManhattanDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2) {
    assert(vectorSlice1.size() == vectorSlice2.size());

    float sum = 0.0f;
    for (size_t i = 0; i< vectorSlice1.size(); i++) {
        sum += std::abs(vectorSlice1[i] - vectorSlice2[i]);
    }
    return sum;
}

// methods operating on std::vector

float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i<a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}