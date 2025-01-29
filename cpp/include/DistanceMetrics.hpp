#ifndef DISTANCEMETRICS_HPP
#define DISTANCEMETRICS_HPP

#include <vector>
#include <string>
#include <span>
#include <Eigen/Dense>
#include <cassert>

enum class DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
};

DistanceMetric stringToDistanceMetric(const std::string& str);
std::string distanceMetricToString(DistanceMetric metric);

// vectorised implementations with no overhead to get an Eigen value
inline float euclidean(const std::span<const float>& v1, const std::span<const float>& v2) {
    assert(v1.size() == v2.size());
    const Eigen::Map<const Eigen::VectorXf> vector1(v1.data(), v1.size());
    const Eigen::Map<const Eigen::VectorXf> vector2(v2.data(), v2.size());
    return (vector1 - vector2).norm();
}
inline float manhattan(const std::span<const float>& v1, const std::span<const float>& v2) {
    assert(v1.size() == v2.size());
    const Eigen::Map<const Eigen::VectorXf> vector1(v1.data(), v1.size());
    const Eigen::Map<const Eigen::VectorXf> vector2(v2.data(), v2.size());
    return (vector1 - vector2).lpNorm<1>();
}
//cosine distance on normalised inputs
inline float cosine(const std::span<const float>& v1, const std::span<const float>& v2) {
    assert(v1.size() == v2.size());
    const Eigen::Map<const Eigen::VectorXf> vector1(v1.data(), v1.size());
    const Eigen::Map<const Eigen::VectorXf> vector2(v2.data(), v2.size());
    assert(std::abs(vector1.norm() - 1.0f) < 1e-6f);
    assert(std::abs(vector2.norm() - 1.0f) < 1e-6f);
    return 1.0f - vector1.dot(vector2);
}

// simple implementations
float computeEuclideanDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2);
float computeManhattanDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2);
float computeCosineDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2, bool normalised=false);
float computeDotProduct(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2);

// different vectorised implementations of euclidean
float computeEuclideanDistanceParUnseq(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2);
float computeEuclideanDistancePragma(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2);
float computeEuclideanEigenMap(const std::span<const float> &vector1, const std::span<const float>& vector2);
float computeEuclideanPureEigen(const Eigen::VectorXf& vector1, const Eigen::VectorXf& vector2);

// methods to operate directly on std::vector
float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b);
float dotProduct(const std::vector<float>& a, const std::vector<float>& b);

#endif //DISTANCEMETRICS_HPP
