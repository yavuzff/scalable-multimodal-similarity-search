#ifndef DISTANCEMETRICS_HPP
#define DISTANCEMETRICS_HPP

#include <vector>
#include <string>
#include <span>

enum class DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
};

DistanceMetric stringToDistanceMetric(const std::string& str);
std::string distanceMetricToString(DistanceMetric metric);

float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b);
float dotProduct(const std::vector<float>& a, const std::vector<float>& b);

float computeEuclideanDistanceFromSlice(const std::span<const float>& storedEntity, size_t startIdx, size_t endIdx,
                               const std::span<const float>& queryEntity, size_t queryStartIdx);

float computeManhattanDistanceFromSlice(const std::span<const float>& storedEntity, size_t startIdx, size_t endIdx,
                               const std::span<const float>& queryEntity, size_t queryStartIdx);

float computeCosineDistanceFromSlice(const std::span<const float>& storedEntity, size_t startIdx, size_t endIdx,
                               const std::span<const float>& queryEntity, size_t queryStartIdx, bool normalised=false);

float computeDotProductFromSlice(const std::span<const float>& storedEntity, size_t startIdx, size_t endIdx,
                               const std::span<const float>& queryEntity, size_t queryStartIdx);

#endif //DISTANCEMETRICS_HPP
