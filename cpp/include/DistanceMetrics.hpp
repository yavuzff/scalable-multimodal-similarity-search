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

float computeEuclideanDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2);

float computeManhattanDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2);

float computeCosineDistance(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2, bool normalised=false);

float computeDotProduct(const std::span<const float>& vectorSlice1, const std::span<const float>& vectorSlice2);

#endif //DISTANCEMETRICS_HPP
