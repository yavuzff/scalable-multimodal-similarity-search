
#ifndef DISTANCEMETRICS_HPP
#define DISTANCEMETRICS_HPP

#include <vector>

float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b);

float dotProduct(const std::vector<float>& a, const std::vector<float>& b);

float computeEuclideanDistanceFromSlice(const std::vector<float>& storedEntity, size_t startIdx, size_t endIdx,
                               const std::vector<float>& queryEntity, size_t queryStartIdx);

float weightedEuclideanDistance(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b,
    const std::vector<float>& weights);

#endif //DISTANCEMETRICS_HPP
