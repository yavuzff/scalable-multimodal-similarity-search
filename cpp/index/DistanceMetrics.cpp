#include "../include/DistanceMetrics.hpp"

#include <vector>

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

float weightedEuclideanDistance(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b,
    const std::vector<float>& weights) {
    float entity_distance = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        const float vector_distance = euclideanDistance(a[i], b[i]);
        entity_distance += vector_distance * weights[i];
    }
    return entity_distance;
}