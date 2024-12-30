#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <span>

void validateAndNormaliseWeights(std::vector<float>& ws, size_t numModalities);

std::vector<std::span<const float>> getSpanViewOfVectors(const std::vector<std::vector<float>> &vectors);

void l2NormalizeVector(std::span<float> vector);

#endif //UTILS_HPP
