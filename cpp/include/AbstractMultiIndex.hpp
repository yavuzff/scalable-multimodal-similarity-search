#ifndef MULTIINDEX_HPP
#define MULTIINDEX_HPP

#include <vector>
#include <span>

#include "DistanceMetrics.hpp"

void validateAndNormaliseWeights(std::vector<float>& ws, size_t numModalities);
std::vector<std::span<const float>> getSpanViewOfVectors(const std::vector<std::vector<float>> &vectors);

// Abstract class for Multi vector K-NN index
class AbstractMultiIndex {
protected:
    size_t numModalities;
    std::vector<size_t> dimensions;
    std::vector<DistanceMetric> distance_metrics;
    std::vector<std::string> str_distance_metrics;
    std::vector<float> weights;
    size_t numEntities = 0;

    size_t validateEntities(const std::vector<std::span<const float>> &entities) const;
    void validateQuery(const std::vector<std::span<const float>> &query, size_t k) const;

public:
    // Constructor: take parameters in by value to gain ownership
    AbstractMultiIndex(size_t the_modalities,
        std::vector<size_t> dims,
        std::vector<std::string> dist_metrics = {},
        std::vector<float> ws = {});

    virtual ~AbstractMultiIndex() = default;

    // add multiple entities - note that the inner vector is flattened for performance
    // To add n entities with k modalities, provide a vector of length k,
    // where each element is a flattened vector of size n * dimensions_of_modality
    virtual void addEntities(const std::vector<std::vector<float>>& entities) = 0;

    virtual void addEntities(const std::vector<std::span<const float>>& entities) = 0;

    // Return indices of the k-nearest neighbors (MAYBE: also return vectors themselves)
    virtual std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k,
                                       const std::vector<float>& query_weights) = 0;

    virtual std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k) = 0;

    virtual std::vector<size_t> search(const std::vector<std::span<const float>>& query, size_t k,
                                   const std::vector<float>& query_weights) = 0;

    virtual std::vector<size_t> search(const std::vector<std::span<const float>>& query, size_t k) = 0;

    virtual void save(const std::string& path) const = 0;

    virtual void load(const std::string& path) = 0;

    [[nodiscard]] size_t getNumModalities() const;
    [[nodiscard]] const std::vector<size_t>& getDimensions() const;
    [[nodiscard]] const std::vector<std::string>& getDistanceMetrics() const;
    [[nodiscard]] const std::vector<float>& getWeights() const;
    [[nodiscard]] size_t getNumEntities() const;

};

#endif //MULTIINDEX_HPP
