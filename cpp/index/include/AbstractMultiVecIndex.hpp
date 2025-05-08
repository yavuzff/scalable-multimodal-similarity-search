#ifndef ABSTRACTMULTIVECINDEX_HPP
#define ABSTRACTMULTIVECINDEX_HPP

#include <vector>
#include <span>

#include "DistanceMetrics.hpp"

// abstract class for multi-vector K-NN index
class AbstractMultiVecIndex {
protected:
    size_t numModalities;
    std::vector<size_t> dimensions;
    std::vector<DistanceMetric> distanceMetrics;
    std::vector<std::string> strDistanceMetrics;
    std::vector<float> indexWeights;
    size_t numEntities = 0;
    std::vector<size_t> toNormalise; // indices of modality vectors to normalise
    size_t totalDimensions; // sum of all dimensions

    [[nodiscard]] size_t validateEntities(const std::vector<std::span<const float>> &entities, const std::vector<size_t>& expectedDimensions) const;
    void validateQuery(const std::vector<std::span<const float>> &query, size_t k, const std::vector<size_t> &expectedDimensions) const;

    virtual void serialize(std::ostream& os) const;
    virtual void deserialize(std::istream& is);

public:
    // Constructor: take parameters in by value to gain ownership
    AbstractMultiVecIndex(size_t theModalities,
        std::vector<size_t> dims,
        std::vector<std::string> distMetrics = {},
        std::vector<float> ws = {});

    virtual ~AbstractMultiVecIndex() = default;

    bool operator==(const AbstractMultiVecIndex& other) const;

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
                                   const std::vector<float>& queryWeights) = 0;

    virtual std::vector<size_t> search(const std::vector<std::span<const float>>& query, size_t k) = 0;

    virtual void save(const std::string& path) const = 0;

    virtual void load(const std::string& path) = 0;

    [[nodiscard]] virtual size_t getNumModalities() const;
    [[nodiscard]] virtual const std::vector<size_t>& getDimensions() const;
    [[nodiscard]] virtual const std::vector<std::string>& getDistanceMetrics() const;
    [[nodiscard]] virtual const std::vector<float>& getWeights() const;
    [[nodiscard]] virtual size_t getNumEntities() const;

};

#endif //ABSTRACTMULTIVECINDEX_HPP
