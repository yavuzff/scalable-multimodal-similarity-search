#ifndef MULTIINDEX_HPP
#define MULTIINDEX_HPP

#include <iostream>
#include <numeric>
#include <vector>


// function to validate and normalise weights
inline void validateAndNormaliseWeights(std::vector<float>& ws, size_t modalities) {
    if (ws.size() != modalities) {
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
    for (size_t i = 0; i < modalities; ++i) {
        ws[i] /= sum;
    }
}

// Abstract class for Multi vector K-NN index
class AbstractMultiIndex {
protected:
    size_t modalities;
    std::vector<size_t> dimensions;
    std::vector<std::string> distance_metrics;
    std::vector<float> weights;
    size_t numEntities = 0;

public:
    // Constructor: take parameters in by value to gain ownership
    AbstractMultiIndex(size_t the_modalities,
        std::vector<size_t> dims,
        std::vector<std::string> dist_metrics,
        std::vector<float> ws)
        : modalities(the_modalities), dimensions(std::move(dims)),distance_metrics(std::move(dist_metrics)) {
        if (modalities == 0) {
            throw std::invalid_argument("Number of modalities must be positive");
        }
        if (dimensions.size() != modalities) {
            throw std::invalid_argument("Number of dimensions must match number of modalities");
        }
        if (distance_metrics.size() != modalities) {
            throw std::invalid_argument("Number of distance metrics must match number of modalities");
        }
        validateAndNormaliseWeights(ws, modalities);
        weights = std::move(ws);

        // print out what we just initialised:
        std::cout << "Created MultiIndex with " << modalities << " modalities" << std::endl;
        for (size_t i = 0; i < modalities; ++i) {
            std::cout << "Modality " << i << " has dimension " << dimensions[i] << ", distance metric " << distance_metrics[i] << " and weight " << weights[i] << std::endl;
        }
    }

    // weights are optional, default is uniform weights which sum to 1
    AbstractMultiIndex(size_t modalities,
        std::vector<size_t> dims,
        std::vector<std::string> distance_metrics)
        : AbstractMultiIndex(modalities, std::move(dims), std::move(distance_metrics), std::vector<float>(modalities, 1.0f / modalities)) {}


    virtual ~AbstractMultiIndex() = default;

    // add multiple entities - note that the inner vector is flattened for performance
    // To add n entities with k modalities, provide a vector of length k,
    // where each element is a flattened vector of size n * dimensions_of_modality
    virtual void addEntities(const std::vector<std::vector<float>>& entities) = 0;


    // Return indices of the k-nearest neighbors (MAYBE: also return vectors themselves)
    virtual std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k,
                                       const std::vector<float>& query_weights) = 0;

    virtual std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k) = 0;

    virtual void save(const std::string& path) const = 0;

    virtual void load(const std::string& path) = 0;

    //getters for the properties
    [[nodiscard]] size_t getModalities() const {
        return modalities;
    }

    [[nodiscard]] const std::vector<size_t>& getDimensions() const {
        return dimensions;
    }

    [[nodiscard]] const std::vector<std::string>& getDistanceMetrics() const {
        return distance_metrics;
    }

    [[nodiscard]] const std::vector<float>& getWeights() const {
        return weights;
    }

    [[nodiscard]] size_t getNumEntities() const {
        return numEntities;
    }

};

#endif //MULTIINDEX_HPP
