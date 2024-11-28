#ifndef MULTIINDEX_HPP
#define MULTIINDEX_HPP

#include <utility>
#include <vector>

// Abstract class for Multi vector K-NN index
class AbstractMultiIndex {
public:
    size_t modalities;
    std::vector<size_t> dimensions;
    std::vector<std::string> distance_metrics;
    std::vector<float> weights;

    // Constructor: take parameters in by value to gain ownership
    AbstractMultiIndex(size_t modalities,
        std::vector<size_t> dims,
        std::vector<std::string> distance_metrics,
        std::vector<float> weights)
        : modalities(modalities), dimensions(std::move(dims)),distance_metrics(std::move(distance_metrics)), weights(std::move(weights)) {}
        //validate parameters

    virtual ~AbstractMultiIndex() = default;

    // add a single entity to the index
    virtual void add(const std::vector<std::vector<float>>& entity) = 0;

    // add multiple entites
    // virtual void addMultiple(const std::vector<std::vector<std::vector<float>>>& entity) = 0;


    // Return indices of the k-nearest neighbors (MAYBE: also return vectors themselves)
    virtual std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k,
                                       const std::vector<float>& query_weights) = 0;

    virtual std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k) = 0;

    virtual void save(const std::string& path) const = 0;

    virtual void load(const std::string& path) = 0;

};


#endif //MULTIINDEX_HPP
