#ifndef EXACTMULTIVECINDEX_HPP
#define EXACTMULTIVECINDEX_HPP

#include "AbstractMultiVecIndex.hpp"
#include "DistanceMetrics.hpp"
#include <span>

class ExactMultiVecIndex : public AbstractMultiVecIndex {
    std::vector<std::vector<float>> storedEntities;

    [[nodiscard]] std::vector<size_t> internalSearch(const std::vector<std::span<const float>>& query, size_t k,
                        const std::vector<float>& normalisedWeights) const;

    void outputEntities() const;

public:
    ExactMultiVecIndex(size_t numModalities,
                    std::vector<size_t> dims,
                    std::vector<std::string> distanceMetrics = {},
                    std::vector<float> weights = {});

    void addEntities(const std::vector<std::vector<float>>& entities) override;

    void addEntities(const std::vector<std::span<const float>>& entities) override;

    std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k, const std::vector<float>& queryWeights) override;

    std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k) override;

    std::vector<size_t> search(const std::vector<std::span<const float>>& query, size_t k, const std::vector<float>& queryWeights) override;

    std::vector<size_t> search(const std::vector<std::span<const float>>& query, size_t k) override;

    void save(const std::string& path) const override;

    void load(const std::string& path) override;

};

#endif //EXACTMULTIVECINDEX_HPP
