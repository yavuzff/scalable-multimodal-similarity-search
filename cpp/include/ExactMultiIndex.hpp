#ifndef EXACTMULTIINDEX_HPP
#define EXACTMULTIINDEX_HPP

#include "AbstractMultiIndex.hpp"

class ExactMultiIndex : public AbstractMultiIndex {

private:
    std::vector<std::vector<std::vector<float>>> entities;
    void validateEntity(const std::vector<std::vector<float>> &entity) const;

public:
    ExactMultiIndex(const size_t modalities,
                    std::vector<size_t> dims,
                    std::vector<std::string> distance_metrics,
                    std::vector<float> weights)
        : AbstractMultiIndex(modalities, std::move(dims), std::move(distance_metrics), std::move(weights)) {}

    ExactMultiIndex(const size_t modalities,
                    std::vector<size_t> dims,
                    std::vector<std::string> distance_metrics): AbstractMultiIndex(modalities, std::move(dims), std::move(distance_metrics)) {}

    void add(const std::vector<std::vector<float>>& entity) override;

    std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k, const std::vector<float>& query_weights) override;

    std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k) override;

    void save(const std::string& path) const override;

    void load(const std::string& path) override;
};

#endif //EXACTMULTIINDEX_HPP
