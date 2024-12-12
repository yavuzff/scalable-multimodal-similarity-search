#ifndef EXACTMULTIINDEX_HPP
#define EXACTMULTIINDEX_HPP

#include "AbstractMultiIndex.hpp"

class ExactMultiIndex : public AbstractMultiIndex {
    std::vector<std::vector<float>> storedEntities;
    size_t validateEntities(const std::vector<std::vector<float>> &entities) const;

public:
    ExactMultiIndex(size_t modalities,
                    std::vector<size_t> dims,
                    std::vector<std::string> distance_metrics,
                    std::vector<float> weights);

    ExactMultiIndex(size_t modalities,
                    std::vector<size_t> dims,
                    std::vector<std::string> distance_metrics);


    void addEntities(const std::vector<std::vector<float>>& entities) override;

    std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k, const std::vector<float>& query_weights) override;

    std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k) override;

    void save(const std::string& path) const override;

    void load(const std::string& path) override;

};

#endif //EXACTMULTIINDEX_HPP
