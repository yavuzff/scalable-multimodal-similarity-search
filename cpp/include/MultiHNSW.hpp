#ifndef MULTIHNSW_HPP
#define MULTIHNSW_HPP

#include "AbstractMultiIndex.hpp"

class MultiHNSW : public AbstractMultiIndex {

    // HNSW parameters + inherited parameters
    float distributionScaleFactor;
    size_t targetDegree;
    size_t maxDegree;
    size_t efConstruction;
    size_t efSearch;
    size_t seed;

    std::vector<std::vector<float>> entityStorage;

    friend class MultiHNSWTest;  // grant access to the test class

    // private methods
    void addToEntityStorage(const std::vector<std::span<const float>>& entities, size_t num_entities);
    std::span<const float> getEntityModality(size_t entityId, size_t modality) const;
    float computeDistanceBetweenEntities(size_t entityId1, size_t entityId2, const std::vector<float>& queryWeights) const;


public:
    MultiHNSW(size_t numModalities,
                    std::vector<size_t> dims,
                    std::vector<std::string> distanceMetrics = {},
                    std::vector<float> weights = {},
                    float distributionScaleFactor = 1.0f,
                    size_t targetDegree = 32,
                    size_t maxDegree = 32,
                    size_t efConstruction = 200,
                    size_t efSearch = 50,
                    size_t seed = 42);

    void addEntities(const std::vector<std::vector<float>>& entities) override;

    void addEntities(const std::vector<std::span<const float>>& entities) override;

    std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k, const std::vector<float>& queryWeights) override;

    std::vector<size_t> search(const std::vector<std::vector<float>>& query, size_t k) override;

    std::vector<size_t> search(const std::vector<std::span<const float>>& query, size_t k, const std::vector<float>& queryWeights) override;

    std::vector<size_t> search(const std::vector<std::span<const float>>& query, size_t k) override;

    void save(const std::string& path) const override;

    void load(const std::string& path) override;

    [[nodiscard]] float getDistributionScaleFactor() const;
    [[nodiscard]] size_t getTargetDegree() const;
    [[nodiscard]] size_t getMaxDegree() const;
    [[nodiscard]] size_t getEfConstruction() const;
    [[nodiscard]] size_t getEfSearch() const;
    [[nodiscard]] size_t getSeed() const;

    void setEfSearch(size_t efSearch);
};



#endif
