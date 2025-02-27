#ifndef MULTIVECHNSW_HPP
#define MULTIVECHNSW_HPP

#include <queue>
#include <random>

#include "AbstractMultiVecIndex.hpp"

class MultiVecHNSW : public AbstractMultiVecIndex {
public:
    using entity_id_t = u_int32_t;

private:
    // parameters + inherited parameters
    float distributionScaleFactor;
    size_t targetDegree;
    size_t maxDegree;
    size_t efConstruction;
    size_t efSearch;
    size_t seed;

    // internal data structures
    std::vector<std::vector<float>> entityStorageByModality;
    std::vector<float> entityStorage;

    struct Node {
        std::vector<std::vector<entity_id_t>> neighboursPerLayer;
    };
    std::vector<Node> nodes;

    entity_id_t entryPoint;
    size_t maxLevel;
    size_t maxDegreeLayer0;  // set to 2 * maxDegree

    mutable std::mt19937 generator;

    // private methods
    void validateParameters() const;
    void addToEntityStorage(const std::vector<std::span<const float>>& entities, size_t num_entities);
    std::span<const float> getEntityFromEntityId(entity_id_t entityId) const;

    // storage by modality
    void addToEntityStorageByModality(const std::vector<std::span<const float>>& entities, size_t num_entities);
    std::span<const float> getEntityModalityFromEntityId(entity_id_t entityId, size_t modality) const;

    float computeDistance(std::span<const float> entity1,  std::span<const float> entity2, const std::vector<float>& weights) const;

    [[nodiscard]] size_t generateRandomLevel() const;
    void addEntityToGraph(entity_id_t entityId);
    std::vector<entity_id_t> internalSearch(const std::vector<std::span<const float>>& userQuery, size_t k, const std::vector<float>& weights) const;

    [[nodiscard]] std::priority_queue<std::pair<float, entity_id_t>> searchLayer(std::span<const float> entity, const std::vector<entity_id_t> &entryPoints, const std::vector<float>& weights, size_t ef, size_t layer) const;
    [[nodiscard]] std::priority_queue<std::pair<float, entity_id_t>> searchLayer(entity_id_t entityId, const std::vector<entity_id_t> &entryPoints, const std::vector<float>& weights, size_t ef, size_t layer) const;

    void selectNearestCandidates(std::priority_queue<std::pair<float, entity_id_t>> &candidates, size_t resultSize) const;
    std::priority_queue<std::pair<float, entity_id_t>> selectNearestCandidates(entity_id_t targetEntityId, std::span<entity_id_t> candidates, size_t numSelected, const std::vector<float>& weights) const;
    void selectDiversifiedCandidates(std::priority_queue<std::pair<float, entity_id_t>>& candidates, size_t targetSelectedNeighbours, const std::vector<float>& weights) const;

    void addAndPruneEdgesForExistingNodes(entity_id_t newEntityId, const std::vector<std::pair<float, entity_id_t>> &connectedNeighbours, size_t layer);

    friend class MultiVecHNSWTest;  // grant access to the test class
    // Stats: number of edges traversed, number of distances computed, number of nodes visited

public:
    MultiVecHNSW(size_t numModalities,
                    std::vector<size_t> dims,
                    std::vector<std::string> distanceMetrics = {},
                    std::vector<float> weights = {},
                    float distributionScaleFactor = 0.0f,
                    size_t targetDegree = 32,
                    size_t maxDegree = 32,
                    size_t efConstruction = 200,
                    size_t efSearch = 50,
                    size_t seed = 42);

    class Builder {
    public:
        Builder(size_t numModalities, const std::vector<size_t> &dims);

        Builder& setDistanceMetrics(const std::vector<std::string>& val);
        Builder& setWeights(const std::vector<float>& val);
        Builder& setDistributionScaleFactor(float val);
        Builder& setTargetDegree(size_t val);
        Builder& setMaxDegree(size_t val);
        Builder& setEfConstruction(size_t val);
        Builder& setEfSearch(size_t val);
        Builder& setSeed(size_t val);

        [[nodiscard]] MultiVecHNSW build() const;

    private:
        size_t numModalities;
        std::vector<size_t> dims;
        std::vector<std::string> distanceMetrics = {};
        std::vector<float> weights = {};
        float distributionScaleFactor = 0.0f;
        size_t targetDegree = 32;
        size_t maxDegree = 32;
        size_t efConstruction = 200;
        size_t efSearch = 50;
        size_t seed = 42;
    };


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

    void printGraph() const;
};



#endif
