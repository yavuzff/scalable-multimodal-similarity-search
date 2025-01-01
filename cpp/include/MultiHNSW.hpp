#ifndef MULTIHNSW_HPP
#define MULTIHNSW_HPP

#include <queue>

#include "AbstractMultiIndex.hpp"

class MultiHNSW : public AbstractMultiIndex {
    using entity_id_t = int32_t;

    // parameters + inherited parameters
    float distributionScaleFactor;
    size_t targetDegree;
    size_t maxDegree;
    size_t efConstruction;
    size_t efSearch;
    size_t seed;

    // internal data structures
    std::vector<std::vector<float>> entityStorage;
    struct Node {
        std::vector<std::vector<entity_id_t>> edgesPerLayer;
    };
    std::vector<Node> nodes;

    entity_id_t entryPoint;
    size_t maxLayer;

    /*
     * Nodes: vector of Node, where index of Node corresponds to entity_id
     * Node: adjacency list for each layer. Adjacency list: entity_id for each layer 0 to l, vector of vector of entity_id.
     *    -> potentially pre-allocate space for each vector of entity_id
     */

    // private methods
    void addToEntityStorage(const std::vector<std::span<const float>>& entities, size_t num_entities);
    std::span<const float> getEntityModality(entity_id_t entityId, entity_id_t modality) const;
    float computeDistanceBetweenEntities(entity_id_t entityId1, entity_id_t entityId2, const std::vector<float>& weights) const;

    [[nodiscard]] int generateRandomLevel() const;
    void addEntityToGraph(entity_id_t entityId);
    std::vector<entity_id_t> internalSearchGraph(const std::vector<float>& query, size_t k, const std::vector<float>& weights, size_t ef) const;
    [[nodiscard]] std::priority_queue<std::pair<float, entity_id_t>> searchLayer(const std::vector<float>& query, std::vector<entity_id_t> entryPoints, size_t ef, size_t layer) const;
    std::vector<entity_id_t> selectNearestCandidates(const std::vector<float>& query, std::priority_queue<std::pair<float, entity_id_t>>& candidates, size_t M) const;
    std::vector<entity_id_t> selectDiversifiedCandidates(const std::vector<float>& query, std::priority_queue<std::pair<float, entity_id_t>>& candidates, size_t M) const;

    friend class MultiHNSWTest;  // grant access to the test class
    // Stats: number of edges traversed, number of distances computed, number of nodes visited

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
