#ifndef MULTIVECHNSW_HPP
#define MULTIVECHNSW_HPP

#include <queue>
#include <random>

#include "AbstractMultiVecIndex.hpp"

constexpr bool TRACK_STATS = false;
constexpr bool USE_LAZY_DISTANCE = true;
constexpr bool REORDER_MODALITY_VECTORS = true; // reordering should be turned on only when USE_LAZY_DISTANCE is true

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
    std::vector<float> entityStorage;
    std::vector<std::vector<float>> entityStorageByModality; // not used in main index, and doesnt support reordering

    struct Node {
        std::vector<std::vector<entity_id_t>> neighboursPerLayer;
        bool operator==(const Node& other) const {
            return neighboursPerLayer == other.neighboursPerLayer;
        }
    };
    std::vector<Node> nodes;

    entity_id_t entryPoint;
    size_t maxLevel;
    size_t maxDegreeLayer0;  // set to 2 * maxDegree

    mutable std::mt19937 generator;

    // reordering of modalities
    std::vector<size_t> modalityReordering; // used to reorder the modalities for distance computation
    std::vector<size_t> originalOrderedDimensions;
    std::vector<std::string> originalOrderedStrDistanceMetrics;
    std::vector<float> originalOrderedIndexWeights;

    // stats - only keep track if flag is set
    mutable unsigned long long num_compute_distance_calls;
    mutable unsigned long long num_lazy_distance_calls;
    mutable unsigned long long num_lazy_distance_cutoff;
    mutable unsigned long long num_vectors_skipped_due_to_cutoff;

    // private methods
    void validateParameters() const;
    void reorderModalities();
    std::vector<size_t> identifyModalityReordering() const;
    void addToEntityStorage(const std::vector<std::span<const float>>& entities, size_t num_entities);
    std::span<const float> getEntityFromEntityId(entity_id_t entityId) const;

    // storage by modality
    void addToEntityStorageByModality(const std::vector<std::span<const float>>& entities, size_t num_entities);
    std::span<const float> getEntityModalityFromEntityId(entity_id_t entityId, size_t modality) const;

    float computeDistance(std::span<const float> entity1,  std::span<const float> entity2, const std::vector<float>& weights) const;
    float computeDistanceLazy(std::span<const float> entity1,  std::span<const float> entity2, const std::vector<float>& weights, float upperBound) const;

    [[nodiscard]] size_t generateRandomLevel() const;
    void addEntityToGraph(entity_id_t entityId);
    std::vector<entity_id_t> internalSearch(const std::vector<std::span<const float>>& userQuery, size_t k, const std::vector<float>& weights) const;

    [[nodiscard]] std::priority_queue<std::pair<float, entity_id_t>> searchLayer(std::span<const float> entity, const std::vector<entity_id_t> &entryPoints, const std::vector<float>& weights, size_t ef, size_t layer) const;
    [[nodiscard]] std::priority_queue<std::pair<float, entity_id_t>> searchLayer(entity_id_t entityId, const std::vector<entity_id_t> &entryPoints, const std::vector<float>& weights, size_t ef, size_t layer) const;

    void selectNearestCandidates(std::priority_queue<std::pair<float, entity_id_t>> &candidates, size_t resultSize) const;
    std::priority_queue<std::pair<float, entity_id_t>> selectNearestCandidates(entity_id_t targetEntityId, std::span<entity_id_t> candidates, size_t numSelected, const std::vector<float>& weights) const;
    void selectDiversifiedCandidates(std::priority_queue<std::pair<float, entity_id_t>>& candidates, size_t targetSelectedNeighbours, const std::vector<float>& weights) const;

    void addAndPruneEdgesForExistingNodes(entity_id_t newEntityId, const std::vector<std::pair<float, entity_id_t>> &connectedNeighbours, size_t layer);

    // serialisation helpers
    static void serializeNode(std::ostream& os, const Node& node);
    static void deserializeNode(std::istream& is, Node& node);

    // serialisation methods
    void serialize(std::ostream& os) const override;
    void deserialize(std::istream& is) override;

    friend class MultiVecHNSWTest;  // grant access to the test class

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

    // static method to load the index from a file
    static MultiVecHNSW loadIndex(const std::string& path);

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

    // equality operator
    bool operator==(const MultiVecHNSW& other) const;

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

    // methods for stats
    [[nodiscard]] unsigned long long getNumComputeDistanceCalls() const;
    [[nodiscard]] unsigned long long getNumLazyDistanceCalls() const;
    [[nodiscard]] unsigned long long getNumLazyDistanceCutoff() const;
    [[nodiscard]] unsigned long long getNumVectorsSkippedDueToCutoff() const;
    void resetStats();

    // override getters
    [[nodiscard]] const std::vector<size_t>& getDimensions() const override;
    [[nodiscard]] const std::vector<std::string>& getDistanceMetrics() const override;
    [[nodiscard]] const std::vector<float>& getWeights() const override;

    void setEfSearch(size_t efSearch);

    void printGraph() const;
};


#endif