#include "../include/MultiHNSW.hpp"
#include "../include/utils.hpp"
#include "../include/common.hpp"

#include <iostream>
#include <random>

using namespace std;
using entity_id_t = int32_t;

MultiHNSW::MultiHNSW(size_t numModalities,
                     vector<size_t> dims,
                     vector<string> distanceMetrics,
                     vector<float> weights,
                     float distributionScaleFactor,
                     size_t targetDegree,
                     size_t maxDegree,
                     size_t efConstruction,
                     size_t efSearch,
                     size_t seed): AbstractMultiIndex(numModalities, std::move(dims), std::move(distanceMetrics), std::move(weights)),
                                   distributionScaleFactor(distributionScaleFactor), targetDegree(targetDegree), maxDegree(maxDegree), efConstruction(efConstruction),
                                    efSearch(efSearch), seed(seed), generator(seed){
    // validate parameters
    if (distributionScaleFactor <= 0) {
        throw invalid_argument("Distribution scale factor must be positive");
    }
    // check that targetDegree is at least 1
    if (targetDegree < 1) {
        throw invalid_argument("Target degree must be at least 1");
    }
    if (maxDegree < targetDegree) {
        throw invalid_argument("Max degree must be at least the target degree");
    }
    if (efConstruction < 1) {
        throw invalid_argument("efConstruction must be at least 1");
    }
    if (efSearch < 1) {
        throw invalid_argument("efSearch must be at least 1");
    }

    // initialise storage
    entityStorage.resize(numModalities);
}

void MultiHNSW::addEntities(const vector<vector<float>>& entities) {
    addEntities(getSpanViewOfVectors(entities));
}

void MultiHNSW::addEntities(const vector<span<const float>>& entities) {
    const size_t numNewEntities = validateEntities(entities);
    numEntities += numNewEntities;
    debug_printf("Adding %zu entities to MultiHNSW!\n", numNewEntities);
    addToEntityStorage(entities, numNewEntities);

    // update the HNSW structure
}

void MultiHNSW::addToEntityStorage(const vector<span<const float>>& entities, size_t numNewEntities) {
    // copy the input entities into the entityStorage
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];
        entityStorage[i].insert(entityStorage[i].end(), modalityVectors.begin(), modalityVectors.end());

        // if the modality uses cosine distance, normalise the inserted vectors for this modality
        if (distanceMetrics[i] == DistanceMetric::Cosine) {
            cout << "Storing normalised vectors for modality " << i << " to efficiently compute cosine distance" << endl;
            for (size_t j = 0; j < numNewEntities; ++j) {
                l2NormalizeVector(span(entityStorage[i]).subspan(j * dimensions[i], dimensions[i]));
            }
        }
    }
}

span<const float> MultiHNSW::getEntityModality(entity_id_t entityId, entity_id_t modality) const {
    const vector<float>& modalityData = entityStorage[modality];
    return span(modalityData.data() + entityId * dimensions[modality], dimensions[modality]);
}

float MultiHNSW::computeDistanceBetweenEntities(entity_id_t entityId1, entity_id_t entityId2, const vector<float>& weights) const{
    float dist = 0.0f;
    for (size_t modality = 0; modality < numModalities; ++modality) {
        if (weights[modality] == 0) {
            continue;
        }
        span<const float> vector1 = getEntityModality(entityId1, modality);
        span<const float> vector2 = getEntityModality(entityId2, modality);

        float modalityDistance;
        switch(distanceMetrics[modality]){
            case DistanceMetric::Euclidean:
                modalityDistance = computeEuclideanDistance(vector1, vector2);
                break;
            case DistanceMetric::Manhattan:
                modalityDistance = computeManhattanDistance(vector1, vector2);
                break;
            case DistanceMetric::Cosine:
                // vectors are already normalised
                modalityDistance = computeCosineDistance(vector1, vector2, true);
                break;
            default:
                throw invalid_argument("Invalid distance metric. You should not be seeing this message.");
        }

        // aggregate distance by summing modalityDistance*weight
        dist += weights[modality] * modalityDistance;
    }
    return dist;
}


vector<size_t> MultiHNSW::search(const vector<span<const float>>& query, size_t k, const vector<float>& queryWeights) {
    validateQuery(query, k);
    cout << "Searching MultiHNSW with query weights!" << endl;
    return vector<size_t>();
}

vector<size_t> MultiHNSW::search(const vector<span<const float>>& query, size_t k) {
    validateQuery(query, k);
    cout << "Searching MultiHNSW without query weights!" << endl;
    return vector<size_t>();
}

vector<size_t> MultiHNSW::search(const vector<vector<float>>& query, size_t k, const vector<float>& queryWeights) {
    return search(getSpanViewOfVectors(query), k, queryWeights);
}

vector<size_t> MultiHNSW::search(const vector<vector<float>>& query, size_t k) {
    return search(getSpanViewOfVectors(query), k);
}

void MultiHNSW::addEntityToGraph(entity_id_t entityId) {

}

[[nodiscard]] int MultiHNSW::generateRandomLevel() const {
    // generate a random level using -ln(U(0,1)) * distributionScaleFactor
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const int level = static_cast<int>(-log(distribution(generator)) * distributionScaleFactor);
    return level;
}

//vector<entity_id_t> MultiHNSW::internalSearchGraph(const vector<float>& query, size_t k, const vector<float>& weights, size_t ef) const;
//[[nodiscard]] priority_queue<pair<float, entity_id_t>> searchLayer(const vector<float>& query, vector<entity_id_t> entryPoints, size_t ef, size_t layer) const;
//vector<entity_id_t> selectNearestCandidates(const vector<float>& query, priority_queue<pair<float, entity_id_t>>& candidates, size_t M) const;
//vector<entity_id_t> selectDiversifiedCandidates(const vector<float>& query, priority_queue<pair<float, entity_id_t>>& candidates, size_t M) const;



void MultiHNSW::save(const string& path) const {
    cout << "Saving MultiHNSW to " << path << endl;
}

void MultiHNSW::load(const string& path) {
    cout << "Loading index from " << path << endl;
}

//implement getters and setters
float MultiHNSW::getDistributionScaleFactor() const {
    return distributionScaleFactor;
}

size_t MultiHNSW::getTargetDegree() const {
    return targetDegree;
}

size_t MultiHNSW::getMaxDegree() const {
    return maxDegree;
}

size_t MultiHNSW::getEfConstruction() const {
    return efConstruction;
}

size_t MultiHNSW::getEfSearch() const {
    return efSearch;
}

size_t MultiHNSW::getSeed() const {
    return seed;
}

void MultiHNSW::setEfSearch(size_t efSearch) {
    if (efSearch < 1) {
        throw invalid_argument("efSearch must be at least 1");
    }
    this->efSearch = efSearch;
}