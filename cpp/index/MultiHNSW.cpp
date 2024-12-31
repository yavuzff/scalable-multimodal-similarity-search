#include "../include/MultiHNSW.hpp"
#include "../include/utils.hpp"

#include <iostream>

MultiHNSW::MultiHNSW(size_t numModalities,
                     std::vector<size_t> dims,
                     std::vector<std::string> distanceMetrics,
                     std::vector<float> weights,
                     float distributionScaleFactor,
                     size_t targetDegree,
                     size_t maxDegree,
                     size_t efConstruction,
                     size_t efSearch,
                     size_t seed): AbstractMultiIndex(numModalities, std::move(dims), std::move(distanceMetrics), std::move(weights)),
                                   distributionScaleFactor(distributionScaleFactor), targetDegree(targetDegree), maxDegree(maxDegree), efConstruction(efConstruction), efSearch(efSearch), seed(seed) {
    // validate parameters
    if (distributionScaleFactor <= 0) {
        throw std::invalid_argument("Distribution scale factor must be positive");
    }
    // check that targetDegree is at least 1
    if (targetDegree < 1) {
        throw std::invalid_argument("Target degree must be at least 1");
    }
    if (maxDegree < targetDegree) {
        throw std::invalid_argument("Max degree must be at least the target degree");
    }
    if (efConstruction < 1) {
        throw std::invalid_argument("efConstruction must be at least 1");
    }
    if (efSearch < 1) {
        throw std::invalid_argument("efSearch must be at least 1");
    }

    // initialise storage
    entityStorage.resize(numModalities);
}

void MultiHNSW::addEntities(const std::vector<std::vector<float>>& entities) {
    addEntities(getSpanViewOfVectors(entities));
}

void MultiHNSW::addEntities(const std::vector<std::span<const float>>& entities) {
    const size_t numNewEntities = validateEntities(entities);
    numEntities += numNewEntities;
    std::cout << "Adding " << numNewEntities << " entities to MultiHNSW!" << std::endl;

    addToEntityStorage(entities, numNewEntities);

    // update the HNSW structure
}

void MultiHNSW::addToEntityStorage(const std::vector<std::span<const float>>& entities, size_t numNewEntities) {
    // copy the input entities into the entityStorage
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];
        entityStorage[i].insert(entityStorage[i].end(), modalityVectors.begin(), modalityVectors.end());

        // if the modality uses cosine distance, normalise the inserted vectors for this modality
        if (distanceMetrics[i] == DistanceMetric::Cosine) {
            std::cout << "Storing normalised vectors for modality " << i << " to efficiently compute cosine distance" << std::endl;
            for (size_t j = 0; j < numNewEntities; ++j) {
                l2NormalizeVector(std::span(entityStorage[i]).subspan(j * dimensions[i], dimensions[i]));
            }
        }
    }
}

std::span<const float> MultiHNSW::getEntityModality(size_t entityId, size_t modality) const {
    const std::vector<float>& modalityData = entityStorage[modality];
    return std::span(modalityData.data() + entityId * dimensions[modality], dimensions[modality]);
}

float MultiHNSW::computeDistanceBetweenEntities(size_t entityId1, size_t entityId2, const std::vector<float>& weights) const{
    float dist = 0.0f;
    for (size_t modality = 0; modality < numModalities; ++modality) {
        if (weights[modality] == 0) {
            continue;
        }
        std::span<const float> vector1 = getEntityModality(entityId1, modality);
        std::span<const float> vector2 = getEntityModality(entityId2, modality);

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
                throw std::invalid_argument("Invalid distance metric. You should not be seeing this message.");
        }

        // aggregate distance by summing modalityDistance*weight
        dist += weights[modality] * modalityDistance;
    }
    return dist;
}


std::vector<size_t> MultiHNSW::search(const std::vector<std::span<const float>>& query, size_t k, const std::vector<float>& queryWeights) {
    validateQuery(query, k);
    std::cout << "Searching MultiHNSW with query weights!" << std::endl;
    return std::vector<size_t>();
}

std::vector<size_t> MultiHNSW::search(const std::vector<std::span<const float>>& query, size_t k) {
    validateQuery(query, k);
    std::cout << "Searching MultiHNSW without query weights!" << std::endl;
    return std::vector<size_t>();
}

std::vector<size_t> MultiHNSW::search(const std::vector<std::vector<float>>& query, size_t k, const std::vector<float>& queryWeights) {
    return search(getSpanViewOfVectors(query), k, queryWeights);
}

std::vector<size_t> MultiHNSW::search(const std::vector<std::vector<float>>& query, size_t k) {
    return search(getSpanViewOfVectors(query), k);
}

void MultiHNSW::save(const std::string& path) const {
    std::cout << "Saving MultiHNSW to " << path << std::endl;
}

void MultiHNSW::load(const std::string& path) {
    std::cout << "Loading index from " << path << std::endl;
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
        throw std::invalid_argument("efSearch must be at least 1");
    }
    this->efSearch = efSearch;
}