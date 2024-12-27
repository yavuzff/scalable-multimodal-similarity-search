#include "../include/MultiHNSW.hpp"

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
    //validate parameters
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
}

void MultiHNSW::addEntities(const std::vector<std::span<const float>>& entities) {
    const size_t numNewEntities = validateEntities(entities);
    numEntities += numNewEntities;
    std::cout << "Adding " << numNewEntities << " entities to MultiHNSW!" << std::endl;
};

void MultiHNSW::addEntities(const std::vector<std::vector<float>>& entities) {
    addEntities(getSpanViewOfVectors(entities));
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