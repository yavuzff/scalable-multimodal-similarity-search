#include "include/MultiVecHNSW.hpp"

MultiVecHNSW::Builder::Builder(size_t numModalities, const std::vector<size_t> &dims): numModalities(numModalities), dims(dims) {};

MultiVecHNSW::Builder& MultiVecHNSW::Builder::setDistanceMetrics(const std::vector<std::string>& val) {
    distanceMetrics = val;
    return *this;
}

MultiVecHNSW::Builder& MultiVecHNSW::Builder::setWeights(const std::vector<float>& val) {
    weights = val;
    return *this;
}

MultiVecHNSW::Builder& MultiVecHNSW::Builder::setDistributionScaleFactor(float val) {
    distributionScaleFactor = val;
    return *this;
}

MultiVecHNSW::Builder& MultiVecHNSW::Builder::setTargetDegree(size_t val) {
    targetDegree = val;
    return *this;
}

MultiVecHNSW::Builder& MultiVecHNSW::Builder::setMaxDegree(size_t val) {
    maxDegree = val;
    return *this;
}

MultiVecHNSW::Builder& MultiVecHNSW::Builder::setEfConstruction(size_t val) {
    efConstruction = val;
    return *this;
}

MultiVecHNSW::Builder& MultiVecHNSW::Builder::setEfSearch(size_t val) {
    efSearch = val;
    return *this;
}

MultiVecHNSW::Builder& MultiVecHNSW::Builder::setSeed(size_t val) {
    seed = val;
    return *this;
}

MultiVecHNSW MultiVecHNSW::Builder::build() const {
    return MultiVecHNSW(numModalities, dims, distanceMetrics, weights,
                     distributionScaleFactor, targetDegree, maxDegree,
                     efConstruction, efSearch, seed);
}
