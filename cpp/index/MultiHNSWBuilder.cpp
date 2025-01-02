#include "../include/MultiHNSW.hpp"

MultiHNSW::Builder::Builder(size_t numModalities, const std::vector<size_t> &dims): numModalities(numModalities), dims(dims) {};

MultiHNSW::Builder& MultiHNSW::Builder::setDistanceMetrics(const std::vector<std::string>& val) {
    distanceMetrics = val;
    return *this;
}

MultiHNSW::Builder& MultiHNSW::Builder::setWeights(const std::vector<float>& val) {
    weights = val;
    return *this;
}

MultiHNSW::Builder& MultiHNSW::Builder::setDistributionScaleFactor(float val) {
    distributionScaleFactor = val;
    return *this;
}

MultiHNSW::Builder& MultiHNSW::Builder::setTargetDegree(size_t val) {
    targetDegree = val;
    return *this;
}

MultiHNSW::Builder& MultiHNSW::Builder::setMaxDegree(size_t val) {
    maxDegree = val;
    return *this;
}

MultiHNSW::Builder& MultiHNSW::Builder::setEfConstruction(size_t val) {
    efConstruction = val;
    return *this;
}

MultiHNSW::Builder& MultiHNSW::Builder::setEfSearch(size_t val) {
    efSearch = val;
    return *this;
}

MultiHNSW::Builder& MultiHNSW::Builder::setSeed(size_t val) {
    seed = val;
    return *this;
}

MultiHNSW MultiHNSW::Builder::build() const {
    return MultiHNSW(numModalities, dims, distanceMetrics, weights,
                     distributionScaleFactor, targetDegree, maxDegree,
                     efConstruction, efSearch, seed);
}
