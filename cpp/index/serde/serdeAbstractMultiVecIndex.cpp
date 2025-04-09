#include "../../include/AbstractMultiVecIndex.hpp"
#include "../../include/serde/serdeUtils.hpp"

void AbstractMultiVecIndex::serialize(std::ostream& os) const {
    // serialise size_t variables
    os.write(reinterpret_cast<const char*>(&numModalities), sizeof(numModalities));
    os.write(reinterpret_cast<const char*>(&numEntities), sizeof(numEntities));
    os.write(reinterpret_cast<const char*>(&totalDimensions), sizeof(totalDimensions));

    // serialise vectors
    serializeVector(os, dimensions);
    serializeVector(os, distanceMetrics);

    // serialise string vector
    size_t numStrs = strDistanceMetrics.size();
    os.write(reinterpret_cast<const char*>(&numStrs), sizeof(numStrs));
    for (const auto& str : strDistanceMetrics) {
        size_t len = str.size();
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        os.write(str.data(), len);
    }

    serializeVector(os, indexWeights);
    serializeVector(os, toNormalise);
}

void AbstractMultiVecIndex::deserialize(std::istream& is) {
    // deserialise size_t variables
    is.read(reinterpret_cast<char*>(&numModalities), sizeof(numModalities));
    is.read(reinterpret_cast<char*>(&numEntities), sizeof(numEntities));
    is.read(reinterpret_cast<char*>(&totalDimensions), sizeof(totalDimensions));

    // deserialise vectors
    deserializeVector(is, dimensions);
    deserializeVector(is, distanceMetrics);

    // deserialise string vector
    size_t numStrs;
    is.read(reinterpret_cast<char*>(&numStrs), sizeof(numStrs));
    strDistanceMetrics.resize(numStrs);
    for (auto& str : strDistanceMetrics) {
        size_t len;
        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        str.resize(len);
        is.read(str.data(), len);
    }
    deserializeVector(is, indexWeights);
    deserializeVector(is, toNormalise);
}


