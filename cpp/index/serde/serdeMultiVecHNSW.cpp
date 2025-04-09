#include <iostream>

#include "../../include/MultiVecHNSW.hpp"
#include "../../include/serde/serdeUtils.hpp"


// node serialisation
void MultiVecHNSW::serializeNode(std::ostream& os, const Node& node) {
    serializeNestedVector(os, node.neighboursPerLayer);
}

void MultiVecHNSW::deserializeNode(std::istream& is, Node& node) {
    deserializeNestedVector(is, node.neighboursPerLayer);
}


void MultiVecHNSW::serialize(std::ostream& os) const {
    std::cout << "Serializing MultiVecHNSW index..." << std::endl;
    // note: we do not serialise the generator

    AbstractMultiVecIndex::serialize(os);

    // serialise compile-time constants as runtime values
    os.write(reinterpret_cast<const char*>(&TRACK_STATS), sizeof(TRACK_STATS));
    os.write(reinterpret_cast<const char*>(&USE_LAZY_DISTANCE), sizeof(USE_LAZY_DISTANCE));
    os.write(reinterpret_cast<const char*>(&REORDER_MODALITY_VECTORS), sizeof(REORDER_MODALITY_VECTORS));

    // serialise parameters
    os.write(reinterpret_cast<const char*>(&distributionScaleFactor), sizeof(distributionScaleFactor));
    os.write(reinterpret_cast<const char*>(&targetDegree), sizeof(targetDegree));
    os.write(reinterpret_cast<const char*>(&maxDegree), sizeof(maxDegree));
    os.write(reinterpret_cast<const char*>(&efConstruction), sizeof(efConstruction));
    os.write(reinterpret_cast<const char*>(&efSearch), sizeof(efSearch));
    os.write(reinterpret_cast<const char*>(&seed), sizeof(seed));

    // serialise storage
    serializeVector(os, entityStorage);
    serializeNestedVector(os, entityStorageByModality);

    size_t nodeCount = nodes.size();
    os.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));
    for (const auto& node : nodes) {
        serializeNode(os, node);
    }

    os.write(reinterpret_cast<const char*>(&entryPoint), sizeof(entryPoint));
    os.write(reinterpret_cast<const char*>(&maxLevel), sizeof(maxLevel));
    os.write(reinterpret_cast<const char*>(&maxDegreeLayer0), sizeof(maxDegreeLayer0));

    if constexpr (REORDER_MODALITY_VECTORS) {
        serializeVector(os, modalityReordering);
        serializeVector(os, originalOrderedDimensions);

        // serialise string vector
        size_t numStrs = originalOrderedStrDistanceMetrics.size();
        os.write(reinterpret_cast<const char*>(&numStrs), sizeof(numStrs));
        for (const auto& str : originalOrderedStrDistanceMetrics) {
            size_t len = str.size();
            os.write(reinterpret_cast<const char*>(&len), sizeof(len));
            os.write(str.data(), len);
        }

        serializeVector(os, originalOrderedIndexWeights);
    }

    if constexpr (TRACK_STATS) {
        os.write(reinterpret_cast<const char*>(&num_compute_distance_calls), sizeof(num_compute_distance_calls));
        os.write(reinterpret_cast<const char*>(&num_lazy_distance_calls), sizeof(num_lazy_distance_calls));
        os.write(reinterpret_cast<const char*>(&num_lazy_distance_cutoff), sizeof(num_lazy_distance_cutoff));
        os.write(reinterpret_cast<const char*>(&num_vectors_skipped_due_to_cutoff), sizeof(num_vectors_skipped_due_to_cutoff));
    }
}

void MultiVecHNSW::deserialize(std::istream& is) {
    std::cout << "Deserializing MultiVecHNSW index..." << std::endl;
    // note: we do not deserialise the generator
    AbstractMultiVecIndex::deserialize(is);

    // deserialise and check compile-time constants
    bool trackStats, useLazyDistance, reorderModalityVectors;
    is.read(reinterpret_cast<char*>(&trackStats), sizeof(trackStats));
    is.read(reinterpret_cast<char*>(&useLazyDistance), sizeof(useLazyDistance));
    is.read(reinterpret_cast<char*>(&reorderModalityVectors), sizeof(reorderModalityVectors));

    if (trackStats != TRACK_STATS || useLazyDistance != USE_LAZY_DISTANCE || reorderModalityVectors != REORDER_MODALITY_VECTORS) {
        throw std::runtime_error("Error: deserialized configuration does not match compile-time constants.");
    }

    is.read(reinterpret_cast<char*>(&distributionScaleFactor), sizeof(distributionScaleFactor));
    is.read(reinterpret_cast<char*>(&targetDegree), sizeof(targetDegree));
    is.read(reinterpret_cast<char*>(&maxDegree), sizeof(maxDegree));
    is.read(reinterpret_cast<char*>(&efConstruction), sizeof(efConstruction));
    is.read(reinterpret_cast<char*>(&efSearch), sizeof(efSearch));
    is.read(reinterpret_cast<char*>(&seed), sizeof(seed));

    deserializeVector(is, entityStorage);
    deserializeNestedVector(is, entityStorageByModality);

    size_t nodeCount;
    is.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
    nodes.resize(nodeCount);
    for (auto& node : nodes) {
        deserializeNode(is, node);
    }

    is.read(reinterpret_cast<char*>(&entryPoint), sizeof(entryPoint));
    is.read(reinterpret_cast<char*>(&maxLevel), sizeof(maxLevel));
    is.read(reinterpret_cast<char*>(&maxDegreeLayer0), sizeof(maxDegreeLayer0));

    if constexpr (REORDER_MODALITY_VECTORS) {
        deserializeVector(is, modalityReordering);
        deserializeVector(is, originalOrderedDimensions);

        // deserialise string vector
        size_t numStrs;
        is.read(reinterpret_cast<char*>(&numStrs), sizeof(numStrs));
        originalOrderedStrDistanceMetrics.resize(numStrs);
        for (auto& str : originalOrderedStrDistanceMetrics) {
            size_t len;
            is.read(reinterpret_cast<char*>(&len), sizeof(len));
            str.resize(len);
            is.read(str.data(), len);
        }

        deserializeVector(is, originalOrderedIndexWeights);
    }

    if constexpr (TRACK_STATS) {
        is.read(reinterpret_cast<char*>(&num_compute_distance_calls), sizeof(num_compute_distance_calls));
        is.read(reinterpret_cast<char*>(&num_lazy_distance_calls), sizeof(num_lazy_distance_calls));
        is.read(reinterpret_cast<char*>(&num_lazy_distance_cutoff), sizeof(num_lazy_distance_cutoff));
        is.read(reinterpret_cast<char*>(&num_vectors_skipped_due_to_cutoff), sizeof(num_vectors_skipped_due_to_cutoff));
    }
}
