#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <filesystem>

#include "../index/include/MultiVecHNSW.hpp"
#include "../index/include/utils.hpp"
#include "../index/include/common.hpp"

using namespace std;
namespace fs = std::filesystem;

float getRandomFloat() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

class MultiVecHNSWTest {
public:
    static MultiVecHNSW initaliseTest1() {
        MultiVecHNSW multiVecHNSW = MultiVecHNSW::Builder(2, {3, 3})
            .setDistanceMetrics({"euclidean", "manhattan"})
            .setWeights({0.5f, 0.5f})
            .build();

        return multiVecHNSW;
    }

    static void testAddToEntityStorageByModality1(MultiVecHNSW& hnsw) {
        std::vector<std::vector<float>> entities = {
            {1.0f, 2.0f, 3.0f,      4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f,      10.0f, 11.0f, 12.0f}
        };
        hnsw.addToEntityStorageByModality(getSpanViewOfVectors(entities), 2);

        REQUIRE(hnsw.entityStorageByModality.size() == 2);
        REQUIRE(hnsw.entityStorageByModality[0].size() == 6);
        REQUIRE(hnsw.entityStorageByModality[1].size() == 6);
    }

    static void testAddToEntityStorage(MultiVecHNSW& hnsw) {
        std::vector<std::vector<float>> entities = {
            {1.0f, 2.0f, 3.0f,      4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f,      10.0f, 11.0f, 12.0f}
        };
        hnsw.addToEntityStorage(getSpanViewOfVectors(entities), 2);

        REQUIRE(hnsw.entityStorage.size() == 12);

        if constexpr (REORDER_MODALITY_VECTORS) {
            //manhattan vectors should be before euclidean
            std::vector<float> expectedResultsReordered = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f, 4.0f, 5.0f, 6.0f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedResultsReordered));
        } else {
            // check that the entity storage is in the correct order
            std::vector<float> expectedResultsUnordered = {1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedResultsUnordered));

        }

    }

    static void testGetEntityModalityFromEntityId1(MultiVecHNSW& hnsw) {
        std::vector<std::vector<float>> entities = {
            {1.0f, 2.0f, 3.0f,      4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f,      10.0f, 11.0f, 12.0f}
        };
        hnsw.addToEntityStorageByModality(getSpanViewOfVectors(entities), 2);

        auto vector = hnsw.getEntityModalityFromEntityId(1, 1);
        REQUIRE(vector.size() == 3);
        REQUIRE(vector[0] == 10.0f);
        REQUIRE(vector[1] == 11.0f);
        REQUIRE(vector[2] == 12.0f);

        vector = hnsw.getEntityModalityFromEntityId(1, 0);
        REQUIRE(vector.size() == 3);
        REQUIRE(vector[0] == 4.0f);
        REQUIRE(vector[1] == 5.0f);
        REQUIRE(vector[2] == 6.0f);
    }

    static void testGetEntityFromEntityId1(MultiVecHNSW& hnsw) {
        std::vector<std::vector<float>> entities = {
            {1.0f, 2.0f, 3.0f,      4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f,      10.0f, 11.0f, 12.0f}
        };
        hnsw.addToEntityStorage(getSpanViewOfVectors(entities), 2);

        if constexpr (!REORDER_MODALITY_VECTORS) {
            auto vector1 = hnsw.getEntityFromEntityId(0);
            std::vector<float> expectedResults1 = {1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f};
            REQUIRE_THAT(vector1, Catch::Matchers::RangeEquals(expectedResults1));

            auto vector2 = hnsw.getEntityFromEntityId(1);
            std::vector<float> expectedResults2 = {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};
            REQUIRE_THAT(vector2, Catch::Matchers::RangeEquals(expectedResults2));
        } else {
            auto vector1 = hnsw.getEntityFromEntityId(0);
            std::vector<float> expectedResults1 = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
            REQUIRE_THAT(vector1, Catch::Matchers::RangeEquals(expectedResults1));

            auto vector2 = hnsw.getEntityFromEntityId(1);
            std::vector<float> expectedResults2 = {10.0f, 11.0f, 12.0f, 4.0f, 5.0f, 6.0f};
            REQUIRE_THAT(vector2, Catch::Matchers::RangeEquals(expectedResults2));
        }
    }

    static void testComputeDistanceBetweenEntities1(MultiVecHNSW& hnsw) {
        std::vector<std::vector<float>> entities = {
            {3.0f, 2.0f, 3.0f,      3.0f, -1.0f, 7.0f},
            {3.0f, 2.0f, 3.0f,      3.0f, -1.0f, 7.0f}
        };
        auto spanEntities = getSpanViewOfVectors(entities);
        hnsw.addToEntityStorage(spanEntities, 2);

        std::vector<float> weights = {0.5f, 0.5f};
        float distance = hnsw.computeDistance(hnsw.getEntityFromEntityId(0), hnsw.getEntityFromEntityId(1), weights);
        // 2.5 + 3.5 = 6.0f
        REQUIRE_THAT(distance, Catch::Matchers::WithinAbs(6.0f, 0.0001));
    }

    static void testGetEntityModalityFromEntityId2() {
        if constexpr (REORDER_MODALITY_VECTORS) {
            std::cout << "Skipping testGetEntityModalityFromEntityId2 as REORDER_MODALITY_VECTORS is true" << std::endl;
            return;
        }

        MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
            .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
            .setWeights({0.3f, 0.5f, 0.2f})
            .build();

        std::vector<std::vector<float>> entities = {
            {1.0f,                  2.0f,                   3.0f},
            {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.f}
        };
        hnsw.addToEntityStorageByModality(getSpanViewOfVectors(entities), 3);

        REQUIRE(hnsw.entityStorageByModality.size() == 3);
        REQUIRE(hnsw.entityStorageByModality[0].size() == 3);
        REQUIRE(hnsw.entityStorageByModality[1].size() == 6);
        REQUIRE(hnsw.entityStorageByModality[2].size() == 9);

        auto vector = hnsw.getEntityModalityFromEntityId(1, 2); // cosine so should be normalised
        REQUIRE(vector.size() == 3);
        REQUIRE_THAT(vector[0], Catch::Matchers::WithinAbs(0.5427628252422066, 0.0001));
        REQUIRE_THAT(vector[1], Catch::Matchers::WithinAbs(0.5766855018198446, 0.0001));
        REQUIRE_THAT(vector[2], Catch::Matchers::WithinAbs(0.6106081783974825, 0.0001));

        vector = hnsw.getEntityModalityFromEntityId(2, 2); // cosine so should be normalised
        REQUIRE(vector.size() == 3);
        REQUIRE_THAT(vector[0], Catch::Matchers::WithinAbs(0.548026257310873, 0.0001));
        REQUIRE_THAT(vector[1], Catch::Matchers::WithinAbs(0.576869744537761, 0.0001));
        REQUIRE_THAT(vector[2], Catch::Matchers::WithinAbs(0.6057132317646491, 0.0001));

        vector = hnsw.getEntityModalityFromEntityId(0, 1);
        REQUIRE(vector.size() == 2);
        REQUIRE(vector[0] == 7.0f);
        REQUIRE(vector[1] == 8.0f);
    }

    static void testGetEntityFromEntityId2() {
        MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
            .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
            .setWeights({0.3f, 0.5f, 0.2f})
            .build();

        std::vector<std::vector<float>> entities = {
            {1.0f,                  2.0f,                   3.0f},
            {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
        };
        hnsw.addToEntityStorage(getSpanViewOfVectors(entities), 3);

        if constexpr (!REORDER_MODALITY_VECTORS) {
            std::vector<float> expectedStorage = {1.0f, 7.0f, 8.0f, 0.5352015302352019f, 0.5763708787148328f, 0.6175402271944638f,
                2.0f, 9.0f, 10.0f, 0.5427628252422066f, 0.5766855018198446f, 0.6106081783974825f,
                3.0f, 11.0f, 12.0f, 0.548026257310873f, 0.576869744537761f, 0.6057132317646491f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        } else {
            std::vector<float> expectedStorage = {7.0f, 8.0f, 1.0f,0.5352015302352019f, 0.5763708787148328f, 0.6175402271944638f,
            9.0f, 10.0f, 2.0f, 0.5427628252422066f, 0.5766855018198446f, 0.6106081783974825f,
            11.0f, 12.0f, 3.0f, 0.548026257310873f, 0.576869744537761f, 0.6057132317646491f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }

        auto vector0 = hnsw.getEntityFromEntityId(0);
        if constexpr (!REORDER_MODALITY_VECTORS) {
            std::vector<float> expectedVector0 = {1.0f, 7.0f, 8.0f, 0.5352015302352019f, 0.5763708787148328f, 0.6175402271944638f};
            REQUIRE_THAT(vector0, Catch::Matchers::RangeEquals(expectedVector0));
        } else {
            std::vector<float> expectedVector0 = {7.0f, 8.0f, 1.0f, 0.5352015302352019f, 0.5763708787148328f, 0.6175402271944638f};
            REQUIRE_THAT(vector0, Catch::Matchers::RangeEquals(expectedVector0));
        }

        auto vector1 = hnsw.getEntityFromEntityId(1);
        if constexpr (!REORDER_MODALITY_VECTORS) {
            std::vector<float> expectedVector1 = {2.0f, 9.0f, 10.0f, 0.5427628252422066f, 0.5766855018198446f, 0.6106081783974825f};
            REQUIRE_THAT(vector1, Catch::Matchers::RangeEquals(expectedVector1));
        } else {
            std::vector<float> expectedVector1 = {9.0f, 10.0f, 2.0f, 0.5427628252422066f, 0.5766855018198446f, 0.6106081783974825f};
            REQUIRE_THAT(vector1, Catch::Matchers::RangeEquals(expectedVector1));
        }

        auto vector2 = hnsw.getEntityFromEntityId(2);
        if constexpr (!REORDER_MODALITY_VECTORS) {
            std::vector<float> expectedVector2 = {3.0f, 11.0f, 12.0f, 0.548026257310873f, 0.576869744537761f, 0.6057132317646491f};
            REQUIRE_THAT(vector2, Catch::Matchers::RangeEquals(expectedVector2));
        } else {
            std::vector<float> expectedVector2 = {11.0f, 12.0f, 3.0f, 0.548026257310873f, 0.576869744537761f, 0.6057132317646491f};
            REQUIRE_THAT(vector2, Catch::Matchers::RangeEquals(expectedVector2));
        }
    }

    static void testGenerateRandomLevel() {
        MultiVecHNSW hnsw = MultiVecHNSW::Builder(2, {3, 3}).build();

        int level = hnsw.generateRandomLevel();
        debug_printf("Random level: %d\n", level);
        REQUIRE(level >= 0);

        level = hnsw.generateRandomLevel();
        debug_printf("Random level: %d\n", level);
        REQUIRE(level >= 0);

        level = hnsw.generateRandomLevel();
        debug_printf("Random level: %d\n", level);
        REQUIRE(level >= 0);
    }

    static void testSearchLayerUsingEntityStorageByModality() {
        size_t numModalities = 1;
        vector<size_t> dims = {1};

        MultiVecHNSW multiVecHNSW = MultiVecHNSW::Builder(numModalities, dims)
            .setEfConstruction(10)
            .setEfSearch(10)
            .setTargetDegree(2)
            .setMaxDegree(3)
            .setSeed(42)
            .build();

        // add 5 entities
        std::vector<std::vector<float>> entities = {
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
        };
        multiVecHNSW.addToEntityStorage(getSpanViewOfVectors(entities), 5);

        // create 5 nodes with each node having 2 layers
        multiVecHNSW.nodes.resize(5);
        for (auto& node : multiVecHNSW.nodes) {
            node.neighboursPerLayer.resize(2);
        }
        multiVecHNSW.maxLevel = 1;

        // create graph
        multiVecHNSW.nodes[0].neighboursPerLayer[0] = {1, 2};
        multiVecHNSW.nodes[1].neighboursPerLayer[0] = {0, 3};
        multiVecHNSW.nodes[2].neighboursPerLayer[0] = {0, 4};
        multiVecHNSW.nodes[3].neighboursPerLayer[0] = {1};
        multiVecHNSW.nodes[4].neighboursPerLayer[0] = {2};

        multiVecHNSW.nodes[0].neighboursPerLayer[1] = {1};
        multiVecHNSW.nodes[1].neighboursPerLayer[1] = {0};

        // explicitly allocate memory for the query, so that the span refers to persistent data (not temporary data)
        std::vector<float> queryData = {5.0f};
        // we are using flattened query as this is the internal representation used for searchLayer
        auto query = span<const float>(queryData);
        vector<float> weights = {1.0f};
        size_t ef = 3;

        std::cout << "Searching for entity with query: " << query[0] << std::endl;

        SECTION("Test layer 1 search") {
            vector<MultiVecHNSW::entity_id_t> entryPoints = {0};
            auto resultLayer1 = multiVecHNSW.searchLayer(query, entryPoints, weights, ef, 1);

            vector<MultiVecHNSW::entity_id_t> expectedResults1 = {0, 1}; // max heap is based on distance, so results are in reverse order
            vector<MultiVecHNSW::entity_id_t> idsResultLayer1;

            while (!resultLayer1.empty()) {
                idsResultLayer1.push_back(resultLayer1.top().second);
                resultLayer1.pop();
            }

            REQUIRE_THAT(idsResultLayer1, Catch::Matchers::RangeEquals(expectedResults1));
        }

        SECTION("Test layer 0 search") {
            vector<MultiVecHNSW::entity_id_t> entryPoints = {3};  // start from a different entry point
            auto resultLayer0 = multiVecHNSW.searchLayer(query, entryPoints, weights, ef, 0);

            vector<MultiVecHNSW::entity_id_t> expectedResults0 = {2, 3, 4};
            vector<MultiVecHNSW::entity_id_t> idsResultLayer0;

            while (!resultLayer0.empty()) {
                idsResultLayer0.push_back(resultLayer0.top().second);
                resultLayer0.pop();
            }
            REQUIRE_THAT(idsResultLayer0, Catch::Matchers::RangeEquals(expectedResults0));
        }
    }

    static void testAddManyRandomEntities() {
        MultiVecHNSW index = MultiVecHNSW::Builder(2, {3, 3})
            .setDistanceMetrics({"euclidean", "manhattan"})
            .setWeights({0.5f, 0.5f})
            .setTargetDegree(3)
            .setMaxDegree(6)
            .build();

        //std::vector<std::vector<float>> entities = {
        //    {3.0f, 1.0f, 5.0f,      1.99f, 3.0f, 4.0f},
        //    {3.0f, 1.0f, 5.0f,      1.99f, 3.0f, 4.0f}
        //};

        size_t numEntities = 100;
        std::vector<float> modality1;
        std::vector<float> modality2;
        for (size_t i = 0; i < numEntities; i++) {
            // 3 dimensions per entity
            for (size_t j = 0; j < 3; j++) {
                modality1.push_back(getRandomFloat());
                modality2.push_back(getRandomFloat());
            }
        }
        std::vector<std::span<const float>> entities = {std::span(modality1), std::span(modality2)};

        index.addEntities(entities);
        index.printGraph();

    }

    static void testAddSingleEntities() {
        MultiVecHNSW index = MultiVecHNSW::Builder(1, {1})
            .setDistanceMetrics({ "manhattan"})
            .setTargetDegree(3)
            .setMaxDegree(6)
            .build();

        // create entities containing values from 0 to 100
        std::vector<float> nums = {};
        size_t numEntities = 100;
        nums.reserve(numEntities);
        for (size_t i = 0; i < numEntities; i++) {
            nums.push_back(i);
        }

        // shuffle the numbers
        std::random_device rd;  // Seed for random number generator
        std::mt19937 gen(rd()); // Mersenne Twister engine
        std::shuffle(nums.begin(), nums.end(), gen);

        std::vector<std::span<const float>> entities = {std::span(nums)};

        index.addEntities(entities);
        index.printGraph();

        // output index mappings
        debug_printf("Printing (entity id, entity vector) for %lu entities\n", numEntities);
        for (size_t i = 0; i < numEntities; i++) {
            debug_printf("(%lu, %d) ", i, int(entities[0][i]));
        }
    }

    static void testAddAndSearch1() {
        MultiVecHNSW index = MultiVecHNSW::Builder(2, {3, 3})
            .setDistanceMetrics({"euclidean", "manhattan"})
            .setWeights({0.5f, 0.5f})
            .setTargetDegree(3)
            .setMaxDegree(6)
            .setEfConstruction(1)
            .setEfSearch(1)
            .build();

        std::vector<std::vector<float>> entities = {
            {3.0f, 1.0f, 5.0f,      1.99f, 3.0f, 4.0f},
            {3.0f, 1.0f, 5.0f,      1.99f, 3.0f, 4.0f}
        };
        index.addEntities(entities);

        SECTION("Add items") {
            std::vector<std::vector<float>> query = {
                {3.0f, 2.0f, 3.0f}, {3.0f, 2.0f, 3.0f}
            };

            std::vector<float> weights1 = {1.0f, 0.0f};
            auto result = index.search(query, 1, weights1);
            REQUIRE(result == std::vector<size_t>{1});

            // weights 0.5f, 0.5f
            result = index.search(query, 1);
            REQUIRE(result == std::vector<size_t>{1});

            std::vector<float> weights2 = {0.05f, 0.995f};
            result = index.search(query, 1, weights2);
            REQUIRE(result == std::vector<size_t>{1});

            std::vector<float> weights3 = {0.02f, 0.98f};
            result = index.search(query, 1, weights3);
            REQUIRE(result == std::vector<size_t>{1});

            std::vector<float> weights4 = {0.019f, 0.981f};
            result = index.search(query, 1, weights4);
            REQUIRE(result == std::vector<size_t>{0});

            std::vector<float> weights5 = {0.0f, 1.0f};
            result = index.search(query, 1, weights5);
            REQUIRE(result == std::vector<size_t>{0});
        }
    }

    static void testAddAndSearch2() {
        MultiVecHNSW index = MultiVecHNSW::Builder(2, {3, 3})
            .setDistanceMetrics({"cosine", "manhattan"})
            .setWeights({0.5f, 0.5f})
            .setTargetDegree(3)
            .setMaxDegree(6)
            .setEfConstruction(1)
            .setEfSearch(1)
            .build();

        std::vector<std::vector<float>> entities = {
            {3.0f, 1.0f, 5.0f, 1.99f, 3.0f, 4.0f}, // Modality 1: 2 entities, dimension 3
            {3.0f, 1.0f, 5.0f, 1.99f, 3.0f, 4.0f}  // Modality 2: 2 entities, dimension 3
        };

        index.addEntities(entities);

        if constexpr (!REORDER_MODALITY_VECTORS) {
            std::vector<float> expectedStorage = {
                0.50709255283711f, 0.1690308509457033f, 0.8451542547285166f, 3.0, 1.0f, 5.0f,
                0.3697881993120397f, 0.55746964720408f, 0.7432928629387733f, 1.99f, 3.0f, 4.0f
            };
            REQUIRE_THAT(index.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        } else {
            std::vector<float> expectedStorage = {
                3.0f, 1.0f, 5.0f, 0.50709255283711f, 0.1690308509457033f, 0.8451542547285166f,
                1.99f, 3.0f, 4.0f, 0.3697881993120397f, 0.55746964720408f, 0.7432928629387733f,
            };
            REQUIRE_THAT(index.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }

        index.printGraph();
        std::vector<std::vector<float>> query = {
            {3.0f, 2.0f, 3.0f}, {3.0f, 2.0f, 3.0f} // query vectors for both modalities
        };

        SECTION("Search with cosine and manhattan metrics") {
            std::vector<float> weights1 = {1.0f, 0.0f};
            auto result = index.search(query, 1, weights1);
            REQUIRE(result == std::vector<size_t>{1}); // (0.063025, 0.050365)

            result = index.search(query, 1);
            REQUIRE(result == std::vector<size_t>{1}); // Even weights (1.5315125, 1.530182)

            std::vector<float> weights2 = {0.45f, 0.55f};
            result = index.search(query, 1, weights2);
            REQUIRE(result == std::vector<size_t>{1}); // (1.67836125, 1.6781638)

            std::vector<float> weights3 = {0.44f, 0.56f};
            result = index.search(query, 1, weights3);
            REQUIRE(result == std::vector<size_t>{0}); // (1.707731, 1.707760)

            std::cout << "Search with weights 0.0, 1.0" << std::endl;
            std::vector<float> weights4 = {0.0f, 1.0f};
            result = index.search(query, 1, weights4);
            std::cout << "About to  assert: " << result[0] << std::endl;
            REQUIRE(result == std::vector<size_t>{0}); // (3, 3.01)
            std::cout << "END with with weights 0.0, 1.0. Returned: " << result[0] << std::endl;
        }
    }

    static void testReordering() {

        if constexpr (!REORDER_MODALITY_VECTORS) {
            return ; // skip test if not reordering
        }

        SECTION("Test reordering manhattan < euclidean < cosine") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.5f, 0.2f})
                .build();

            std::vector<std::vector<float>> entities = {
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            };

            std::vector<size_t> expectedReordering = {1, 0, 2};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Manhattan, DistanceMetric::Euclidean, DistanceMetric::Cosine};
            std::vector<float> expectedWeights = {0.5f, 0.3f, 0.2f};
            std::vector<size_t> expectedDimensions = {2, 1, 3};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);

            hnsw.addEntities(entities);
            std::vector<float> expectedStorage = {7.0f, 8.0f, 1.0f, 0.5352015302352019f, 0.5763708787148328f, 0.6175402271944638f,
                 9.0f, 10.0f, 2.0f, 0.5427628252422066f, 0.5766855018198446f, 0.6106081783974825f,
                 11.0f, 12.0f, 3.0f, 0.548026257310873f, 0.576869744537761f, 0.6057132317646491f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }

        SECTION("Test reordering manhattan < euclidean < cosine when weight is 0") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.2f, 0.0f, 0.8f})
                .build();

            std::vector<std::vector<float>> entities = {
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            };

            std::vector<size_t> expectedReordering = {0, 2, 1};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Euclidean, DistanceMetric::Cosine, DistanceMetric::Manhattan};
            std::vector<float> expectedWeights = {0.2f, 0.8f, 0.0f};
            std::vector<size_t> expectedDimensions = {1, 3, 2};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);
        }

        SECTION("Test reordering metrics and weights") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"manhattan", "euclidean", "euclidean"})
                .setWeights({0.3f, 0.2f, 0.5f})
                .build();

            std::vector<std::vector<float>> entities = {
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            };

            std::vector<size_t> expectedReordering = {0, 2, 1};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Manhattan, DistanceMetric::Euclidean, DistanceMetric::Euclidean};
            std::vector<float> expectedWeights = {0.3f, 0.5f, 0.2f};
            std::vector<size_t> expectedDimensions = {1, 3, 2};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);

            hnsw.addEntities(entities);
            std::vector<float> expectedStorage = { 1.0f,  13.0f, 14.0f, 15.0f, 7.0f, 8.0f,
                    2.0f, 16.0f, 17.0f, 18.0f, 9.0f, 10.0f,
                  3.0f, 19.0f, 20.f, 21.0f, 11.0f, 12.0f,};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }

        SECTION("Test reordering weights only") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "euclidean", "euclidean"})
                .setWeights({0.5f, 0.2f, 0.3f})
                .build();

            std::vector<std::vector<float>> entities = {
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            };

            std::vector<size_t> expectedReordering = {0, 2, 1};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Euclidean, DistanceMetric::Euclidean, DistanceMetric::Euclidean};
            std::vector<float> expectedWeights = {0.5f, 0.3f, 0.2f};
            std::vector<size_t> expectedDimensions = {1, 3, 2};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);

            hnsw.addEntities(entities);
            std::vector<float> expectedStorage = { 1.0f,  13.0f, 14.0f, 15.0f, 7.0f, 8.0f,
                    2.0f, 16.0f, 17.0f, 18.0f, 9.0f, 10.0f,
                  3.0f, 19.0f, 20.f, 21.0f, 11.0f, 12.0f,};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }

        SECTION("Test reordering dimensions only") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "euclidean", "euclidean"})
                .setWeights({0.2f, 0.2f, 0.2f})
                .build();

            std::vector<std::vector<float>> entities = {
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            };

            std::vector<size_t> expectedReordering = {2, 1, 0};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Euclidean, DistanceMetric::Euclidean, DistanceMetric::Euclidean};
            std::vector<float> expectedWeights = {0.333333313f, 0.333333313f, 0.333333313f};
            std::vector<size_t> expectedDimensions = {3, 2, 1};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);

            hnsw.addEntities(entities);
            std::vector<float> expectedStorage = {13.0f, 14.0f, 15.0f, 7.0f, 8.0f, 1.0f,
                        16.0f, 17.0f, 18.0f, 9.0f, 10.0f, 2.0f,
                        19.0f, 20.f, 21.0f, 11.0f, 12.0f, 3.0f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }


        SECTION("Test reordering metric and dimensions") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 3, 2})
                .setDistanceMetrics({"euclidean", "euclidean", "manhattan"})
                .setWeights({0.2f, 0.2f, 0.2f})
                .build();

            std::vector<std::vector<float>> entities = {
                {1.0f,                  2.0f,                   3.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
            };

            std::vector<size_t> expectedReordering = {2, 1, 0};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Manhattan, DistanceMetric::Euclidean, DistanceMetric::Euclidean};
            std::vector<float> expectedWeights = {0.333333313f, 0.333333313f, 0.333333313f};
            std::vector<size_t> expectedDimensions = {2, 3, 1};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);

            hnsw.addEntities(entities);
            std::vector<float> expectedStorage = {7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 1.0f,
                        9.0f, 10.0f, 16.0f, 17.0f, 18.0f, 2.0f,
                        11.0f, 12.0f, 19.0f, 20.f, 21.0f, 3.0f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }

        SECTION("Test reordering weights and dimensions") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {3, 1, 2})
                .setDistanceMetrics({"euclidean", "euclidean", "euclidean"})
                .setWeights({0.2f, 0.4f, 0.4f})
                .build();

            std::vector<std::vector<float>> entities = {
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f},
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
            };

            std::vector<size_t> expectedReordering = {2, 1, 0};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Euclidean, DistanceMetric::Euclidean, DistanceMetric::Euclidean};
            std::vector<float> expectedWeights = {0.4f, 0.4f, 0.2f};
            std::vector<size_t> expectedDimensions = {2, 1, 3};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);

            hnsw.addEntities(entities);
            std::vector<float> expectedStorage = {7.0f, 8.0f, 1.0f, 13.0f, 14.0f, 15.0f,
                        9.0f, 10.0f, 2.0f, 16.0f, 17.0f, 18.0f,
                        11.0f, 12.0f, 3.0f, 19.0f, 20.f, 21.0f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }

        SECTION("Test reordering metrics no matter the weights") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.2f, 0.5f})
                .build();

            std::vector<std::vector<float>> entities = {
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            };

            std::vector<size_t> expectedReordering = {1, 0, 2};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Manhattan, DistanceMetric::Euclidean, DistanceMetric::Cosine};
            std::vector<float> expectedWeights = {0.2f, 0.3f, 0.5f};
            std::vector<size_t> expectedDimensions = {2, 1, 3};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);

            hnsw.addEntities(entities);
            std::vector<float> expectedStorage = {7.0f, 8.0f, 1.0f, 0.5352015302352019f, 0.5763708787148328f, 0.6175402271944638f,
                 9.0f, 10.0f, 2.0f, 0.5427628252422066f, 0.5766855018198446f, 0.6106081783974825f,
                 11.0f, 12.0f, 3.0f, 0.548026257310873f, 0.576869744537761f, 0.6057132317646491f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }

        SECTION("Test reordering metrics no matter the dimensions") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {2, 1, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.5f, 0.2f})
                .build();

            std::vector<std::vector<float>> entities = {
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {1.0f,                  2.0f,                   3.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            };

            std::vector<size_t> expectedReordering = {1, 0, 2};
            std::vector<DistanceMetric> expectedDistanceMetrics = {DistanceMetric::Manhattan, DistanceMetric::Euclidean, DistanceMetric::Cosine};
            std::vector<float> expectedWeights = {0.5f, 0.3f, 0.2f};
            std::vector<size_t> expectedDimensions = {1, 2, 3};

            REQUIRE(hnsw.modalityReordering == expectedReordering);
            REQUIRE(hnsw.distanceMetrics == expectedDistanceMetrics);
            REQUIRE(hnsw.indexWeights == expectedWeights);
            REQUIRE(hnsw.dimensions == expectedDimensions);

            hnsw.addEntities(entities);
            std::vector<float> expectedStorage = {1.0f, 7.0f, 8.0f, 0.5352015302352019f, 0.5763708787148328f, 0.6175402271944638f,
                 2.0f, 9.0f, 10.0f, 0.5427628252422066f, 0.5766855018198446f, 0.6106081783974825f,
                 3.0f, 11.0f, 12.0f, 0.548026257310873f, 0.576869744537761f, 0.6057132317646491f};
            REQUIRE_THAT(hnsw.entityStorage, Catch::Matchers::RangeEquals(expectedStorage));
        }
    }

    static void testEquality() {
        SECTION("Test equality of identical HNSWs") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });
            hnsw2.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });

            REQUIRE(hnsw1 == hnsw2);
        }

        SECTION("Test equality with different distance metrics") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "euclidean", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });
            hnsw2.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });

            REQUIRE(hnsw1 != hnsw2);
        }

        SECTION("Test equality with different weights") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.4f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });
            hnsw2.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });

            REQUIRE(hnsw1 != hnsw2);
        }

        SECTION("Test equality with different modality reordering") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();
            hnsw2.modalityReordering = {2, 1, 0};

            REQUIRE(hnsw1 != hnsw2);
        }

        SECTION("Test equality with different entities") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });
            hnsw2.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 9.0f,            9.0f, 10.0f,            11.0f, 12.0f}, // the second item here is different
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });

            REQUIRE(hnsw1 != hnsw2);
        }

        SECTION("Test equality with identical but different storage order") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });
            hnsw2.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });

            // should be unequal due to different storage order
            std::reverse(hnsw2.entityStorage.begin(), hnsw2.entityStorage.end());

            REQUIRE(hnsw1 != hnsw2);
        }

        SECTION("Test equality with identical but different internal state") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });
            hnsw2.addEntities({
                {1.0f,                  2.0f,                   3.0f},
                {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.0f}
            });

            // change hnsw1's internal data to simulate a difference
            hnsw1.distributionScaleFactor = 99.0f;

            REQUIRE(hnsw1 != hnsw2);
        }
    }

    static void testSerializationAndDeserialization() {
        SECTION("Test serialization and deserialization of identical HNSWs") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            // serialise to stringstream
            std::stringstream ss;
            hnsw1.serialize(ss);

            // deserialise back into a new object
            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(1, {2}).build();
            hnsw2.deserialize(ss);

            REQUIRE(hnsw1 == hnsw2);
        }

        SECTION("Test deserialization of HNSW with different internal state") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            std::stringstream ss;
            hnsw1.serialize(ss);

            // modify internal state of hnsw1 after serialisation
            hnsw1.distributionScaleFactor = 99.0f;

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(1, {2}).build();
            hnsw2.deserialize(ss);

            REQUIRE(hnsw1 != hnsw2);
        }

        SECTION("Test deserialization failure due to corrupted data") {
            MultiVecHNSW hnsw1 = MultiVecHNSW::Builder(3, {1, 2, 3})
                    .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                    .setWeights({0.3f, 0.5f, 0.2f})
                    .build();

            hnsw1.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            std::stringstream ss;
            hnsw1.serialize(ss);

            // corrupt the stream by writing extra bytes to the start
            std::string corruptedData = "corrupted_data";
            ss.seekp(0);
            ss.write(corruptedData.c_str(), corruptedData.size());
            ss.seekg(0);

            MultiVecHNSW hnsw2 = MultiVecHNSW::Builder(1, {2}).build();
            hnsw2.deserialize(ss);

            // assert different
            REQUIRE(hnsw1 != hnsw2);
        }
    }

    static void testLoadAndSave() {
        const std::string testDir = "./tests/";

        SECTION("Test saving to a valid path") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.5f, 0.2f})
                .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            std::string filePath = testDir + "hnsw_test.bin";
            REQUIRE_NOTHROW(hnsw.save(filePath));

            // verify the file exists
            REQUIRE(fs::exists(filePath));

            // cleanup
            fs::remove(filePath);
        }

        SECTION("Test saving to a non-existent directory") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.5f, 0.2f})
                .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            std::string filePath = testDir + "new_directory/hnsw_test.bin";

            REQUIRE_NOTHROW(hnsw.save(filePath));

            // verify the file exists and the directory was created
            REQUIRE(fs::exists(filePath));
            REQUIRE(fs::exists(testDir + "new_directory"));

            // cleanup
            fs::remove(filePath);
            fs::remove_all(testDir + "new_directory");
        }

        SECTION("Test loading from a valid file") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.5f, 0.2f})
                .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            std::string filePath = testDir + "hnsw_test.bin";
            hnsw.save(filePath);

            MultiVecHNSW loadedHNSW = MultiVecHNSW::Builder(1, {2}).build();
            REQUIRE_NOTHROW(loadedHNSW.load(filePath));

            // verify that the loaded object matches the original
            REQUIRE(hnsw == loadedHNSW);

            // cleanup
            fs::remove(filePath);
        }

        SECTION("Test loading from a non-existent file") {
            std::string filePath = testDir + "non_existent_file.bin";
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(1, {2}).build();

            REQUIRE_THROWS_AS(hnsw.load(filePath), std::runtime_error);
        }

        SECTION("Test loading index using loadIndex") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.5f, 0.2f})
                .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            std::string filePath = testDir + "index_test.bin";
            hnsw.save(filePath);

            MultiVecHNSW loadedHNSW = MultiVecHNSW::loadIndex(filePath);

            // verify that the loaded index is as expected
            REQUIRE(hnsw == loadedHNSW);

            // cleanup
            fs::remove(filePath);
        }

        SECTION("Test loading index from a non-existent file") {
            std::string filePath = testDir + "non_existent_index.bin";
            REQUIRE_THROWS_AS(MultiVecHNSW::loadIndex(filePath), std::runtime_error);
        }

    }

    static void testStats() {
        if constexpr (!TRACK_STATS) {
            return;
        }

        SECTION("Test stats increase") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.5f, 0.2f})
                .setEfConstruction(1)
                .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            REQUIRE(hnsw.getNumComputeDistanceCalls() > 0);
            REQUIRE(hnsw.getNumLazyDistanceCalls() > 0);
            REQUIRE(hnsw.getNumLazyDistanceCutoff() >= 0);
            REQUIRE(hnsw.getNumVectorsSkippedDueToCutoff() >= 0);
        }

        SECTION("Test cutoff stats") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.1f, 0.9f, 0.1f})
                .setEfConstruction(1)
                .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 8.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f, 16.0f, 17.0f, 18.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f}
            });

            REQUIRE(hnsw.getNumComputeDistanceCalls() > 0);
            REQUIRE(hnsw.getNumLazyDistanceCalls() > 0);
            REQUIRE(hnsw.getNumLazyDistanceCutoff() > 0);
            REQUIRE(hnsw.getNumVectorsSkippedDueToCutoff() > 0);
        }

        SECTION("Test stats reset") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.1f, 0.9f, 0.1f})
                .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f}
            });

            hnsw.resetStats();
            REQUIRE(hnsw.getNumComputeDistanceCalls() == 0);
            REQUIRE(hnsw.getNumLazyDistanceCalls() == 0);
            REQUIRE(hnsw.getNumLazyDistanceCutoff() == 0);
            REQUIRE(hnsw.getNumVectorsSkippedDueToCutoff() == 0);
        }

        SECTION("Test stats search increase") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
                .setDistanceMetrics({"cosine", "manhattan", "cosine"})
                .setWeights({0.1f, 0.9f, 0.1f})
                .setEfConstruction(1)
                .setEfSearch(1)
                .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 8.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f, 16.0f, 17.0f, 18.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f}
            });

            hnsw.resetStats();

            std::vector<std::vector<float>> query = {{1.0f}, {2.0f, 3.0f}, {4.0,5.0f,6.0f}};

            hnsw.search(query, 5);

            REQUIRE(hnsw.getNumComputeDistanceCalls() > 0);
            REQUIRE(hnsw.getNumLazyDistanceCalls() > 0);
            REQUIRE(hnsw.getNumLazyDistanceCutoff() >= 0);
            REQUIRE(hnsw.getNumVectorsSkippedDueToCutoff() >= 0);
        }

        SECTION("Test stats search cutoff") {
            MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {1, 2, 3})
              .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
              .setWeights({0.1f, 0.9f, 0.1f})
              .setEfConstruction(1)
              .setEfSearch(1)
              .build();

            hnsw.addEntities({
                {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 8.0f},
                {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.f, 21.0f, 16.0f, 17.0f, 18.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f}
            });

            hnsw.resetStats();

            std::vector<std::vector<float>> query = {{1.0f}, {99.0f, 98.0f}, {4.0,5.0f,6.0f}};

            hnsw.search(query, 1);

            REQUIRE(hnsw.getNumComputeDistanceCalls() > 0);
            REQUIRE(hnsw.getNumLazyDistanceCalls() > 0);
            REQUIRE(hnsw.getNumLazyDistanceCutoff() > 0);
            REQUIRE(hnsw.getNumVectorsSkippedDueToCutoff() > 0);
        }
    }

};

TEST_CASE("MultiVecHNSWBuilder builds", "[MultiVecHNSWBuilder]") {
    SECTION("Test default build") {

        MultiVecHNSW multiVecHNSW = MultiVecHNSW::Builder(2, {3,3})
                              .setEfSearch(50)
                              .build();

        REQUIRE(multiVecHNSW.getNumModalities() == 2);
        REQUIRE_THAT(multiVecHNSW.getDimensions(), Catch::Matchers::RangeEquals(std::vector<size_t> {3, 3}));;
        REQUIRE_THAT(multiVecHNSW.getWeights(), Catch::Matchers::RangeEquals(std::vector<float> {0.5, 0.5}));;

        REQUIRE(multiVecHNSW.getEfSearch() == 50);
        REQUIRE(multiVecHNSW.getTargetDegree() == 32);
        REQUIRE(multiVecHNSW.getMaxDegree() == 32);
        REQUIRE_THAT(multiVecHNSW.getDistributionScaleFactor() , Catch::Matchers::WithinAbs(1/log(multiVecHNSW.getTargetDegree()), 0.0001));
        REQUIRE(multiVecHNSW.getEfConstruction() == 200);
        REQUIRE(multiVecHNSW.getSeed() == 42);
    }
}

TEST_CASE("MultiVecHNSW Basic Tests", "[MultiVecHNSW]") {
    MultiVecHNSW hnsw = MultiVecHNSWTest::initaliseTest1();
    SECTION("Test addToEntityStorageByModality") {
        MultiVecHNSWTest::testAddToEntityStorageByModality1(hnsw);
    }

    SECTION("Test addToEntityStorage") {
        MultiVecHNSWTest::testAddToEntityStorage(hnsw);
    }

    SECTION("Test getEntityModality") {
        MultiVecHNSWTest::testGetEntityModalityFromEntityId1(hnsw);
    }

    SECTION("Test getEntityFromEntityId") {
        MultiVecHNSWTest::testGetEntityFromEntityId1(hnsw);
    }

    SECTION("Test getEntityModality2") {
        MultiVecHNSWTest::testGetEntityModalityFromEntityId2();
    }

    SECTION("Test getEntityFromEntityId2") {
        MultiVecHNSWTest::testGetEntityFromEntityId2();
    }

    SECTION("Test computeDistanceBetweenEntities") {
        MultiVecHNSWTest::testComputeDistanceBetweenEntities1(hnsw);
    }

    SECTION("Test generateRandomLevel") {
        MultiVecHNSWTest::testGenerateRandomLevel();
    }
}

TEST_CASE("MultiVecHNSW SearchLayer", "[SearchLayer]") {
    MultiVecHNSWTest::testSearchLayerUsingEntityStorageByModality();
}

TEST_CASE("MultiVecHNSW AddEntities", "[AddEntities]") {
    MultiVecHNSWTest::testAddManyRandomEntities();
    MultiVecHNSWTest::testAddSingleEntities();
}

TEST_CASE("MultiVecHNSW Search", "[Search]") {
    MultiVecHNSWTest::testAddAndSearch1();
    MultiVecHNSWTest::testAddAndSearch2();
}

TEST_CASE("MultiVecHNSW testReordering", "[Reordering]") {
    MultiVecHNSWTest::testReordering();
}

TEST_CASE("MultiVecHNSW testEquality", "[Equality]") {
    MultiVecHNSWTest::testEquality();
}

TEST_CASE("MultiVecHNSW testSerDe", "[SerDe]") {
    MultiVecHNSWTest::testSerializationAndDeserialization();
}

TEST_CASE("MultiVecHNSW testLoadAndSave", "[LoadAndSave]") {
    MultiVecHNSWTest::testLoadAndSave();
}

TEST_CASE("MultiVecHNSW testStats", "[Stats]") {
    MultiVecHNSWTest::testStats();
}