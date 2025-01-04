#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include "../include/MultiHNSW.hpp"
#include "../include/utils.hpp"
#include "../include/common.hpp"

using namespace std;

float getRandomFloat() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

class MultiHNSWTest {
public:
    static MultiHNSW initaliseTest1() {
        MultiHNSW multiHNSW = MultiHNSW::Builder(2, {3, 3})
            .setDistanceMetrics({"euclidean", "manhattan"})
            .setWeights({0.5f, 0.5f})
            .build();

        return multiHNSW;
    }

    static void testAddToEntityStorage1(MultiHNSW& hnsw) {
        std::vector<std::vector<float>> entities = {
            {1.0f, 2.0f, 3.0f,      4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f,      10.0f, 11.0f, 12.0f}
        };
        hnsw.addToEntityStorage(getSpanViewOfVectors(entities), 2);

        REQUIRE(hnsw.entityStorage.size() == 2);
        REQUIRE(hnsw.entityStorage[0].size() == 6);
        REQUIRE(hnsw.entityStorage[1].size() == 6);
    }

    static void testGetEntityModality1(MultiHNSW& hnsw) {
        std::vector<std::vector<float>> entities = {
            {1.0f, 2.0f, 3.0f,      4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f,      10.0f, 11.0f, 12.0f}
        };
        hnsw.addToEntityStorage(getSpanViewOfVectors(entities), 2);

        auto vector = hnsw.getEntityModality(1, 1);
        REQUIRE(vector.size() == 3);
        REQUIRE(vector[0] == 10.0f);
        REQUIRE(vector[1] == 11.0f);
        REQUIRE(vector[2] == 12.0f);

        vector = hnsw.getEntityModality(1, 0);
        REQUIRE(vector.size() == 3);
        REQUIRE(vector[0] == 4.0f);
        REQUIRE(vector[1] == 5.0f);
        REQUIRE(vector[2] == 6.0f);

    }

    static void testComputeDistanceBetweenEntities1(MultiHNSW& hnsw) {
        std::vector<std::vector<float>> entities = {
            {3.0f, 2.0f, 3.0f,      3.0f, -1.0f, 7.0f},
            {3.0f, 2.0f, 3.0f,      3.0f, -1.0f, 7.0f}
        };
        auto spanEntities = getSpanViewOfVectors(entities);
        hnsw.addToEntityStorage(spanEntities, 2);

        std::vector<float> weights = {0.5f, 0.5f};
        float distance = hnsw.computeDistanceBetweenEntities(0, 1, weights);
        // 2.5 + 3.5 = 6.0f
        REQUIRE_THAT(distance, Catch::Matchers::WithinAbs(6.0f, 0.0001));
    }

    static void testGetEntityModality2() {
        MultiHNSW hnsw = MultiHNSW::Builder(3, {1, 2, 3})
            .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
            .setWeights({0.3f, 0.5f, 0.2f})
            .build();

        std::vector<std::vector<float>> entities = {
            {1.0f,                  2.0f,                   3.0f},
            {7.0f, 8.0f,            9.0f, 10.0f,            11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f,   16.0f, 17.0f, 18.0f,    19.0f, 20.f, 21.f}
        };
        hnsw.addToEntityStorage(getSpanViewOfVectors(entities), 2);

        REQUIRE(hnsw.entityStorage.size() == 3);
        REQUIRE(hnsw.entityStorage[0].size() == 3);
        REQUIRE(hnsw.entityStorage[1].size() == 6);
        REQUIRE(hnsw.entityStorage[2].size() == 9);

        auto vector = hnsw.getEntityModality(1, 2); // cosine so should be normalised
        REQUIRE(vector.size() == 3);
        REQUIRE_THAT(vector[0], Catch::Matchers::WithinAbs(0.5427628252422066, 0.0001));
        REQUIRE_THAT(vector[1], Catch::Matchers::WithinAbs(0.5766855018198446, 0.0001));
        REQUIRE_THAT(vector[2], Catch::Matchers::WithinAbs(0.6106081783974825, 0.0001));

        vector = hnsw.getEntityModality(0, 1);
        REQUIRE(vector.size() == 2);
        REQUIRE(vector[0] == 7.0f);
        REQUIRE(vector[1] == 8.0f);
    }

    static void testGenerateRandomLevel() {
        MultiHNSW hnsw = MultiHNSW::Builder(2, {3, 3}).build();

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

    static void testSearchLayer() {
        size_t numModalities = 1;
        vector<size_t> dims = {1};

        MultiHNSW multiHNSW = MultiHNSW::Builder(numModalities, dims)
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
        multiHNSW.addToEntityStorage(getSpanViewOfVectors(entities), 5);

        // create 5 nodes with each node having 2 layers
        multiHNSW.nodes.resize(5);
        for (auto& node : multiHNSW.nodes) {
            node.neighboursPerLayer.resize(2);
        }
        multiHNSW.maxLevel = 1;

        // create graph
        multiHNSW.nodes[0].neighboursPerLayer[0] = {1, 2};
        multiHNSW.nodes[1].neighboursPerLayer[0] = {0, 3};
        multiHNSW.nodes[2].neighboursPerLayer[0] = {0, 4};
        multiHNSW.nodes[3].neighboursPerLayer[0] = {1};
        multiHNSW.nodes[4].neighboursPerLayer[0] = {2};

        multiHNSW.nodes[0].neighboursPerLayer[1] = {1};
        multiHNSW.nodes[1].neighboursPerLayer[1] = {0};

        // explicitly allocate memory for the query, so that the span refers to persistent data (not temporary data)
        std::vector<float> queryData = {5.0f};
        vector<span<const float>> query = {span<const float>(queryData)};
        vector<float> weights = {1.0f};
        size_t ef = 3;

        SECTION("Test layer 1 search") {
            vector<MultiHNSW::entity_id_t> entryPoints = {0};
            auto resultLayer1 = multiHNSW.searchLayer(query, entryPoints, weights, ef, 1);

            vector<MultiHNSW::entity_id_t> expectedResults1 = {0, 1}; // max heap is based on distance, so results are in reverse order
            vector<MultiHNSW::entity_id_t> idsResultLayer1;

            while (!resultLayer1.empty()) {
                idsResultLayer1.push_back(resultLayer1.top().second);
                resultLayer1.pop();
            }

            REQUIRE_THAT(idsResultLayer1, Catch::Matchers::RangeEquals(expectedResults1));
        }

        SECTION("Test layer 0 search") {
            vector<MultiHNSW::entity_id_t> entryPoints = {3};  // start from a different entry point
            auto resultLayer0 = multiHNSW.searchLayer(query, entryPoints, weights, ef, 0);

            vector<MultiHNSW::entity_id_t> expectedResults0 = {2, 3, 4};
            vector<MultiHNSW::entity_id_t> idsResultLayer0;

            while (!resultLayer0.empty()) {
                idsResultLayer0.push_back(resultLayer0.top().second);
                resultLayer0.pop();
            }
            REQUIRE_THAT(idsResultLayer0, Catch::Matchers::RangeEquals(expectedResults0));
        }
    }

    static void testAddManyRandomEntities() {
        MultiHNSW index = MultiHNSW::Builder(2, {3, 3})
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
        MultiHNSW index = MultiHNSW::Builder(1, {1})
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
        MultiHNSW index = MultiHNSW::Builder(2, {3, 3})
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
        MultiHNSW index = MultiHNSW::Builder(2, {3, 3})
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
        index.printGraph();
        std::vector<std::vector<float>> query = {
            {3.0f, 2.0f, 3.0f}, {3.0f, 2.0f, 3.0f} // Query for both modalities
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

            std::vector<float> weights4 = {0.0f, 1.0f};
            result = index.search(query, 1, weights4);
            REQUIRE(result == std::vector<size_t>{0}); // (3, 3.01)
        }
    }
};

TEST_CASE("MultiHNSWBuilder builds", "[MultiHNSWBuilder]") {
    SECTION("Test default build") {

        MultiHNSW multiHNSW = MultiHNSW::Builder(2, {3,3})
                              .setEfSearch(50)
                              .build();

        REQUIRE(multiHNSW.getNumModalities() == 2);
        REQUIRE_THAT(multiHNSW.getDimensions(), Catch::Matchers::RangeEquals(std::vector<size_t> {3, 3}));;
        REQUIRE_THAT(multiHNSW.getWeights(), Catch::Matchers::RangeEquals(std::vector<float> {0.5, 0.5}));;

        REQUIRE(multiHNSW.getEfSearch() == 50);

        REQUIRE(multiHNSW.getDistributionScaleFactor() == 1.0);
        REQUIRE(multiHNSW.getTargetDegree() == 32);
        REQUIRE(multiHNSW.getMaxDegree() == 32);
        REQUIRE(multiHNSW.getEfConstruction() == 200);
        REQUIRE(multiHNSW.getSeed() == 42);
    }
}

TEST_CASE("MultiHNSW Basic Tests", "[MultiHNSW]") {
    MultiHNSW hnsw = MultiHNSWTest::initaliseTest1();
    SECTION("Test addToEntityStorage") {
        MultiHNSWTest::testAddToEntityStorage1(hnsw);
    }

    SECTION("Test getEntityModality") {
        MultiHNSWTest::testGetEntityModality1(hnsw);
    }

    SECTION("Test getEntityModality2") {
        MultiHNSWTest::testGetEntityModality2();
    }

    SECTION("Test computeDistanceBetweenEntities") {
        MultiHNSWTest::testComputeDistanceBetweenEntities1(hnsw);
    }

    SECTION("Test generateRandomLevel") {
        MultiHNSWTest::testGenerateRandomLevel();
    }
}

TEST_CASE("MultiHNSW SearchLayer", "[SearchLayer]") {
    MultiHNSWTest::testSearchLayer();
}

TEST_CASE("MultiHNSW AddEntities", "[AddEntities]") {
    MultiHNSWTest::testAddManyRandomEntities();
    MultiHNSWTest::testAddSingleEntities();
}

TEST_CASE("MultiHNSW Search", "[Search]") {
    MultiHNSWTest::testAddAndSearch1();
    MultiHNSWTest::testAddAndSearch2();
}