#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../include/MultiHNSW.hpp"
#include "../include/utils.hpp"

class MultiHNSWTest {
public:
    static MultiHNSW initaliseTest1() {
        size_t numModalities = 2;
        std::vector<size_t> dimensions = {3, 3};
        std::vector<std::string> distanceMetrics = {"euclidean", "manhattan"};
        std::vector<float> weights = {0.5f, 0.5f};

        MultiHNSW hnsw(numModalities, dimensions, distanceMetrics, weights);
        return hnsw;
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
        size_t numModalities = 3;
        std::vector<size_t> dimensions = {1, 2, 3};
        std::vector<std::string> distanceMetrics = {"euclidean", "manhattan", "cosine"};
        std::vector<float> weights = {0.3f, 0.5f, 0.2f};

        MultiHNSW hnsw(numModalities, dimensions, distanceMetrics, weights);

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

};


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

}
