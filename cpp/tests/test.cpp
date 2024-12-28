#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../include/DistanceMetrics.hpp"

#include <vector>
#include <span>
#include <cmath>

TEST_CASE("Test stringToDistanceMetric", "[DistanceMetrics]") {
    REQUIRE(stringToDistanceMetric("euclidean") == DistanceMetric::Euclidean);
    REQUIRE(stringToDistanceMetric("manhattan") == DistanceMetric::Manhattan);
}

TEST_CASE("computeEuclideanDistanceFromSlice calculates correct distance", "[euclidean]") {
    SECTION("Simple case with small vectors") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f};
        std::vector<float> queryEntity = {1.0f, 2.0f, 4.0f};

        float result = computeEuclideanDistanceFromSlice(storedEntity, 0, 3, queryEntity, 0);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(1.0f, 0.0001));
    }

    SECTION("Partial slice comparison") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> queryEntity = {4.0f, 3.0f, 2.0f, 1.0f};

        float result = computeEuclideanDistanceFromSlice(storedEntity, 1, 3, queryEntity, 1);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(2.0f), 0.0001));

    }

    SECTION("Zero distance when vectors are identical") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f};
        std::vector<float> queryEntity = {1.0f, 2.0f, 3.0f};

        float result = computeEuclideanDistanceFromSlice(storedEntity, 0, 3, queryEntity, 0);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(0.0f), 0.0001));
    }

    SECTION("Single-element slice") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f};
        std::vector<float> queryEntity = {3.0f, 6.0f, 1.0f};

        float result = computeEuclideanDistanceFromSlice(storedEntity, 2, 3, queryEntity, 1);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(9.0f), 0.0001));
    }

    SECTION("Empty slice should return 0 distance") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f};
        std::vector<float> queryEntity = {1.0f, 2.0f, 3.0f};

        float result = computeEuclideanDistanceFromSlice(storedEntity, 1, 1, queryEntity, 1);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(0.0f), 0.0001));
    }

    SECTION("Different start and end indices for query and storedEntity") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> queryEntity = {0.0f, 0.0f, 1.0f, 2.0f, 3.0f};

        float result = computeEuclideanDistanceFromSlice(storedEntity, 1, 4, queryEntity, 2);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(3.0f), 0.0001));
    }

    SECTION("Negative numbers in vectors") {
        std::vector<float> storedEntity = {-1.0f, -2.0f, -3.0f};
        std::vector<float> queryEntity = {-3.0f, -2.0f, -1.0f};

        float result = computeEuclideanDistanceFromSlice(storedEntity, 0, 3, queryEntity, 0);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(8.0f), 0.0001));
    }
}
