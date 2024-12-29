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

        float result = computeEuclideanDistance(std::span(storedEntity), std::span(queryEntity));
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(1.0f, 0.0001));
    }

    SECTION("Partial slice comparison") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> queryEntity = {4.0f, 3.0f, 2.0f, 1.0f};

        float result = computeEuclideanDistance(
            std::span(storedEntity).subspan(1, 2),
            std::span(queryEntity).subspan(1, 2));
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(2.0f), 0.0001));
    }

    SECTION("Zero distance when vectors are identical") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f};
        std::vector<float> queryEntity = {1.0f, 2.0f, 3.0f};

        float result = computeEuclideanDistance(std::span(storedEntity), std::span(queryEntity));
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(0.0f), 0.0001));
    }

    SECTION("Different start and end indices for query and storedEntity") {
        std::vector<float> storedEntity = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> queryEntity = {0.0f, 0.0f, 1.0f, 2.0f, 3.0f};

        float result = computeEuclideanDistance(
            std::span(storedEntity).subspan(1,3),
            std::span(queryEntity).subspan(2,3));
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(3.0f), 0.0001));
    }

    SECTION("Negative numbers in vectors") {
        std::vector<float> storedEntity = {-1.0f, -2.0f, -3.0f};
        std::vector<float> queryEntity = {-3.0f, -2.0f, -1.0f};

        float result = computeEuclideanDistance(std::span(storedEntity), std::span(queryEntity));
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(std::sqrt(8.0f), 0.0001));
    }
}

TEST_CASE("computeDotProduct calculates correct dot product", "[dot_product]") {
    SECTION("Simple case from slice") {
        std::vector<float> vector1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> vector2 = {4.0f, 5.0f, 6.0f, 7.0f};

        float result = computeDotProduct(std::span(vector1).subspan(2,2), std::span(vector2).subspan(0,2));
        REQUIRE(result == 32.0f);
    }
}

TEST_CASE("computeManhattanDistance calculates correct Manhattan distance", "[manhattan]") {
    SECTION("Simple case from slice") {
        std::vector<float> vector1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> vector2 = {4.0f, 5.0f, 6.0f, 7.0f};

        float result = computeManhattanDistance(std::span(vector1).subspan(1,2), std::span(vector2).subspan(2,2));
        REQUIRE(result == 8.0f);
    }
}

TEST_CASE("computeCosineDistance calculates correct cosine distance", "[cosine_distance]") {

    SECTION("Normalised identical vectors") {
        float value = static_cast<float>(1.0 / std::sqrt(2));
        std::vector<float> vector1 = {value, value};
        std::vector<float> vector2 = {value, value};

        float result = computeCosineDistance(std::span(vector1), std::span(vector2), true);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(0, 0.0001));
    }

    SECTION("Normalised opposite vectors span") {
        float value = static_cast<float>(1.0 / std::sqrt(2));
        std::vector<float> vector1 = {0.0f, 0.0f, value, value};
        std::vector<float> vector2 = {0.0f,  -value, -value};

        float result = computeCosineDistance(
            std::span(vector1).subspan(2, 2),
            std::span(vector2).subspan(1, 2),
            true);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(2, 0.0001));
    }

    SECTION("Normalised orthogonal vectors") {
        std::vector<float> vector1 = {1.0f, 0.0f};
        std::vector<float> vector2 = {0.f, 1.0f};

        float result = computeCosineDistance(std::span(vector1), std::span(vector2), true);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(1, 0.0001));
    }

    SECTION("Non-normalised vectors") {
        std::vector<float> vector1 = {3.0f, 4.0f, 5.0f};
        std::vector<float> vector2 = {-2.0f, -4.0f, 6.0f};

        float result = computeCosineDistance(std::span(vector1), std::span(vector2), false);
        REQUIRE_THAT(result, Catch::Matchers::WithinAbs(0.848814, 0.0001));
    }
}
