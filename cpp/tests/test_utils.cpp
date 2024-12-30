#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../include/utils.hpp"

TEST_CASE("validateAndNormaliseWeights validates and normalises weights", "[weights]") {
    SECTION("Correct weights normalization") {
        std::vector<float> weights = {1.0f, 2.0f, 3.0f};
        size_t numModalities = 3;

        validateAndNormaliseWeights(weights, numModalities);

        REQUIRE(weights.size() == numModalities);
        REQUIRE_THAT(weights[0], Catch::Matchers::WithinAbs(1.0f / 6.0f, 0.0001));
        REQUIRE_THAT(weights[1], Catch::Matchers::WithinAbs(2.0f / 6.0f, 0.0001));
        REQUIRE_THAT(weights[2], Catch::Matchers::WithinAbs(3.0f / 6.0f, 0.0001));
    }

    SECTION("Weights mismatch with modalities") {
        std::vector<float> weights = {1.0f, 2.0f};
        size_t numModalities = 3;

        REQUIRE_THROWS_AS(validateAndNormaliseWeights(weights, numModalities), std::invalid_argument);
        REQUIRE_THROWS_WITH(validateAndNormaliseWeights(weights, numModalities), "Number of weights must match number of modalities");
    }

    SECTION("Negative weights are invalid") {
        std::vector<float> weights = {1.0f, -2.0f, 3.0f};
        size_t numModalities = 3;

        REQUIRE_THROWS_AS(validateAndNormaliseWeights(weights, numModalities), std::invalid_argument);
        REQUIRE_THROWS_WITH(validateAndNormaliseWeights(weights, numModalities), "Weights must be non-negative");
    }

    SECTION("All zero weights are invalid") {
        std::vector<float> weights = {0.0f, 0.0f, 0.0f};
        size_t numModalities = 3;

        REQUIRE_THROWS_AS(validateAndNormaliseWeights(weights, numModalities), std::invalid_argument);
        REQUIRE_THROWS_WITH(validateAndNormaliseWeights(weights, numModalities), "Weights must not be all zero");
    }
}

TEST_CASE("getSpanViewOfVectors converts vectors to spans", "[span_view]") {
    SECTION("Correct span conversion") {
        std::vector<std::vector<float>> vectors = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f},
            {6.0f}
        };

        std::vector<std::span<const float>> spans = getSpanViewOfVectors(vectors);

        REQUIRE(spans.size() == vectors.size());
        for (size_t i = 0; i < vectors.size(); ++i) {
            REQUIRE(spans[i].size() == vectors[i].size());
            for (size_t j = 0; j < vectors[i].size(); ++j) {
                REQUIRE(spans[i][j]== vectors[i][j]);
            }
        }
    }
}

TEST_CASE("l2NormalizeVector normalises a vector to unit length", "[l2_normalize]") {
    SECTION("Normalizes a non-zero vector") {
        std::vector<float> vec = {3.0f, 4.0f};
        std::span<float> span(vec);

        l2NormalizeVector(span);

        REQUIRE_THAT(vec[0], Catch::Matchers::WithinAbs(0.6f, 0.0001));
        REQUIRE_THAT(vec[1], Catch::Matchers::WithinAbs(0.8f, 0.0001));
    }

    SECTION("Normalizes a single-element vector") {
        std::vector<float> vec = {5.0f};
        std::span<float> span(vec);

        l2NormalizeVector(span);

        REQUIRE_THAT(vec[0], Catch::Matchers::WithinAbs(1.0f, 0.0001));
    }

    SECTION("Normalizes sub span of vector") {
        std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::span<float> span = std::span(vec).subspan(2, 2);

        l2NormalizeVector(span);

        REQUIRE(vec[0] == 1.0f);
        REQUIRE(vec[1] == 2.0f);
        REQUIRE_THAT(vec[2], Catch::Matchers::WithinAbs(0.6f, 0.0001));
        REQUIRE_THAT(vec[3], Catch::Matchers::WithinAbs(0.8f, 0.0001));
        REQUIRE(vec[4] == 5.0f);
    }

    SECTION("Throws when normalizing a zero vector") {
        std::vector<float> vec = {0.0f, 0.0f};
        std::span<float> span(vec);

        REQUIRE_THROWS_AS(l2NormalizeVector(span), std::invalid_argument);
        REQUIRE_THROWS_WITH(l2NormalizeVector(span), "One or both input vectors have zero magnitude");
    }
}