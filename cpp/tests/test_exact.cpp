#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../index/include/ExactMultiVecIndex.hpp"
#include "../index/include/utils.hpp"
#include <vector>
#include <cmath>


TEST_CASE("ExactMultiVecIndex with euclidean and manhattan", "[ExactMultiVecIndex]") {
    size_t numModalities = 2;
    std::vector<size_t> dimensions = {3, 3};
    std::vector<std::string> distanceMetrics = {"euclidean", "manhattan"};
    std::vector<float> weights = {0.5f, 0.5f};

    ExactMultiVecIndex index(numModalities, dimensions, distanceMetrics, weights);

    std::vector<std::vector<float>> entities = {
        {3.0f, 1.0f, 5.0f,      1.99f, 3.0f, 4.0f},
        {3.0f, 1.0f, 5.0f,      1.99f, 3.0f, 4.0f}
    };

    index.addEntities(entities);

    SECTION("Correct initialisation") {
        REQUIRE(index.getNumModalities() == numModalities);
        REQUIRE(index.getDimensions() == dimensions);
        REQUIRE(index.getDistanceMetrics() == distanceMetrics);
        REQUIRE(index.getWeights() == weights);
        REQUIRE(index.getNumEntities() == 2);
    }

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


TEST_CASE("ExactMultiVecIndex with cosine and manhattan", "[ExactMultiVecIndex]") {
    size_t numModalities = 2;
    std::vector<size_t> dimensions = {3, 3};
    std::vector<std::string> distanceMetrics = {"cosine", "manhattan"};
    std::vector<float> weights = {0.5f, 0.5f};

    ExactMultiVecIndex index(numModalities, dimensions, distanceMetrics, weights);

    std::vector<std::vector<float>> entities = {
        {3.0f, 1.0f, 5.0f, 1.99f, 3.0f, 4.0f}, // Modality 1: 2 entities, dimension 3
        {3.0f, 1.0f, 5.0f, 1.99f, 3.0f, 4.0f}  // Modality 2: 2 entities, dimension 3
    };

    index.addEntities(entities);

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
