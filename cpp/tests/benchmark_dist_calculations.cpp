#include "cnpy.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "../include/DistanceMetrics.hpp"


std::string DATASET_ENTITY_COUNT = "20000";
std::string PREP_DATASET_PATH = "/Users/yavuz/data/LAION-" + DATASET_ENTITY_COUNT + "/";
std::string IMAGE_VECTORS_PATH = PREP_DATASET_PATH + "vectors/image_vectors.npy";
std::string IMAGE_VECTORS32_PATH = PREP_DATASET_PATH + "vectors/image_vectors32.npy";
std::string TEXT_VECTORS_PATH = PREP_DATASET_PATH + "vectors/text_vectors.npy";

int NUM_INDEX_ENTITIES = 1000;


int main() {
    // load image vectors
    cnpy::NpyArray arrImages = cnpy::npy_load(IMAGE_VECTORS_PATH);

    // print shape which is 2D: (N, d1) // 768
    std::vector<size_t> imagesShape = arrImages.shape;
    std::cout << "Images Shape: ";
    for (size_t dim : imagesShape) std::cout << dim << " ";
    std::cout << std::endl;
    if (imagesShape.size() != 2) {
        throw std::runtime_error("Expected 2D shape for images, but got: " + std::to_string(imagesShape.size()));
    }

    // print word size - 4 bytes for float, 8 bytes for double
    std::cout << "Images Word size: " << arrImages.word_size << std::endl;

    // define float data outside the for loop to ensure the data is in scope throughout the whole function
    std::vector<float> floatData;
    std::span<const float> imageVectors;
    if (arrImages.word_size == 4) {
        imageVectors = std::span<const float>(arrImages.data<float>(), arrImages.data<float>() + arrImages.num_vals);
        //throw std::runtime_error("Image vectors were not expected to be 4 bytes: " + std::to_string(arrImages.word_size));
    } else if (arrImages.word_size == 8) {
        std::cout << "Converting double array to float array by allocating new memory" << std::endl;
        // gain access to the double data
        std::span<const double> doubleData(arrImages.data<double>(), arrImages.num_vals);
        assert(doubleData.size() == arrImages.num_vals);

        // cast each double element to float and create a span
        floatData.resize(doubleData.size());
        std::transform(doubleData.begin(), doubleData.end(), floatData.begin(),
                       [](double val) { return static_cast<float>(val); });
        // now, create a span for the float data
        imageVectors = std::span<const float>(floatData.data(), floatData.size());
    } else {
        throw std::runtime_error("Unsupported word size for vectors: " + std::to_string(arrImages.word_size));
    }
    assert(imageVectors.size() == imagesShape[0] * imagesShape[1]);
    if (NUM_INDEX_ENTITIES != 0) {
        imageVectors = imageVectors.subspan(0, NUM_INDEX_ENTITIES * imagesShape[1]);
    }

    std::span<const float> query = imageVectors.subspan(0, imagesShape[1]);

    // measure the average time in milliseconds to compute the distance between two vectors
    size_t numTrials = 10000;

    // computeEuclideanDistance
    std::chrono::duration<double> totalDuration(0);
    for (size_t i = 0; i < numTrials; i++) {
        std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
        auto start = std::chrono::high_resolution_clock::now();
        computeEuclideanDistance(query, vec);
        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += end - start;
    }
    std::cout << "Average time computeEuclideanDistance: "
              << totalDuration.count() * 1000 << "ms" << std::endl;

    // computeEuclideanDistancePragma
    totalDuration = std::chrono::duration<double>(0);
    for (size_t i = 0; i < numTrials; i++) {
        std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
        auto start = std::chrono::high_resolution_clock::now();
        computeEuclideanDistancePragma(query, vec);
        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += end - start;
    }
    std::cout << "Average time computeEuclideanDistancePragma: "
              << totalDuration.count() * 1000 << "ms" << std::endl;

    // computeEuclideanDistanceParUnseq
    totalDuration = std::chrono::duration<double>(0);
    for (size_t i = 0; i < numTrials; i++) {
        std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
        auto start = std::chrono::high_resolution_clock::now();
        computeEuclideanDistanceParUnseq(query, vec);
        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += end - start;
    }
    std::cout << "Average time computeEuclideanDistanceParUnseq: "
              << totalDuration.count() * 1000 << "ms" << std::endl;


    // computeEuclideanEigenMap
    totalDuration = std::chrono::duration<double>(0);
    for (size_t i = 0; i < numTrials; i++) {
        std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
        auto start = std::chrono::high_resolution_clock::now();
        computeEuclideanEigenMap(query, vec);
        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += end - start;
    }
    std::cout << "Average time computeEuclideanEigenMap: "
              << totalDuration.count() * 1000 << "ms" << std::endl;


    //computeEuclideanPureEigen
    totalDuration = std::chrono::duration<double>(0);
    Eigen::VectorXf eigenQuery = Eigen::Map<const Eigen::VectorXf>(query.data(), query.size());
    for (size_t i = 0; i < numTrials; i++) {
        std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
        Eigen::VectorXf eigenVec = Eigen::Map<const Eigen::VectorXf>(vec.data(), vec.size());
        auto start = std::chrono::high_resolution_clock::now();
        computeEuclideanPureEigen(eigenQuery, eigenVec);
        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += end - start;
    }
    std::cout << "Average time computeEuclideanPureEigen: "
          << totalDuration.count() * 1000 << "ms" << std::endl;

    //computeEuclideanPureEigen inline
    totalDuration = std::chrono::duration<double>(0);
    eigenQuery = Eigen::Map<const Eigen::VectorXf>(query.data(), query.size());
    for (size_t i = 0; i < numTrials; i++) {
        std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
        Eigen::VectorXf eigenVec = Eigen::Map<const Eigen::VectorXf>(vec.data(), vec.size());
        auto start = std::chrono::high_resolution_clock::now();
        assert(eigenVec.size() == eigenQuery.size());
        (eigenVec - eigenQuery).norm();
        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += end - start;
    }
    std::cout << "Average time computeEuclideanPureEigen inline: "
          << totalDuration.count() * 1000 << "ms" << std::endl;

    {
        //computeEuclideanEigenMap inline
        totalDuration = std::chrono::duration<double>(0);
        for (size_t i = 0; i < numTrials; i++) {
            std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
            auto start = std::chrono::high_resolution_clock::now();
            Eigen::Map<const Eigen::VectorXf> eigenMapQuery(query.data(), query.size());
            Eigen::Map<const Eigen::VectorXf> eigenVec(vec.data(), vec.size());
            assert(eigenVec.size() == eigenMapQuery.size());
            (eigenVec - eigenMapQuery).norm();
            auto end = std::chrono::high_resolution_clock::now();
            totalDuration += end - start;
        }
        std::cout << "Average time computeEuclideanEigenMap inline: "
              << totalDuration.count() * 1000 << "ms" << std::endl;
    }

    {
        //cosine
        totalDuration = std::chrono::duration<double>(0);
        for (size_t i = 0; i < numTrials; i++) {
            std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
            auto start = std::chrono::high_resolution_clock::now();
            cosine(query, vec);
            auto end = std::chrono::high_resolution_clock::now();
            totalDuration += end - start;
        }
        std::cout << "Average time cosine: "
              << totalDuration.count() * 1000 << "ms" << std::endl;
    }

    {
        //manhattan
        totalDuration = std::chrono::duration<double>(0);
        for (size_t i = 0; i < numTrials; i++) {
            std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
            auto start = std::chrono::high_resolution_clock::now();
            manhattan(query, vec);
            auto end = std::chrono::high_resolution_clock::now();
            totalDuration += end - start;
        }
        std::cout << "Average time manhattan: "
              << totalDuration.count() * 1000 << "ms" << std::endl;
    }

    {
        //euclidean
        totalDuration = std::chrono::duration<double>(0);
        for (size_t i = 0; i < numTrials; i++) {
            std::span<const float> vec = imageVectors.subspan(i * imagesShape[1], imagesShape[1]);
            auto start = std::chrono::high_resolution_clock::now();
            euclidean(query, vec);
            auto end = std::chrono::high_resolution_clock::now();
            totalDuration += end - start;
        }
        std::cout << "Average time euclidean: "
              << totalDuration.count() * 1000 << "ms" << std::endl;
    }


}
