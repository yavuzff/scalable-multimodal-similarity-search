#include "cnpy.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "../include/ExactMultiIndex.hpp"
#include "../include/MultiHNSW.hpp"

std::string DATASET_ENTITY_COUNT = "20000";
std::string PREP_DATASET_PATH = "/Users/yavuz/data/LAION-" + DATASET_ENTITY_COUNT + "/";
std::string IMAGE_VECTORS_PATH = PREP_DATASET_PATH + "vectors/image_vectors.npy";
std::string IMAGE_VECTORS32_PATH = PREP_DATASET_PATH + "vectors/image_vectors32.npy";
std::string TEXT_VECTORS_PATH = PREP_DATASET_PATH + "vectors/text_vectors.npy";

int NUM_INDEX_ENTITIES = 1000;

int main() {
    // load text vectors
    cnpy::NpyArray arrTexts = cnpy::npy_load(TEXT_VECTORS_PATH);

    // print shape which is 2D: (N, d1) // 384
    std::vector<size_t> textsShape = arrTexts.shape;
    std::cout << "Texts Shape: ";
    for (size_t dim : textsShape) std::cout << dim << " ";
    std::cout << std::endl;
    if (textsShape.size() != 2) {
        throw std::runtime_error("Expected 2D shape for texts, but got: " + std::to_string(textsShape.size()));
    }

    // print word size - 4 bytes for float, 8 bytes for double
    std::cout << "Texts Word size: " << arrTexts.word_size << std::endl;

    std::span<const float> textVectors;
    if (arrTexts.word_size == 4) {
        // since the type is float (4 bytes), we can get the data directly as float
        // get std::span<const float> for the contents of the array
        textVectors = std::span<const float>(arrTexts.data<float>(), arrTexts.data<float>() + arrTexts.num_vals);
    } else {
        throw std::runtime_error("Unsupported word size for vectors: " + std::to_string(arrTexts.word_size));
    }
    assert(textVectors.size() == textsShape[0] * textsShape[1]); // e.g. (6426, 768)

    if (NUM_INDEX_ENTITIES != 0) {
        textVectors = textVectors.subspan(0, NUM_INDEX_ENTITIES * textsShape[1]);
    }

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
    } else {
        throw std::runtime_error("Unsupported word size for vectors: " + std::to_string(arrImages.word_size));
    }
    // now, create a span for the float data
    imageVectors = std::span<const float>(floatData.data(), floatData.size());
    assert(imageVectors.size() == imagesShape[0] * imagesShape[1]);
    if (NUM_INDEX_ENTITIES != 0) {
        imageVectors = imageVectors.subspan(0, NUM_INDEX_ENTITIES * imagesShape[1]);
    }

    std::vector<std::span<const float>> entities = {textVectors, imageVectors};
    std::vector<std::string> distanceMetrics = {"cosine", "cosine"};

    MultiHNSW index = MultiHNSW::Builder(2, {textsShape[1], imagesShape[1]}).setDistanceMetrics(distanceMetrics).build();
    auto start = std::chrono::high_resolution_clock::now();
    index.addEntities(entities);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Adding entities took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    ExactMultiIndex exactIndex = ExactMultiIndex(2, {textsShape[1], imagesShape[1]}, distanceMetrics);
    exactIndex.addEntities(entities);

    int id = 60;
    size_t k = 30;
    std::span<const float> query1text = textVectors.subspan(id * textsShape[1], textsShape[1]);
    std::span<const float> query1image = imageVectors.subspan(id * imagesShape[1], imagesShape[1]);
    std::vector<std::span<const float>> query1 = {query1text, query1image};

    std::vector<size_t> multiHNSWresults = index.search(query1, k);
    std::cout << "MultiHNSW search results for query " << id << ": ";
    for (size_t result : multiHNSWresults) {
        std::cout << result << " ";
    }
    std::cout << std::endl;

    std::vector<size_t> results = exactIndex.search(query1, k);
    std::cout << "Exact---- search results for query " << id << ": ";
    for (size_t result : results) {
        std::cout << result << " ";
    }
    std::cout << std::endl;

    return 0;
}
