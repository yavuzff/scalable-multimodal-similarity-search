#include "cnpy.h"
#include <iostream>
#include <vector>

#include<string>

std::string DATASET_ENTITY_COUNT = "10000";
std::string PREP_DATASET_PATH = "/Users/yavuz/data/LAION-" + DATASET_ENTITY_COUNT + "/";
std::string IMAGE_VECTORS_PATH = PREP_DATASET_PATH + "vectors/image_vectors.npy";
std::string TEXT_VECTORS_PATH = PREP_DATASET_PATH + "vectors/text_vectors.npy";

int main() {

    // load text vectors
    cnpy::NpyArray arr = cnpy::npy_load(TEXT_VECTORS_PATH);

    // print shape
    std::vector<size_t> shape = arr.shape;
    std::cout << "Shape: ";
    for (size_t dim : shape) std::cout << dim << " ";
    std::cout << std::endl;


    // load image vectors
    cnpy::NpyArray arrImages = cnpy::npy_load(IMAGE_VECTORS_PATH);

    // print shape
    std::vector<size_t> shapeImage = arrImages.shape;
    std::cout << "Shape: ";
    for (size_t dim : shapeImage) std::cout << dim << " ";
    std::cout << std::endl;

    return 0;
}
