#include <iostream>
#include <fstream>

#include "include/simple-knn/ExactKNNIndex.hpp"
#include "include/ExactMultiVecIndex.hpp"
#include "include/MultiVecHNSW.hpp"

void exact_demo() {
    ExactKNNIndex index;

    // Add some vectors
    index.add({1.0f, 2.0f});
    index.add({3.0f, 4.0f});
    index.add({5.0f, 6.0f});
    index.add({1.0f, 1.0f});

    std::vector<size_t> neighbours = index.search({1.0f, 1.0f}, 2);

    std::cout << "Indices of nearest neighbors: ";
    for (int idx : neighbours) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
}

void multivec_exact_demo() {
    // initialise an index
    size_t modalities = 2;
    std::vector<size_t> dims = {1, 2};
    std::vector<std::string> distance_metrics = {"euclidean", "euclidean"};
    std::vector<float> weights = {0.5f, 0.5f};
    ExactMultiVecIndex index(modalities, dims, distance_metrics);

    // add some entities
    index.addEntities({{1.0f}, {2.0f, 3.0f}}); // add a single entity

    // define 3 entities to add
    std::vector<std::vector<float>> entities = {{3.0f, 5.0f, 1.0f}, {4.0f, 5.0f, 1.0f, 1.0f, 100.0f, 1.0f}};
    index.addEntities(entities);

    // search for nearest neighbors
    //std::vector<size_t> neighbours = index.search({{1.0f}, {1.0f, 1.0f}}, 2,  {0.5f, 0.5f});
    std::vector<size_t> neighbours = index.search({{1.0f}, {1.0f, 1.0f}}, 2);

    std::cout << "Indices of nearest neighbors: ";
    for (size_t idx : neighbours) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
}

//multivec hnsw demo
void multivec_hnsw_demo() {
    // initialise an index
    size_t modalities = 2;
    std::vector<size_t> dims = {1, 2};
    std::vector<std::string> distance_metrics = {"euclidean", "euclidean"};
    std::vector<float> weights = {0.5f, 0.5f};
    MultiVecHNSW index(modalities, dims, distance_metrics, {}, 1.0f, 32, 32, 200, 50, 20);

    // print properties of the index
    std::cout << "Number of modalities: " << index.getNumModalities() << std::endl;
    std::cout << "Seed: " << index.getSeed() << std::endl;
    std::cout << "Target degree: " << index.getTargetDegree() << std::endl;
    std::cout << "efSearch: " << index.getEfSearch() << std::endl;

    index.addEntities({{1.0f}, {2.0f, 3.0f}}); // add a single entity
    std::vector<size_t> neighbours = index.search({{1.0f}, {1.0f, 1.0f}}, 2, {0.5f, 0.5f});
}

void serde_demo() {
    MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {2, 1, 3})
                .setDistanceMetrics({"euclidean", "manhattan", "cosine"})
                .setWeights({0.3f, 0.5f, 0.2f})
                .build();

    hnsw.addEntities({{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
                            {1.0f, 2.0f, 3.0f},
                            {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}}); // add 3 entities

    std::string path = "saved/sample.dat";
    hnsw.save(path);

    MultiVecHNSW loaded = MultiVecHNSW::Builder(1, {1})
                .setDistanceMetrics({"euclidean"})
                .setWeights({0.3f})
                .build();
    loaded.load(path);

    hnsw == loaded ? std::cout << "Equal when loading from member function" : std::cout << "Not Equal when loading from member function";
    std::cout << std::endl;

    // load with static method
    MultiVecHNSW loaded2 = MultiVecHNSW::loadIndex(path);
    loaded2 == hnsw ? std::cout << "Equal when loading from static function" : std::cout << "Not Equal when loading from static function";
    std::cout << std::endl;

    // search the graphs
    std::vector<std::vector<float>> query = {{2.0f, 3.0f}, {1.0f}, {4.0f, 5.0f, 6.0f}};
    std::vector<size_t> neighbours = hnsw.search(query, 2, {0.5f, 0.5f, 0.5f});
    std::cout << "Indices of nearest neighbors: ";
    for (size_t idx : neighbours) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::vector<size_t> loaded_neighbours = loaded.search(query, 2, {0.5f, 0.5f, 0.5f});
    std::cout << "Indices of nearest neighbors in saved index: ";
    for (size_t idx : loaded_neighbours) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    std::vector<size_t> loaded_neighbours2 = loaded2.search(query, 2, {0.5f, 0.5f, 0.5f});
    std::cout << "Indices of nearest neighbors in index loaded statically: ";
    for (size_t idx : loaded_neighbours) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
}

void minimal_serde() {
    MultiVecHNSW hnsw = MultiVecHNSW::Builder(3, {2, 1, 3}).build();
    hnsw.save("index.bin");
}

void load_with_wrong_compile_time_flags_demo() {
    // load the index into this path by running previous demo with a compile time flag
    // then, in another program run, compiled with the opposite compile time flag, run this function.
    // the function should raise an exception
    std::string path = "saved/sample.dat";
    MultiVecHNSW loaded = MultiVecHNSW::loadIndex(path);
}
int main() {

    serde_demo();
    //load_with_wrong_compile_time_flags_demo();

    return 0;
}