#ifndef EXACTKNNINDEX_HPP
#define EXACTKNNINDEX_HPP

#include "../../include/simple-knn/AbstractKNNIndex.hpp"

class ExactKNNIndex : public KNNIndex {
private:
    std::vector<std::vector<float>> data; // store collection of vectors

public:
    ~ExactKNNIndex() override;

    // Add a vector to the index
    void add(const std::vector<float>& vector) override;

    // Perform a k-nearest neighbor search
    [[nodiscard]] std::vector<int> search(const std::vector<float>& query, size_t k) const override;
};


#endif //EXACTKNNINDEX_HPP
