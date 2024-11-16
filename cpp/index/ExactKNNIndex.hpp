#ifndef EXACTKNNINDEX_HPP
#define EXACTKNNINDEX_HPP

#include "KNNIndex.hpp"

class ExactKNNIndex : public KNNIndex {
private:
    std::vector<std::vector<float>> data; // Stored vectors

public:
    ~ExactKNNIndex() override; // Virtual destructor

    // Add a vector to the index
    void add(const std::vector<float>& vector) override;

    // Perform a k-nearest neighbor search
    [[nodiscard]] std::vector<int> search(const std::vector<float>& query, size_t k) const override;
};


#endif //EXACTKNNINDEX_HPP
