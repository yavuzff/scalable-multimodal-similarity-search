#ifndef KNNINDEX_HPP
#define KNNINDEX_HPP

#include <vector>

// Abstract class for KNN Index
class KNNIndex {
public:
    // Virtual destructor for proper cleanup of derived classes
    virtual ~KNNIndex() = default;

    // Pure virtual function to add a vector to the index
    virtual void add(const std::vector<float>& vector) = 0;

    // Pure virtual function to search given query vector
    // Returns indices of the k-nearest vectors
    [[nodiscard]] virtual std::vector<int> search(const std::vector<float>& query, size_t k) const = 0;
};


#endif //KNNINDEX_HPP
