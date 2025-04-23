#ifndef SERDEUTILS_HPP
#define SERDEUTILS_HPP

#include <vector>
#include <ostream>

// contains serialisation helpers

// serde for vector<T>
template<typename T>
void serializeVector(std::ostream& os, const std::vector<T>& vec) {
    size_t size = vec.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
}

template<typename T>
void deserializeVector(std::istream& is, std::vector<T>& vec) {
    size_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    is.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
}

// serde for vector<vector<T>>
template<typename T>
void serializeNestedVector(std::ostream& os, const std::vector<std::vector<T>>& nestedVec) {
    size_t outerSize = nestedVec.size();
    os.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));
    for (const auto& inner : nestedVec) {
        serializeVector(os, inner);
    }
}

template<typename T>
void deserializeNestedVector(std::istream& is, std::vector<std::vector<T>>& nestedVec) {
    size_t outerSize;
    is.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
    nestedVec.resize(outerSize);
    for (auto& inner : nestedVec) {
        deserializeVector(is, inner);
    }
}



#endif //SERDEUTILS_HPP
