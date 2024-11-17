//
// Created by Yavuz on 16/11/2024.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../index/ExactKNNIndex.hpp"
#include <iostream>

namespace py = pybind11;

std::vector<float> convert_to_vector(const py::object &vector) {
    //perform checks on numpy array input and convert it to std::vector<float>
    if (py::isinstance<py::array>(vector)) {
        py::array_t<float> array = vector.cast<py::array_t<float>>();
        const py::buffer_info buf = array.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Number of dimensions must be one");
        }
        if (buf.format != py::format_descriptor<float>::format()) {
            throw std::runtime_error("Data type must be float32");
        }
        return std::vector<float>((float*)buf.ptr, (float*)buf.ptr + buf.size);
    }
    else {
        throw std::runtime_error("Input must be a numpy array");
    }
}

//function to convert vector<int> to numpy array to return to python
py::array_t<int> convert_to_numpy(const std::vector<int> &result) {
    py::array_t<int> result_array(result.size());
    auto result_buffer = result_array.request();
    int *result_ptr = (int *)result_buffer.ptr;
    for (size_t i = 0; i < result.size(); i++) {
        result_ptr[i] = result[i];
    }
    return result_array;
}

class ExactIndex {
private:
    ExactKNNIndex index;

public:
    //constructor which initializes ExactKNNIndex index
    ExactIndex() : index(ExactKNNIndex()){}

    void add(const py::object &vector) {
        index.add(convert_to_vector(vector));
    };

    [[nodiscard]] py::array_t<int> search(const py::object &query, size_t k) const {
        return convert_to_numpy(index.search(convert_to_vector(query), k));
    };
};


PYBIND11_MODULE(cppindex, m) {
    m.doc() = "Exact KNN Index"; // optional module docstring
    //m.def("add", &add, "A function that adds two numbers");
    py::class_<ExactIndex>(m, "ExactIndex")
        .def(py::init<>())
        .def("add", &ExactIndex::add)
        .def("search", &ExactIndex::search);
}