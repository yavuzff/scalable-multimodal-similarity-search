#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../index/ExactKNNIndex.hpp"

namespace py = pybind11;

std::vector<float> convert_to_vector(const py::object &vector) {
    // perform checks on numpy array input and convert it to std::vector<float>
    // note that it uses the same memory as the numpy array
    if (py::isinstance<py::array>(vector)) {
        const auto array = vector.cast<py::array_t<float>>();
        const py::buffer_info buf = array.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Input must be a 1D array");
        }
        if (buf.format != py::format_descriptor<float>::format()) {
            throw std::runtime_error("Data type must be float32");
        }
        return {static_cast<float*>(buf.ptr), static_cast<float*>(buf.ptr) + buf.size};
    }
    throw std::runtime_error("Input must be a numpy array");
}

py::array_t<int> convert_to_numpy(const std::vector<int> &result) {
    // convert std::vector<int> of ids to numpy array of integers
    // note that we are copying the data to a new location for the numpy array
    py::array_t<int> result_array(static_cast<py::ssize_t>(result.size()));
    const py::buffer_info result_buffer = result_array.request();
    std::memcpy(result_buffer.ptr, result.data(), result.size() * sizeof(int));
    return result_array;
}

class ExactIndex {
private:
    ExactKNNIndex index;

public:
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