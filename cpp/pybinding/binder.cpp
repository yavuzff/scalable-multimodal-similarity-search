#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../index/simple-knn/ExactKNNIndex.hpp"
#include "../index/ExactMultiIndex.hpp"

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

class ExactKNNIndexPyWrapper {
    ExactKNNIndex index;

public:
    ExactKNNIndexPyWrapper() : index(ExactKNNIndex()){}

    void add(const py::object &vector) {
        index.add(convert_to_vector(vector));
    };

    [[nodiscard]] py::array_t<int> search(const py::object &query, size_t k) const {
        return convert_to_numpy(index.search(convert_to_vector(query), k));
    };
};


class ExactMultiIndexPyWrapper {
    ExactMultiIndex index;

public:

    ExactMultiIndexPyWrapper(const size_t modalities,
                             const std::vector<size_t> &dims,
                             const std::vector<std::string> &distance_metrics,
                             const std::optional<std::vector<float>> &weights)
            : index(weights ? ExactMultiIndex(modalities, dims, distance_metrics, *weights)
                            : ExactMultiIndex(modalities, dims, distance_metrics)) {}


    void save(const std::string &path) const {
        index.save(path);
    }

    void load(const std::string &path) {
        index.load(path);
    }
};



PYBIND11_MODULE(cppindex, m) {
    m.doc() = "This module contains different knn indexes: ExactIndex, ExactMultiIndex"; // optional module docstring

    // simple exact index
    py::class_<ExactKNNIndexPyWrapper>(m, "ExactIndex")
        .def(py::init<>())
        .def("add", &ExactKNNIndexPyWrapper::add)
        .def("search", &ExactKNNIndexPyWrapper::search);


    py::class_<ExactMultiIndexPyWrapper>(m, "ExactMultiIndex")
        // note that pybind11/stl.h automatic conversions occur here, which copy these vectors - this is fine for initialisation
        .def(py::init<size_t, const std::vector<size_t>&, const std::vector<std::string>&, const std::optional<std::vector<float>>&>(), py::arg("modalities"), py::arg("dims"), py::arg("distance_metrics"), py::arg("weights")=std::nullopt)

        .def("save", &ExactMultiIndexPyWrapper::save, "Method to save index", py::arg("path"))
        .def("load", &ExactMultiIndexPyWrapper::load, "Method to load index", py::arg("path"));

}
