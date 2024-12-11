#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../include/simple-knn/ExactKNNIndex.hpp"
#include "../include/ExactMultiIndex.hpp"
#include <span>

namespace py = pybind11;

// perform checks on array-like input and convert it to std::vector<span>
// note that span means we use the same memory as the input array
std::span<float> convert_to_span(const py::object &input) {
    // casting the input to a numpy array with C-style memory layout and forcecasting
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();

    if (buffer.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }
    return std::span<float>(static_cast<float*>(buffer.ptr), buffer.size);
}


// convert std::vector<size_t> of ids to numpy array of integers
py::array_t<size_t> convert_to_numpy(const std::vector<size_t> &result) {
    // note that we are copying the data to a new location for the numpy array
    py::array_t<size_t> result_array(static_cast<py::ssize_t>(result.size()));
    const py::buffer_info result_buffer = result_array.request();
    std::memcpy(result_buffer.ptr, result.data(), result.size() * sizeof(size_t));
    return result_array;
}

class ExactKNNIndexPyWrapper {
    ExactKNNIndex index;

public:
    ExactKNNIndexPyWrapper() : index(ExactKNNIndex()){}

    void add(const py::object &vector) {
        auto span = convert_to_span(vector);
        index.add(std::vector(span.begin(), span.end()));
    };

    [[nodiscard]] py::array_t<size_t> search(const py::object &query, size_t k) const {
        auto span = convert_to_span(query);
        return convert_to_numpy(index.search(std::vector<float>(span.begin(), span.end()), k));
    };
};


class ExactMultiIndexPyWrapper {
public:
    ExactMultiIndex index;

    ExactMultiIndexPyWrapper(const size_t modalities,
                             const std::vector<size_t> &dims,
                             const std::vector<std::string> &distance_metrics,
                             const std::optional<std::vector<float>> &weights)
            : index(weights ? ExactMultiIndex(modalities, dims, distance_metrics, *weights)
                            : ExactMultiIndex(modalities, dims, distance_metrics)) {}

    void addEntities(const py::object &entities) {

    }

    // [[nodiscard]] py::array_t<int> search(const py::object &query, size_t k) const {
    //     std::vector<size_t> res = //
    //     return convert_to_numpy(res);
    // };

    void save(const std::string &path) const {
        index.save(path);
    }

    void load(const std::string &path) {
        index.load(path);
    }

    [[nodiscard]] size_t modalities() const {
        return index.modalities;
    }

    [[nodiscard]]const std::vector<size_t>& dimensions() const {
        return index.dimensions;
    }

    [[nodiscard]] const std::vector<std::string>& distance_metrics() const {
        return index.distance_metrics;
    }

    [[nodiscard]] const std::vector<float>& weights() const {
        return index.weights;
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

        .def("add_entities", &ExactMultiIndexPyWrapper::addEntities, "Adds multiple entities to the index. To add `n` entities with `k` modalities, provide a list of length `k`, where each element is a 2D numpy array of shape `(n, dimensions_of_modality)`. Each array corresponds to one modality.",
            py::arg("entities"))
        // .def("search", &ExactMultiIndexPyWrapper::search, "Returns the indices for the k-nearest neighbors of a query entity. Query should be a list of length `k`, where each element is a vector for that modality",
        //     py::arg("query"),
        //     py::arg("k"),
        //     py::arg("query_weights")=std::nullopt)

        .def("save", &ExactMultiIndexPyWrapper::save, "Method to save index", py::arg("path"))
        .def("load", &ExactMultiIndexPyWrapper::load, "Method to load index", py::arg("path"))

        // read-only attributes
        .def_property_readonly("modalities", &ExactMultiIndexPyWrapper::modalities)
        .def_property_readonly("dimensions", &ExactMultiIndexPyWrapper::dimensions)
        .def_property_readonly("distance_metrics", &ExactMultiIndexPyWrapper::distance_metrics)
        .def_property_readonly("weights", &ExactMultiIndexPyWrapper::weights);

}
