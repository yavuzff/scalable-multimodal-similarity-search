#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/ExactMultiVecIndex.hpp"
#include "../include/MultiVecHNSW.hpp"

#include <span>

namespace py = pybind11;

// cast the input object to a numpy array with C-style memory layout and forcecasting buffer
pybind11::buffer_info to_buffer(const py::object &pyobject) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(pyobject);
    auto buffer = items.request();
    return buffer;
}

// perform checks on array-like py object and convert it to std::vector<span>
// note that span means we use the same memory as the input array
std::span<const float> convert_py_to_span(const py::object &pyArrayLike) {
    const auto buffer = to_buffer(pyArrayLike);

    if (buffer.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }
    return std::span<const float>(static_cast<float*>(buffer.ptr), buffer.size);
}

// convert a buffer to a flattened span
std::span<const float> buffer_to_flattened_span(const pybind11::buffer_info &buffer) {
    if (buffer.ndim == 1 || buffer.ndim == 2) {
        // for 2D array: return a span corresponding to the flattened version of the buffer
        // since it's stored in row-major order, we can treat it as a contiguous block of memory.
        return std::span<const float>(static_cast<const float*>(buffer.ptr), buffer.size);
    }
    throw std::runtime_error("Subarray must be a 1D or 2D array-like object");
}

// convert std::vector<size_t> of ids to numpy array of integers
py::array_t<size_t> convert_to_numpy(const std::vector<size_t> &result) {
    // note that we are copying the data to a new location for the numpy array
    py::array_t<size_t> result_array(static_cast<py::ssize_t>(result.size()));
    const py::buffer_info result_buffer = result_array.request();
    std::memcpy(result_buffer.ptr, result.data(), result.size() * sizeof(size_t));
    return result_array;
}


class ExactMultiVecIndexPyWrapper {
public:
    ExactMultiVecIndex index;

    ExactMultiVecIndexPyWrapper(const size_t numModalities,
                             const std::vector<size_t> &dims,
                             const std::vector<std::string> &distance_metrics,
                             const std::vector<float> &weights)
            : index(ExactMultiVecIndex(numModalities, dims, distance_metrics, weights)) {}

    void addEntities(const py::object &entities) {
        //input is a list of numpy arrays (1D or 2D each)
        if (!py::isinstance<py::list>(entities) && !py::isinstance<py::tuple>(entities)) {
            throw std::runtime_error("Input must be a list or tuple of numpy arrays");
        }
        std::vector<std::span<const float>> cppEntities;
        for (const auto& modality: entities) {
            auto buffer = to_buffer(py::cast<py::object>(modality));
            if (buffer.ndim == 2) {
                // if we are adding a 2d array, shape should be (number of entities, dimensions of the modality)
                if (static_cast<size_t>(buffer.shape[1]) != index.getDimensions()[cppEntities.size()]) {
                    throw std::runtime_error("Modality " + std::to_string(cppEntities.size()) + " has incorrect data size: " +
                                             std::to_string(buffer.shape[1]) + " is not equal to the expected dimension " + std::to_string(index.getDimensions()[cppEntities.size()]));
                }
            }
            std::span<const float> span = buffer_to_flattened_span(buffer);
            // copy the span to a vector as we append it to the cppEntities
            cppEntities.emplace_back(span);
        }
        index.addEntities(cppEntities);
    }

    [[nodiscard]] py::array_t<int> search(const py::object &query, const int k, const std::vector<float> &query_weights) {
        if (k < 1) {
            throw std::invalid_argument("k must be at least 1");
        }
        std::vector<std::span<const float>> vecQuery;
        if (py::isinstance<py::list>(query) || py::isinstance<py::tuple>(query)) {
            for (const auto& modalityVector: query) {
                std::span<const float> span = convert_py_to_span(py::cast<py::object>(modalityVector));
                // copy to a vector as we append it to the cppEntities
                vecQuery.emplace_back(span);
            }
        } else {
            throw std::runtime_error("Input must be a list or tuple of numpy arrays");
        }

        const std::vector<size_t> res = query_weights.empty() ? index.search(vecQuery, k) : index.search(vecQuery, k, query_weights);
        return convert_to_numpy(res);
    };

    void save(const std::string &path) const {
        index.save(path);
    }

    void load(const std::string &path) {
        index.load(path);
    }

    [[nodiscard]] size_t numModalities() const {
        return index.getNumModalities();
    }

    [[nodiscard]]const std::vector<size_t>& dimensions() const {
        return index.getDimensions();
    }

    [[nodiscard]] const std::vector<std::string>& distance_metrics() const {
        return index.getDistanceMetrics();
    }

    [[nodiscard]] const std::vector<float>& weights() const {
        return index.getWeights();
    }

    [[nodiscard]] size_t numEntities() const {
        return index.getNumEntities();
    }
};


class MultiVecHNSWPyWrapper {
public:
    MultiVecHNSW index;

    MultiVecHNSWPyWrapper(const size_t numModalities,
                                const std::vector<size_t> &dims,
                                const std::vector<std::string> &distanceMetrics,
                                const std::vector<float> &weights,
                                float distributionScaleFactor,
                                size_t targetDegree,
                                size_t maxDegree,
                                size_t efConstruction,
                                size_t efSearch,
                                size_t seed) : index(MultiVecHNSW(numModalities, dims, distanceMetrics, weights, distributionScaleFactor, targetDegree, maxDegree, efConstruction, efSearch, seed)) {}

    void addEntities(const py::object &entities) {
        //input is a list of numpy arrays (1D or 2D each)
        if (!py::isinstance<py::list>(entities) && !py::isinstance<py::tuple>(entities)) {
            throw std::runtime_error("Input must be a list or tuple of numpy arrays");
        }
        std::vector<std::span<const float>> cppEntities;
        for (const auto& modality: entities) {
            auto buffer = to_buffer(py::cast<py::object>(modality));
            if (buffer.ndim == 2) {
                // if we are adding a 2d array, shape should be (number of entities, dimensions of the modality)
                if (static_cast<size_t>(buffer.shape[1]) != index.getDimensions()[cppEntities.size()]) {
                    throw std::runtime_error("Modality " + std::to_string(cppEntities.size()) + " has incorrect data size: " +
                                             std::to_string(buffer.shape[1]) + " is not equal to the expected dimension " + std::to_string(index.getDimensions()[cppEntities.size()]));
                }
            }
            std::span<const float> span = buffer_to_flattened_span(buffer);
            // copy the span to a vector as we append it to the cppEntities
            cppEntities.emplace_back(span);
        }
        index.addEntities(cppEntities);
    }

    [[nodiscard]] py::array_t<int> search(const py::object &query, const int k, const std::vector<float> &query_weights) {
        if (k < 1) {
            throw std::invalid_argument("k must be at least 1");
        }
        std::vector<std::span<const float>> vecQuery;
        if (py::isinstance<py::list>(query) || py::isinstance<py::tuple>(query)) {
            for (const auto& modalityVector: query) {
                std::span<const float> span = convert_py_to_span(py::cast<py::object>(modalityVector));
                // copy to a vector as we append it to the cppEntities
                vecQuery.emplace_back(span);
            }
        } else {
            throw std::runtime_error("Input must be a list or tuple of numpy arrays");
        }

        const std::vector<size_t> res = query_weights.empty() ? index.search(vecQuery, k) : index.search(vecQuery, k, query_weights);
        return convert_to_numpy(res);
    };


    void setEfSearch(const size_t efSearch) {
        index.setEfSearch(efSearch);
    }

    void printGraph() const {
        index.printGraph();
    }

    void save(const std::string &path) const {
        index.save(path);
    }

    void load(const std::string &path) {
        index.load(path);
    }

    [[nodiscard]] size_t numModalities() const {
        return index.getNumModalities();
    }

    [[nodiscard]]const std::vector<size_t>& dimensions() const {
        return index.getDimensions();
    }

    [[nodiscard]] const std::vector<std::string>& distance_metrics() const {
        return index.getDistanceMetrics();
    }

    [[nodiscard]] const std::vector<float>& weights() const {
        return index.getWeights();
    }

    [[nodiscard]] size_t numEntities() const {
        return index.getNumEntities();
    }

    [[nodiscard]] float distributionScaleFactor() const {
        return index.getDistributionScaleFactor();
    }

    [[nodiscard]] size_t targetDegree() const {
        return index.getTargetDegree();
    }

    [[nodiscard]] size_t maxDegree() const {
        return index.getMaxDegree();
    }

    [[nodiscard]] size_t efConstruction() const {
        return index.getEfConstruction();
    }

    [[nodiscard]] size_t efSearch() const {
        return index.getEfSearch();
    }

    [[nodiscard]] size_t seed() const {
        return index.getSeed();
    }

    [[nodiscard]] unsigned long long num_compute_distance_calls() const {
        return index.getNumComputeDistanceCalls();
    }

    [[nodiscard]] unsigned long long num_lazy_distance_calls() const {
        return index.getNumLazyDistanceCalls();
    }

    [[nodiscard]] unsigned long long num_lazy_distance_cutoff() const {
        return index.getNumLazyDistanceCutoff();
    }

    [[nodiscard]] unsigned long long num_vectors_skipped_due_to_cutoff() const {
        return index.getNumVectorsSkippedDueToCutoff();
    }

    void resetStats() {
        index.resetStats();
    }


};


PYBIND11_MODULE(multivec_index, m) {
    m.doc() = "This module contains different knn indexes: ExactIndex, ExactMultiVecIndex"; // optional module docstring

    py::class_<ExactMultiVecIndexPyWrapper>(m, "ExactMultiVecIndex")
        // note that pybind11/stl.h automatic conversions occur here, which copy these vectors - this is fine for initialisation
        .def(py::init<size_t, const std::vector<size_t>&, const std::vector<std::string>&, const std::vector<float>&>(), py::arg("num_modalities"), py::arg("dimensions"), py::arg("distance_metrics")=std::vector<std::string>(), py::arg("weights")=std::vector<float>())

        .def("add_entities", &ExactMultiVecIndexPyWrapper::addEntities, "Adds multiple entities to the index. To add `n` entities with `k` modalities, provide a list of length `k`, where each element is a 2D numpy array of shape `(n, dimensions_of_modality)`. Each array corresponds to one modality.",
            py::arg("entities"))
        .def("search", &ExactMultiVecIndexPyWrapper::search, "Returns the indices for the k-nearest neighbors of a query entity. Query should be a list of length `k`, where each element is a vector for that modality",
            py::arg("query"),
            py::arg("k"),
            py::arg("query_weights")=std::vector<float>())

        .def("save", &ExactMultiVecIndexPyWrapper::save, "Method to save index", py::arg("path"))
        .def("load", &ExactMultiVecIndexPyWrapper::load, "Method to load index", py::arg("path"))

        // read-only attributes
        .def_property_readonly("num_modalities", &ExactMultiVecIndexPyWrapper::numModalities)
        .def_property_readonly("dimensions", &ExactMultiVecIndexPyWrapper::dimensions)
        .def_property_readonly("distance_metrics", &ExactMultiVecIndexPyWrapper::distance_metrics)
        .def_property_readonly("weights", &ExactMultiVecIndexPyWrapper::weights)
        .def_property_readonly("num_entities", &ExactMultiVecIndexPyWrapper::numEntities);


    py::class_<MultiVecHNSWPyWrapper>(m, "MultiVecHNSW")
       // note that pybind11/stl.h automatic conversions occur here, which copy these vectors - this is fine for initialisation
        .def(py::init<size_t, const std::vector<size_t>&, const std::vector<std::string>&, const std::vector<float>&, float, size_t, size_t, size_t, size_t, size_t>(), py::arg("num_modalities"), py::arg("dimensions"), py::arg("distance_metrics")=std::vector<std::string>(), py::arg("weights")=std::vector<float>(),
            py::arg("distribution_scale_factor") = 0.0f, py::arg("target_degree") = 32, py::arg("max_degree") = 32, py::arg("ef_construction") = 200, py::arg("ef_search") = 50, py::arg("seed") = 42)

        .def("add_entities", &MultiVecHNSWPyWrapper::addEntities, "Adds multiple entities to the index. To add `n` entities with `k` modalities, provide a list of length `k`, where each element is a 2D numpy array of shape `(n, dimensions_of_modality)`. Each array corresponds to one modality.",
           py::arg("entities"))
        .def("search", &MultiVecHNSWPyWrapper::search, "Returns the indices for the k-nearest neighbors of a query entity. Query should be a list of length `k`, where each element is a vector for that modality",
           py::arg("query"),
           py::arg("k"),
           py::arg("query_weights")=std::vector<float>())

        .def("set_ef_search", &MultiVecHNSWPyWrapper::setEfSearch, "Set the efSearch parameter", py::arg("ef_search"))
        .def("print_graph", &MultiVecHNSWPyWrapper::printGraph, "Print the HNSW graph structure")

        .def("save", &MultiVecHNSWPyWrapper::save, "Method to save index", py::arg("path"))
        .def("load", &MultiVecHNSWPyWrapper::load, "Method to load index", py::arg("path"))

        .def("reset_stats", &MultiVecHNSWPyWrapper::resetStats, "Reset the statistics of the index")

        // read-only attributes
        .def_property_readonly("num_modalities", &MultiVecHNSWPyWrapper::numModalities)
        .def_property_readonly("dimensions", &MultiVecHNSWPyWrapper::dimensions)
        .def_property_readonly("distance_metrics", &MultiVecHNSWPyWrapper::distance_metrics)
        .def_property_readonly("weights", &MultiVecHNSWPyWrapper::weights)
        .def_property_readonly("num_entities", &MultiVecHNSWPyWrapper::numEntities)
        .def_property_readonly("distribution_scale_factor", &MultiVecHNSWPyWrapper::distributionScaleFactor)
        .def_property_readonly("target_degree", &MultiVecHNSWPyWrapper::targetDegree)
        .def_property_readonly("max_degree", &MultiVecHNSWPyWrapper::maxDegree)
        .def_property_readonly("ef_construction", &MultiVecHNSWPyWrapper::efConstruction)
        .def_property_readonly("ef_search", &MultiVecHNSWPyWrapper::efSearch)
        .def_property_readonly("seed", &MultiVecHNSWPyWrapper::seed)
        // add methods for stats
        .def_property_readonly("num_compute_distance_calls", &MultiVecHNSWPyWrapper::num_compute_distance_calls)
        .def_property_readonly("num_lazy_distance_calls", &MultiVecHNSWPyWrapper::num_lazy_distance_calls)
        .def_property_readonly("num_lazy_distance_cutoff", &MultiVecHNSWPyWrapper::num_lazy_distance_cutoff)
        .def_property_readonly("num_vectors_skipped_due_to_cutoff", &MultiVecHNSWPyWrapper::num_vectors_skipped_due_to_cutoff);
}
