//
// Created by Yavuz on 16/11/2024.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//#include "../index/add.hpp"
#include "../index/ExactKNNIndex.hpp"

namespace py = pybind11;


PYBIND11_MODULE(cppindex, m) {
    m.doc() = "Exact KNN Index"; // optional module docstring
    //m.def("add", &add, "A function that adds two numbers");
    py::class_<ExactKNNIndex>(m, "ExactKNNIndex")
        .def(py::init<>())
        .def("add", &ExactKNNIndex::add)
        .def("search", &ExactKNNIndex::search);
}