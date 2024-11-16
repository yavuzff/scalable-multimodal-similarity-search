//
// Created by Yavuz on 16/11/2024.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../index/add.hpp"

namespace py = pybind11;


PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}