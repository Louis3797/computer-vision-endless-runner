#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include "../include/HOG.h"

namespace py = pybind11;


PYBIND11_MODULE(hog, m) {
    m.doc() = "Python bindings for HOG";

     py::class_<HOG>(m, "HOG")
        .def(py::init<>())
        .def_static("compute", &HOG::compute, py::arg("image"),
                    py::arg("orientations") = 9,
                    py::arg("pixels_per_cell") = std::make_pair(8, 8),
                    py::arg("cells_per_block") = std::make_pair(3, 3),
                    py::arg("method") = 4,
                    py::arg("sobel") = false,
                    py::arg("visualize") = false,
                    py::arg("normalize_input") = true,
                    py::arg("flatten") = true);
}