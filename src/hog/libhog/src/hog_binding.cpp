#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include "../include/hog.h"

namespace py = pybind11;


PYBIND11_MODULE(cv_hog, m) {
    m.doc() = "Python bindings for HOG function"; // Optional module docstring

    // Expose the hog function to Python
    m.def("hog", &hog, py::arg("image"), py::arg("orientations") = 9,
          py::arg("pixels_per_cell") = std::make_pair(8, 8),
          py::arg("cells_per_block") = std::make_pair(3, 3),
          py::arg("method") = "L2-Hys",
          py::arg("visualize") = false,
          py::arg("transform_sqrt") = true,
          py::arg("flatten") = true);
}