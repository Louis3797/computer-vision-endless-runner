#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include "../include/HOG.h"
#include "../include/HOGDescriptor.h"
#include "../include/PersonDetector.h"
//#include "../include/PersonTracker.h"

namespace py = pybind11;


PYBIND11_MODULE(tracking, m) {
    m.doc() = "Python bindings for the c++ code";

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

     py::class_<HOGDescriptor>(m, "HOGDescriptor")
        .def(py::init<int , const std::pair<int, int> &, const std::pair<int, int> &,
                  unsigned int , bool , bool , bool , bool>())
        .def("compute", &HOGDescriptor::compute, py::arg("image"));

     py::class_<PersonDetector>(m, "PersonDetector")
        .def(py::init<const std::string &,
                   HOGDescriptor &,
                   const double ,
                   const std::pair<int, int> &, const double, const float, const int, const double, const bool, const double, const double>())
        .def("detect", &PersonDetector::detect, py::arg("image"), py::arg("minBboxSize"), py::arg("dilateIterations"), py::arg("dilateKernelSize"),py::arg("openingKernelSize"),py::arg("closingKernelSize"));

}