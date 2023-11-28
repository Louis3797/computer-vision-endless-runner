//
// Created by janra on 26.11.2023.
//
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
#ifndef TRACKING_HOG_H
#define TRACKING_HOG_H


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> hog(const Eigen::MatrixXd &image,
             int orientations,
            std::pair<int, int> pixels_per_cell ,
            std::pair<int, int> cells_per_block ,
             std::string method ,
            bool visualize ,
             bool transform_sqrt,
             bool flatten);

#endif //TRACKING_HOG_H
