#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include "../include/hoghistogram.h"
#include "../include/hog.h"
#include "../include/draw_new.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


using cv::Mat;

std::pair<cv::Mat, cv::Mat> compute_gradients(const cv::Mat &inputImage) {
    cv::Mat grad_x, grad_y;

    // Compute gradients using Prewitt operator
    cv::Mat kernel_x = (cv::Mat_<double>(3, 3) << -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1);

    cv::Mat kernel_y = (cv::Mat_<double>(3, 3) << -1, -1, -1,
            0, 0, 0,
            1, 1, 1);

    cv::filter2D(inputImage, grad_x, -1, kernel_x);
    cv::filter2D(inputImage, grad_y, -1, kernel_y);

    return std::make_pair(grad_y, grad_x);
}


cv::Mat normalize_image(const Mat &image) {
    Mat normalized_image;

    Mat converted_image;
    image.convertTo(converted_image, CV_64F); // Convert to double type
    cv::sqrt(converted_image, normalized_image);

    return normalized_image;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> hog(const Eigen::MatrixXd &image,
                            int orientations = 9,
                            std::pair<int, int> pixels_per_cell = std::make_pair(8, 8),
                            std::pair<int, int> cells_per_block = std::make_pair(3, 3),
                            std::string method = "L2-Hys",
                            bool visualize = false,
                            bool transform_sqrt = true,
                            bool flatten = true) {

    Mat mat_image;

    cv::eigen2cv(image, mat_image);

    Mat transformed_image = transform_sqrt ? normalize_image(mat_image) : mat_image;

    auto pair = compute_gradients(transformed_image);
    Mat g_row = pair.first;
    Mat g_col = pair.second;

    if (g_row.empty() || g_col.empty()) {
        throw std::runtime_error("Computed gradient matrices are empty!");
    }

    if (g_row.type() != CV_64F) {
        g_row.convertTo(g_row, CV_64F);
    }

    if (g_col.type() != CV_64F) {
        g_col.convertTo(g_col, CV_64F);
    }

    if (g_row.size() != g_col.size()) {
        throw std::runtime_error("Gradient matrices have mismatched sizes!");
    }


    int s_row = mat_image.rows;
    int s_col = mat_image.cols;

    int c_row = pixels_per_cell.first;
    int c_col = pixels_per_cell.second;

    int b_row = cells_per_block.first;
    int b_col = cells_per_block.second;

    int n_cells_row = s_row / c_row;  // number of cells along row-axis
    int n_cells_col = s_col / c_col;  // number of cells along col-axis

    // Assuming n_cells_row, n_cells_col, and orientations are known at runtime
    Mat orientation_histogram = Mat::zeros(n_cells_row, n_cells_col * orientations, CV_64F);


    hog_histograms(g_col,
                   g_row,
                   c_col,
                   c_row,
                   s_col,
                   s_row,
                   n_cells_col,
                   n_cells_row,
                   orientations,
                   orientation_histogram);

    int radius = std::min(c_row, c_col) / 2 - 1;

    std::vector<int> orientations_arr(orientations);
    for (int i = 0; i < orientations; ++i) {
        orientations_arr[i] = i;
    }

    std::vector<double> orientation_bin_midpoints(orientations);
    for (int i = 0; i < orientations; ++i) {
        orientation_bin_midpoints[i] = CV_PI * (i + 0.5) / orientations;
    }

    std::vector<double> dr_arr(orientations);
    std::vector<double> dc_arr(orientations);
    for (int i = 0; i < orientations; ++i) {
        dr_arr[i] = radius * std::sin(orientation_bin_midpoints[i]);
        dc_arr[i] = radius * std::cos(orientation_bin_midpoints[i]);
    }

    Mat hog_image = Mat::zeros(s_row, s_col, CV_64F);

    if (visualize) {
        for (int r = 0; r < n_cells_row; ++r) {
            for (int c = 0; c < n_cells_col; ++c) {
                for (int i = 0; i < orientations_arr.size(); ++i) {
                    int o = orientations_arr[i];
                    double dr = dr_arr[i];
                    double dc = dc_arr[i];

                    std::pair<double, double> centre = std::make_pair(r * c_row + c_row / 2, c * c_col + c_col / 2);


                    // Assuming `line` returns a pair of integers as rr and cc
                    auto lines = line(
                            static_cast<int>(centre.first - dc),
                            static_cast<int>(centre.second + dr),
                            static_cast<int>(centre.first + dc),
                            static_cast<int>(centre.second - dr)
                    );


                    auto &rows_array = lines.first;
                    auto &cols_array = lines.second;

                    for (int g = 0; g < rows_array.size(); ++g) {
                        int rr = rows_array[g];
                        int cc = cols_array[g];

                        // Ensure rr and cc are within bounds before accessing hog_image
                        if (rr >= 0 && rr < s_row && cc >= 0 && cc < s_col) {
                            hog_image.at<double>(rr, cc) += orientation_histogram.at<double>(r, c * orientations + o);
                        }
                    }
                }
            }
        }
    }


    int n_blocks_row = (n_cells_row - b_row) + 1;
    int n_blocks_col = (n_cells_col - b_col) + 1;

    if (n_blocks_col <= 0 || n_blocks_row <= 0) {
        int min_row = b_row * c_row;
        int min_col = b_col * c_col;
        throw std::invalid_argument(
                "The input image is too small given the values of pixels_per_cell and cells_per_block. "
                "It should have at least: " + std::to_string(min_row) + " rows and " +
                std::to_string(min_col) + " cols.");
    }


    cv::Mat normalized_blocks;

    for (int r = 0; r < n_blocks_row; ++r) {
        for (int c = 0; c < n_blocks_col; ++c) {
            cv::Rect blockRect(c * orientations, r, b_col * orientations, b_row);
            cv::Mat block = orientation_histogram(blockRect);
            cv::normalize(block, block, 1.0, 0.0, cv::NORM_L2, 6, cv::noArray());
            normalized_blocks.push_back(block);
        }
    }


    if (flatten) {
        normalized_blocks = normalized_blocks.reshape(1, 1);
        cv::transpose(normalized_blocks, normalized_blocks);
    }

    Eigen::MatrixXd eigen_normalized_blocks;

    cv::cv2eigen(normalized_blocks, eigen_normalized_blocks);

    Eigen::MatrixXd eigen_hog_image;
    cv::cv2eigen(hog_image, eigen_hog_image);

    return std::make_pair(eigen_normalized_blocks, eigen_hog_image);


}




