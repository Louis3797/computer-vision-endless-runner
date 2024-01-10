//
// Created by Louis-Kaan Ay on 08.12.23.
//

#ifndef TRACKING_PERSONDETECTOR_H
#define TRACKING_PERSONDETECTOR_H

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <numeric>
#include <opencv2/bgsegm.hpp>

#include "../include/HOGDescriptor.h"
#include "../include/SlidingWindow.h"
#include "../include/NonMaxSuppression.h"
#include "../include/ImagePyramid.h"

using namespace std;

class PersonDetector {
private:
    cv::Ptr<cv::ml::SVM> svm_model_1;
    cv::Ptr<cv::ml::SVM> svm_model_2;

    HOGDescriptor hogDescriptor;
    double scale_factor;
    cv::Size window_size;
    cv::Size stepSize;
    double detection_threshold_1;
    double detection_threshold_2;
    float overlap_threshold;
    double downscale;
    cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(200, 16, true);

    void loadModels(const std::string &svm_model_1_path, const std::string &svm_model_2_path) {
        try {
            svm_model_1 = cv::ml::SVM::load(
                    svm_model_1_path);

        } catch (cv::Exception &e) {
            std::cerr << "There was an error while loading the svm_model_1\n" << e.what() << std::endl;
        }

        try {
            svm_model_2 = cv::ml::SVM::load(
                    svm_model_2_path);
        } catch (cv::Exception &e) {
            std::cerr << "There was an error while loading the svm_model_2\n" << e.what() << std::endl;
        }
    }

    bool isRectInside(const cv::Rect &innerRect, const cv::Rect &outerRect) {
        // Check if the inner rectangle's points lie within the outer rectangle
        return (innerRect.x >= outerRect.x &&
                innerRect.y >= outerRect.y &&
                (innerRect.x + innerRect.width) <= (outerRect.x + outerRect.width) &&
                (innerRect.y + innerRect.height) <= (outerRect.y + outerRect.height));
    }


    void removeRectsInside(std::vector<cv::Rect> &rects) {
        std::vector<cv::Rect>::iterator it = rects.begin();

        while (it != rects.end()) {
            bool isInside = false;

            for (const cv::Rect &otherRect: rects) {
                if (*it != otherRect && isRectInside(*it, otherRect)) {
                    isInside = true;
                    break;
                }
            }

            if (isInside) {
                it = rects.erase(it); // Remove the rectangle if it's inside another
            } else {
                ++it;
            }
        }
    }


public:

    PersonDetector(const std::string &svm_model_1_path, const std::string &svm_model_2_path,
                   HOGDescriptor &hogDescriptor,
                   const double scaleFactor,
                   const std::pair<int, int> &windowSize, const std::pair<int, int> &stepSize,
                   const double detectionThreshold1,
                   const double detectionThreshold2, const float overlapThreshold, const double downscale) :
            hogDescriptor(hogDescriptor),
            scale_factor(scaleFactor),
            window_size(cv::Size(windowSize.first, windowSize.second)),
            stepSize(cv::Size(stepSize.first, stepSize.second)),
            detection_threshold_1(detectionThreshold1),
            detection_threshold_2(detectionThreshold2),
            overlap_threshold(overlapThreshold),
            downscale(downscale) {
        loadModels(svm_model_1_path, svm_model_2_path);
    }


    PersonDetector(const PersonDetector &) = default;

    PersonDetector(PersonDetector &&) = delete;

    PersonDetector &operator=(const PersonDetector &) = delete;

    PersonDetector &operator=(PersonDetector &&) = delete;

    ~PersonDetector() = default;

    /**
     * Performs object detection on the given image using a sliding window approach.
     *
     * @param image The input image for object detection.
     * @return A pair of vectors containing detected regions (Rectangles) and corresponding confidence scores.
     */
    std::pair<std::vector<std::vector<int>>, std::vector<float>> detect(const Eigen::MatrixXd &image) {

        cv::Mat mat_image;
        cv::eigen2cv(image, mat_image);


//        double scale = 0;
        std::vector<cv::Rect> detections;
        std::vector<float> scores;


        cv::Mat fgMask;  // Binary mask to get the foreground

        // Apply background subtraction to get the foreground mask
        pMOG2->apply(mat_image, fgMask);

//        cv::imshow("FG", fgMask);

        Mat thresh, motionMask;

        cv::threshold(fgMask, motionMask, 10, 255, cv::THRESH_BINARY);
//        cv::adaptiveThreshold(fgMask, motionMask, 255, cv::ADAPTIVE_THRESH_MEAN_C,
//                              cv::THRESH_BINARY, 13, 2); // 13
//        cv::GaussianBlur(thresh, motionMask, cv::Size(3, 3), 0);

        cv::imshow("motionMask none", motionMask);

        cv::dilate(motionMask, motionMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)), cv::Size(0,0),3);

        cv::imshow("motionMask dilate", motionMask);
        // Morphological operations to clean up the mask
        morphologyEx(motionMask, motionMask, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        cv::imshow("motionMask opening", motionMask);

        morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));
////
        cv::imshow("motionMask all", motionMask);
        // Find contours in the foreground mask
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
// Find all contours and retrieve the full hierarchy
        findContours(motionMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_L1);


        std::vector<std::vector<cv::Point>> approxContours(contours.size());
        for (size_t i = 0; i < contours.size(); ++i) {
            cv::approxPolyDP(contours[i], approxContours[i], 5, true); // Adjust epsilon value as needed
        }

        cv::Mat contour_image = cv::Mat::zeros(motionMask.size(), CV_8UC3);
        for (size_t i = 0; i < approxContours.size(); ++i) {
            drawContours(contour_image, approxContours, static_cast<int>(i), cv::Scalar(255, 255, 255), 2, cv::LINE_8, hierarchy);
        }

        cv::imshow("Contours", contour_image);



        // Generate an image pyramid for multi-scale detection
//        std::vector<cv::Mat> pyramid = ImagePyramid::generate(mat_image, 8, downscale, window_size);
//
//        // Generate sliding windows for each scaled frame in the pyramid
//        std::vector<std::vector<cv::Rect>> sliding_windows;


//        for (const auto &scaledFrame: pyramid) {
//            std::vector<cv::Rect> scale_windows;
//            for (const auto &window: SlidingWindow::generate(scaledFrame, window_size, stepSize)) {
//                scale_windows.push_back(window);
//            }
//            sliding_windows.push_back(scale_windows);
//        }
//
//        // Perform detection on each window in the sliding windows
//        for (const auto &scale_windows: sliding_windows) {
//            for (const auto &window: scale_windows) {



        std::vector<cv::Rect> contour_detections;


        for (auto &contour: approxContours) {
            cv::Rect rect = cv::boundingRect(contour);

            if (rect.area() > 10000) { //10000

                contour_detections.push_back(rect);
//                int x_orig = (int) (rect.x / scale_factor);
//                int y_orig = (int) (rect.y / scale_factor);
//                int w_orig = (int) (rect.width / scale_factor);
//                int h_orig = (int) (rect.height / scale_factor);
//
//                contour_detections.push_back(cv::Rect(x_orig, y_orig, w_orig, h_orig));
//                scores.push_back(1.0);

            }
        }

//        removeRectsInside(detections);



        for (const cv::Rect &window: contour_detections) {

            int x_orig = (int) (window.x / scale_factor);
            int y_orig = (int) (window.y / scale_factor);
            int w_orig = (int) (window.width / scale_factor);
            int h_orig = (int) (window.height / scale_factor);


            // Extract the region of interest (ROI) from the original image
            cv::Rect potential_bbox(x_orig, y_orig, w_orig, h_orig);


            cv::Mat roi = mat_image(window);

            // Resize the ROI to the desired window size for SVM input
            cv::Mat resized_roi;
            cv::resize(roi, resized_roi, window_size, cv::INTER_AREA);


            // Extract HOG descriptors for the current window
            Eigen::MatrixXd eigen_window;
            cv::cv2eigen(resized_roi, eigen_window);
            Eigen::MatrixXd descriptors = this->hogDescriptor.compute(eigen_window);
            cv::Mat descriptors_mat;
            cv::eigen2cv(descriptors, descriptors_mat);

            // Confirm object presence using confirm method
            std::pair<float, float> res = confirm(descriptors_mat);

            // If confirmed, compute original coordinates, add detection and score
            if (res.first == 1) {
//                    double temp = pow(downscale, scale);


                detections.push_back(potential_bbox);
                float decision = res.second;
                scores.push_back(decision);
            }
        }
//            scale++;



        // Perform non-maximum suppression to filter detections
        std::vector<cv::Rect> picked = NonMaxSuppression::suppress(detections, scores, overlap_threshold);

        for(auto &p: picked){
            cout << "area: " << p.area() << endl;
        }

        std::vector<std::vector<int>> castedRects;

        for (auto &r: picked) {
            std::vector<int> rectValues;
            rectValues.push_back(r.x);
            rectValues.push_back(r.y);
            rectValues.push_back(r.width);
            rectValues.push_back(r.height);
            castedRects.push_back(rectValues);
        }


//        groupRectangles(picked, 1, 0.2);


        return std::make_pair(castedRects, scores);
    }

    /**
     * Confirm the presence of an object in the input vector using SVM models.
     *
     * @param vec Input vector for confirmation.
     * @return A pair containing confirmation status (1.0 for confirmed, else 0.0) and decision score.
     */
    std::pair<float, float> confirm(cv::Mat vec) const {
        // Convert input vector to CV_32F format
        vec.convertTo(vec, CV_32F);

        // Reshape the vector
        cv::Mat vecReshaped = vec.reshape(1, 1);

        // Predict using svm_model_1
        cv::Mat predMat_svm_1, decisionMat_1;
        svm_model_1->predict(vecReshaped, predMat_svm_1);
        svm_model_1->predict(vecReshaped, decisionMat_1, cv::ml::StatModel::RAW_OUTPUT);

        float pred_1 = predMat_svm_1.at<float>(0, 0);
        float decision_1 = decisionMat_1.at<float>(0, 0);



        // Check condition based on prediction from svm_model_1
        if (pred_1 == 1 && (1.0f - abs(decision_1)) > detection_threshold_1) {
            // Predict using svm_model_2
            cv::Mat predMat_svm_2, decisionMat_2;
            svm_model_2->predict(vecReshaped, predMat_svm_2);
            svm_model_2->predict(vecReshaped, decisionMat_2, cv::ml::StatModel::RAW_OUTPUT);

            float pred_2 = predMat_svm_2.at<float>(0, 0);
            float decision_2 = decisionMat_2.at<float>(0, 0);


            // Check condition based on prediction from svm_model_2
            if (pred_2 == 1 && (1.0f - abs(decision_2)) > detection_threshold_2) {

                float decision = ((1.0f - abs(decision_1)) + (1.0f - abs(decision_2))) / 2;

                return std::make_pair(1.0, decision);
            }
        }

        return std::make_pair<float, float>(0.0, 0.0); // Default return if conditions are not met
    }


    const double getScaleFactor() const {
        return scale_factor;
    }

    const cv::Size &getWindowSize() const {
        return window_size;
    }

    const cv::Size &getStepSize() const {
        return stepSize;
    }

    const double getDetectionThreshold1() const {
        return detection_threshold_1;
    }

    const double getDetectionThreshold2() const {
        return detection_threshold_2;
    }

    const float getOverlapThreshold() const {
        return overlap_threshold;
    }

    const double getDownscale() const {
        return downscale;
    }


};


#endif //TRACKING_PERSONDETECTOR_H
