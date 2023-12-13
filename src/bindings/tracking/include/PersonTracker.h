//
// Created by Louis-Kaan Ay on 12.12.23.
//

#ifndef TRACKING_PERSONTRACKER_H
#define TRACKING_PERSONTRACKER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <map>

#include "../include/HOGDescriptor.h"
#include "../include/PersonDetector.h"

class PersonTracker {
private:
    PersonDetector personDetector;
    int nextObjectID;
    std::map<int, cv::KalmanFilter> objectKalmanFilters;
    std::map<int, cv::Rect> objectRects;
    int frameCount;
    double totalTime;


public:
    PersonTracker(PersonDetector &personDetector) : personDetector(personDetector), nextObjectID(0), frameCount(0),
                                                    totalTime(0.0) {
    }

    std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<float>, double> track(const Eigen::MatrixXd &frame) {

        std::pair<std::vector<cv::Rect>, std::vector<float>> res = personDetector.detect(frame);
        std::vector<cv::Rect> picked = res.first;
        std::vector<float> scores = res.second;
        std::vector<int> ids;
        std::vector<cv::Rect> kalmanRects;

        // Update Kalman filters with new detections
        for (size_t i = 0; i < picked.size(); ++i) {
            const cv::Rect &rect = picked[i];
            const double &score = scores[i]; // Retrieve the score corresponding to the rectangle

            bool foundMatch = false;
            int objectID = -1;

            // Check if the detected rect overlaps with existing objects
            for (const auto &obj: objectRects) {
                cv::Rect overlap = rect & obj.second;
                if (overlap.area() > 0) {
                    foundMatch = true;
                    objectID = obj.first;

                    cv::KalmanFilter &kf = objectKalmanFilters[obj.first];
                    cv::Mat measurement = (cv::Mat_<float>(2, 1) << rect.x + rect.width / 2, rect.y + rect.height / 2);
                    cv::Mat prediction = kf.predict();
                    cv::Mat estimated = kf.correct(measurement);

                    cv::Rect estimatedRect(estimated.at<float>(0) - rect.width / 2,
                                           estimated.at<float>(1) - rect.height / 2,
                                           rect.width, rect.height);

                    objectRects[obj.first] = estimatedRect;

                    kalmanRects.push_back(estimatedRect);
                    ids.push_back(obj.first);

                    break;
                }
            }

            if (!foundMatch) {
                // Initialize Kalman filter for new object
                cv::KalmanFilter kf(4, 2);
                kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
                kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
                kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.1;
                kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 0.01;
                kf.statePost.at<float>(0) = rect.x + rect.width / 2;
                kf.statePost.at<float>(1) = rect.y + rect.height / 2;
                kf.statePost.at<float>(2) = 0;
                kf.statePost.at<float>(3) = 0;

                objectKalmanFilters[nextObjectID] = kf;
                objectRects[nextObjectID] = rect;

                kalmanRects.push_back(rect);
                ids.push_back(nextObjectID);
                nextObjectID++;
            }
        }

        // Calculate FPS
        frameCount++;
        double current_time = static_cast<double>(cv::getTickCount());
        double elapsedTime = (current_time - totalTime) / cv::getTickFrequency();
        totalTime = current_time;
        double fps = 1.0 / elapsedTime;

        // Pybind11 will throw an error if we return std::vector<cv::Rects> so we convert it std::vector<std::vector<int>> here
        std::vector<std::vector<int>> castedRects;

        for (auto &r: kalmanRects) {
            std::vector<int> rectValues;
            rectValues.push_back(r.x);
            rectValues.push_back(r.y);
            rectValues.push_back(r.width);
            rectValues.push_back(r.height);
            castedRects.push_back(rectValues);

        }

        return std::make_tuple(castedRects, ids, scores, fps);
    }
};


#endif //TRACKING_PERSONTRACKER_H
