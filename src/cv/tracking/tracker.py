import os

import numpy as np
import cv2
import ultralytics
from ultralytics import YOLO

from src.cv.tracking.track import Track
from src.bindings.tracking.build.tracking import HOGDescriptor, PersonDetector

import time
import colorsys
from scipy.optimize import linear_sum_assignment

from src.cv.utils.iou import calculate_iou

np.random.seed(0)


def generate_bright_color():
    """
    Generate a random bright color in RGB format.

    Returns:
        Tuple of three integers representing bright RGB values.
    """
    # Generate a random hue in [0, 1) to cover the entire color spectrum
    hue = np.random.rand()

    # Set saturation and lightness to high values for brightness
    saturation = 0.8
    lightness = 0.7

    # Convert HSL to RGB
    rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Scale the values to the range [0, 255] and round to integers
    return tuple(int(val * 255) for val in rgb_color)


def linear_assignment(cost_matrix):
    """
    Wrapper function around scipy.optimize.linear_sum_assignment
    :param cost_matrix: The cost matrix for the linear assignment
    :return: An array of row indices and one of corresponding column indices giving the optimal assignment. The cost of the assignment can be computed as cost_matrix[row_ind, col_ind].sum(). The row indices will be sorted; in the case of a square cost matrix they will be equal to numpy.arange(cost_matrix.shape[0])
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def speed_direction(bbox1, bbox2):
    """
    Calculate speed and direction vector between two bounding boxes.

    Parameters:
    - bbox1 (list or array): [x1, y1, x2, y2] representing the first bounding box.
    - bbox2 (list or array): [x1, y1, x2, y2] representing the second bounding box.

    Returns:
    - np.array: Speed and direction vector [dx, dy].
    """
    center_x1, center_y1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    center_x2, center_y2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0

    displacement = np.array([center_x2 - center_x1, center_y2 - center_y1])
    norm = np.linalg.norm(displacement) + 1e-6

    return displacement / norm


def crop_bbox(frame, bbox):
    x1, y1, x2, y2 = bbox.astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        # Handle empty crop (you can log an error, skip, or take appropriate action)
        crop = np.zeros((160, 96, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (96, 160), interpolation=cv2.INTER_AREA)

    return crop


def crop_bbox_from_frame(frame, bboxes):
    return [crop_bbox(frame, bbox) for bbox in bboxes]


def extract_orb_features(image):
    # Initialize ORB feature detector with recommended parameters (may need tweaks according to use case)
    patchSize = 7
    orb = cv2.ORB_create(edgeThreshold=7, patchSize=patchSize, nlevels=8, scaleFactor=1.2, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500, fastThreshold=20)

    # make sure the image shape is larger than the patchSize in ORB parameters
    if min(image.shape[:2]) < patchSize:
        return None, None

    # find the keypoints and descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if isinstance(descriptors, type(None)):
        return None, None

    return keypoints, descriptors


def calculate_orb_score(img1, track_orb_features):
    # Extract ORB features for both images
    kp1, des1 = extract_orb_features(img1)
    kp2, des2 = track_orb_features

    if kp1 is None or kp2 is None or des1 is None or des2 is None:
        return 0

    # consider the appropriate reference for keypoint matches
    num_kp = min(len(kp1), len(kp2))

    # create BFMatcher object and match descriptors
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # ratio test as per Lowe's paper
    good_matches = []
    for match in matches:
        if len(match) != 2:
            continue
        m, n = match
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    score = len(good_matches) / num_kp

    return score


def calculate_iou_metric(detections, tracks):
    iou_matrix = np.zeros(shape=(len(detections), len(tracks)), dtype=np.float32)

    # Calculate costs for iou matrix
    for d, det in enumerate(detections):
        for t, trk in enumerate(tracks):
            iou_matrix[d, t] = calculate_iou(det, trk)

    return iou_matrix


def calculate_orb_metric(detections, track_orb_features):
    orb_matrix = np.zeros(shape=(len(detections), len(track_orb_features)), dtype=np.float32)

    # Calculate costs for iou matrix
    for d, det in enumerate(detections):
        for t, trk in enumerate(track_orb_features):
            orb_matrix[d, t] = calculate_orb_score(det, trk)

    return orb_matrix


def associate_detections_to_trackers(frame, detections, tracks, track_orb_features, iou_threshold=0.3,
                                     iou_metric_weight=1.0,
                                     orb_metric_weight=1.0,
                                     ):
    """
    Assigns detections to tracked objects (both represented as bounding boxes)

    Returns three lists: matches, unmatched_detections, and unmatched_trackers
    """
    # Check if there are no trackers, return empty results
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # Initialize iou cat matrix
    cost_matrix = np.zeros(shape=(len(detections), len(tracks)), dtype=np.float32)

    iou_matrix = calculate_iou_metric(detections, tracks)

    detections_crops = crop_bbox_from_frame(frame, detections)

    orb_matrix = calculate_orb_metric(detections_crops, track_orb_features)

    cost_matrix = -iou_metric_weight * iou_matrix - orb_metric_weight * orb_matrix



    # Find matched indices using IoU matrix and threshold
    matched_indices = linear_assignment(cost_matrix)
    # Find indices of unmatched detections and trackers
    unmatched_detections = [i for i, obj in enumerate(detections) if i not in matched_indices[:, 0]]
    unmatched_trackers = [i for i, obj in enumerate(tracks) if i not in matched_indices[:, 1]]

    # cost_values = np.array([cost_matrix[row_id, col_id] for row_id, col_id in matched_indices])
    # print(cost_values)
    # matches = matched_indices[cost_values > iou_threshold]
    #
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    for track_index in unmatched_trackers:
        orb_features = track_orb_features[track_index]
        if orb_features[0] is not None and orb_features[1] is not None:
            # Check if the track was previously associated with a detection
            # If so, try to find a matching detection in the current frame
            for detection_index in unmatched_detections:
                orb_score = calculate_orb_score(detections_crops[detection_index],
                                                orb_features)
                if orb_score > 0.3:  # Adjust this threshold as needed
                    matched_indices = np.append(matched_indices, [[detection_index, track_index]], axis=0)
                    unmatched_detections.remove(detection_index)
                    break

    if len(matched_indices) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matched_indices)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Tracker(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, iou_metric_weight=1.0, orb_metric_weight=1.0):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.active_tracks: list[Track] = []
        self.iou_metric_weight = iou_metric_weight
        self.orb_metric_weight = orb_metric_weight
        self.frame_count = 0

    def update(self, frame, new_detections, confidenceScore):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers and remove all tracks with nan values after prediction
        self.active_tracks = [t for t in self.active_tracks if not np.any(np.isnan(t.predict()[0]))]

        # Create track np.array with all current tracks
        current_tracks = np.array([t.get_state()[0] for t in self.active_tracks])

        track_orb_features = [[t.orb_keypoints, t.orb_descriptors] for t in self.active_tracks]

        matched, unmatched_detections, unmatched_tracks = associate_detections_to_trackers(frame, new_detections,
                                                                                           current_tracks,
                                                                                           track_orb_features,
                                                                                           self.iou_threshold,
                                                                                           iou_metric_weight=self.iou_metric_weight,
                                                                                           orb_metric_weight=self.orb_metric_weight)
        # update matched trackers with assigned detections
        for detection_num, tracker_num in matched:
            self.active_tracks[tracker_num].update(new_detections[detection_num, :], confidenceScore[detection_num])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_detections:
            new_bbox = new_detections[i, :]
            image_crop = crop_bbox(frame, new_bbox)
            orb_kp, orb_des = extract_orb_features(image_crop)
            new_track = Track(new_bbox, confidenceScore[i], generate_bright_color(), orb_keypoints=orb_kp,
                              orb_descriptors=orb_des)
            self.active_tracks.append(new_track)

        # Remove dead tracks
        self.active_tracks = [track for track in self.active_tracks if track.time_since_update <= self.max_age]

        output_tracks = np.array([np.concatenate((t.get_state()[0], [t.id], [t.conf], t.color))
                                  for t in self.active_tracks
                                  if (t.time_since_update > self.max_age) or (
                                          t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)])

        if len(output_tracks) > 0:
            return output_tracks

        # return empty array if out is empty
        return np.empty((0, 9))


if __name__ == '__main__':

    cv2.setUseOptimized(True);
    cv2.setNumThreads(8);

    hogDescriptor = HOGDescriptor(9, (8, 8), (3, 3), 4, False, False, True, True)

    scale_factor = 0.3
    size = (96, 160)
    detection_threshold = 0.3
    overlap_threshold = 0.3
    bgs_history = 500
    bgs_threshold = 10
    bgs_detectShadows = False
    bgs_learning_rate = 0.1
    bgs_shadow_threshold = 0.5

    personDetector = PersonDetector(
         "/Users/louis/CLionProjects/Tracking/models/svm_model_inria+neg_tt+daimler_16Kpos_15Kneg_no_flipped.xml",

        hogDescriptor, scale_factor,
        size, detection_threshold, overlap_threshold, bgs_history, bgs_threshold, bgs_detectShadows,
        bgs_learning_rate, bgs_shadow_threshold)

    model = YOLO('yolov8n.pt')

    # cap = cv2.VideoCapture("/Users/louis/Downloads/dance_demo.mp4")
    # cap = cv2.VideoCapture("/Users/louis/Downloads/tracking_vid_test1.mp4")
    # cap = cv2.VideoCapture("/Users/louis/Downloads/y2mate.is - Sidekick Dance Challenge 2020 Mastermind-tZPbPDEIaeE-1080pp-1705850872.mp4")
    cap = cv2.VideoCapture("/Users/louis/Documents/Uni/Semester 5/Computer Vision/m3_white_bg.mov")
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("/Users/louis/Documents/Uni/Semester 5/Computer Vision/MOT20-02-raw.mp4")

    tracker = Tracker(max_age=50,
                      min_hits=3,
                      iou_threshold=0.3, iou_metric_weight=2.0,
                      orb_metric_weight=1.0)

    total_time = 0.0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if ret:
            output = frame.copy()
            np.flip(output, 1)
            frame += 1  # detection and frame numbers begin at 1

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_grey = cv2.resize(grey, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            # # Update the tracker with detections for each frame
            detections = personDetector.detect(resized_grey, 4000)
            rects = np.array(detections[0])  # rect format is [x, y, width, height]
            confidenceScores = np.array(detections[1])  # array of float - example [0.87, 1.0, 0.9]
            # #
            # rects = []
            # confidenceScores = []
            # results: ultralytics.engine.results.Results = model(frame)
            #
            # for i in range(results[0].boxes.xyxy.shape[0]):
            #     cls = results[0].boxes.cls.tolist()
            #     conf = results[0].boxes.conf.tolist()
            #     b = results[0].boxes.xyxy.tolist()
            #
            #     if cls[i] == 0 and conf[i] > 0.3:
            #         rects.append(b[i])
            #         confidenceScores.append(conf[i])
            #
            # rects = np.array(rects)
            # rects = rects.astype(int)
            # # if there are any detections
            if len(rects) > 0:
                rects[:, 2:4] += rects[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

                start_time = time.time()

                tracks = tracker.update(frame, rects, confidenceScores)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for t in tracks:
                    bbox = t[:4].astype(int)
                    track_id = int(t[4])
                    track_conf = t[5]
                    track_color_r = t[6]
                    track_color_g = t[7]
                    track_color_b = t[8]

                    cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (track_color_r, track_color_b, track_color_g), 3)
                    cv2.putText(output, f"id: {track_id} conf: {track_conf:0.2f}", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (track_color_r, track_color_b, track_color_g), 2)

            cv2.imshow("output", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
