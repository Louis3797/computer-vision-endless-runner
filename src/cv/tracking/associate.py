import numpy as np
from scipy.optimize import linear_sum_assignment

from src.cv.tracking.Metric import Metric
from src.cv.utils.crop import crop_bboxes_from_frame


def linear_assignment(cost_matrix):
    """
    Perform linear assignment using the Hungarian algorithm.

    Args:
        cost_matrix: The cost matrix to be assigned.

    Returns:
        Array of tuples representing the assigned indices.
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def associate_detections_to_trackers(frame, detections, tracks, track_orb_features, iou_threshold=0.3,
                                     iou_metric_weight=1.0,
                                     orb_metric_weight=1.0,
                                     ):
    """
    Associate detections to existing trackers using IoU and ORB metrics.

    Args:
        frame: Input image frame.
        detections: List of bounding box coordinates for detections.
        tracks: List of bounding box coordinates for existing tracks.
        track_orb_features: List of ORB features for existing tracks.
        iou_threshold: IoU threshold for matching.
        iou_metric_weight: Weight for IoU metric.
        orb_metric_weight: Weight for ORB metric.

    Returns:
        Tuple of matched indices, unmatched detections, and unmatched trackers.
    """
    # Check if there are no trackers, return empty results
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # Initialize iou cat matrix
    cost_matrix = np.zeros(shape=(len(detections), len(tracks)), dtype=np.float32)

    iou_matrix = Metric.calculate_iou_metric(detections, tracks)

    detections_crops = crop_bboxes_from_frame(frame, detections)

    orb_matrix = Metric.calculate_orb_metric(detections_crops, track_orb_features)

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
                orb_score = Metric.calculate_orb_score(detections_crops[detection_index],
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