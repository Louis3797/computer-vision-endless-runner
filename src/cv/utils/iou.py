import numpy as np


def calculate_iou(boxA, boxB):
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = np.minimum(boxA[1] + boxA[3], boxB[1] + boxB[3])

    intersection_area = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    boxA_area = (boxA[2] + 1) * (boxA[3] + 1)
    boxB_area = (boxB[2] + 1) * (boxB[3] + 1)

    iou = intersection_area / (boxA_area + boxB_area - intersection_area)

    return iou