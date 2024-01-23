import cv2
import numpy as np


def crop_bbox(frame, bbox):
    """
    Crop a bounding box from an image frame.

    Args:
        frame: Input image frame.
        bbox: Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        Cropped image.
    """
    x1, y1, x2, y2 = bbox.astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        # Handle empty crop
        # Note: This is not a perfect solution, but it works for now
        crop = np.zeros((160, 96, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (96, 160), interpolation=cv2.INTER_AREA)

    return crop


def crop_bboxes_from_frame(frame, bboxes):
    """
    Crop multiple bounding boxes from an image frame.

    Args:
        frame: Input image frame.
        bboxes: List of bounding box coordinates.

    Returns:
        List of cropped images.
    """
    return [crop_bbox(frame, bbox) for bbox in bboxes]
