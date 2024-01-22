import cv2
import numpy as np

from enum import Enum


class Sections(Enum):
    LEFT = 1
    MID = 2
    RIGHT = 3


def get_section_with_most_boxes(frame, bounding_boxes) -> Sections:
    height, width, _ = frame.shape

    # Calculate section boundaries
    section_width = width // 3
    section1 = section_width
    section2 = 2 * section_width

    # Count bounding boxes in each section
    count_section1 = 0
    count_section2 = 0
    count_section3 = 0

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        box_center = (x1 + x2) // 2

        if box_center < section1:
            count_section1 += 1
        elif section1 <= box_center < section2:
            count_section2 += 1
        else:
            count_section3 += 1

    # Determine section with the most bounding boxes
    max_count = max(count_section1, count_section2, count_section3)

    if max_count == count_section1:
        return Sections.LEFT
    elif max_count == count_section2:
        return Sections.MID
    else:
        return Sections.RIGHT

