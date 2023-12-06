import cv2
import numpy as np
import time
import random


def calculate_intersection(box1, box2):
    x_overlap = max(0, min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1]))
    return x_overlap * y_overlap


class BoundingBoxProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error opening camera")
            exit(-1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        cv2.namedWindow("Split Frame", cv2.WINDOW_NORMAL)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def process_frames(self, x1, y1, x2, y2):
        while True:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                print("No frame captured from camera")
                break

            time.sleep(1)
            random_x = random.randint(0, self.width - (self.width // 3))
            bounding_box = (random_x, self.height // 4, self.width // 3, self.height // 2)

            section1 = (0, 0, self.width // 3, self.height)
            section2 = (self.width // 3, 0, self.width // 3, self.height)
            section3 = ((self.width // 3) * 2, 0, self.width // 3, self.height)

            intersection1 = calculate_intersection(section1, bounding_box)
            intersection2 = calculate_intersection(section2, bounding_box)
            intersection3 = calculate_intersection(section3, bounding_box)

            union1 = (section1[2] * section1[3]) + (bounding_box[2] * bounding_box[3]) - intersection1
            union2 = (section2[2] * section2[3]) + (bounding_box[2] * bounding_box[3]) - intersection2
            union3 = (section3[2] * section3[3]) + (bounding_box[2] * bounding_box[3]) - intersection3

            iou1 = intersection1 / union1
            iou2 = intersection2 / union2
            iou3 = intersection3 / union3

            iou_values = [iou1, iou2, iou3]
            section = iou_values.index(max(iou_values)) + 1

            section_text = f"Section: {section}"
            cv2.putText(frame, section_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            rectangles = [section1, section2, section3, bounding_box]
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

            for rect, color in zip(rectangles, colors):
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, 2)

            cv2.imshow("Split Frame", frame)

            if cv2.waitKey(1) == 27:
                break


if __name__ == "__main__":
    processor = BoundingBoxProcessor()
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    processor.process_frames(x1, y1, x2, y2)
