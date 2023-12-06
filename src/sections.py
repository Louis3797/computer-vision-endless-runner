import cv2
import numpy as np
import time
import random

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

    def process_frames(self):
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

            # Calculate intersection areas
            intersection1 = (min(section1[0] + section1[2], bounding_box[0] + bounding_box[2]) - max(section1[0], bounding_box[0])) * \
                            (min(section1[1] + section1[3], bounding_box[1] + bounding_box[3]) - max(section1[1], bounding_box[1]))

            intersection2 = (min(section2[0] + section2[2], bounding_box[0] + bounding_box[2]) - max(section2[0], bounding_box[0])) * \
                            (min(section2[1] + section2[3], bounding_box[1] + bounding_box[3]) - max(section2[1], bounding_box[1]))

            intersection3 = (min(section3[0] + section3[2], bounding_box[0] + bounding_box[2]) - max(section3[0], bounding_box[0])) * \
                            (min(section3[1] + section3[3], bounding_box[1] + bounding_box[3]) - max(section3[1], bounding_box[1]))

            # Calculate union areas
            union1 = (section1[2] * section1[3]) + (bounding_box[2] * bounding_box[3]) - intersection1
            union2 = (section2[2] * section2[3]) + (bounding_box[2] * bounding_box[3]) - intersection2
            union3 = (section3[2] * section3[3]) + (bounding_box[2] * bounding_box[3]) - intersection3

            # Calculate IoU values
            iou1 = intersection1 / union1
            iou2 = intersection2 / union2
            iou3 = intersection3 / union3

            # Determine which section the bounding box belongs to based on the highest IoU
            iou_values = [iou1, iou2, iou3]
            section = iou_values.index(max(iou_values)) + 1

            # Draw the text indicating the section
            section_text = f"Section: {section}"
            cv2.putText(frame, section_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw the rectangles for sections and bounding box
            cv2.rectangle(frame, (section1[0], section1[1]), (section1[0] + section1[2], section1[1] + section1[3]), (255, 0, 0), 2)  # Blue
            cv2.rectangle(frame, (section2[0], section2[1]), (section2[0] + section2[2], section2[1] + section2[3]), (0, 255, 0), 2)  # Green
            cv2.rectangle(frame, (section3[0], section3[1]), (section3[0] + section3[2], section3[1] + section3[3]), (0, 0, 255), 2)  # Red
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 255, 0), 2)  # Yellow

            cv2.imshow("Split Frame", frame)

            if cv2.waitKey(1) == 27:
                break

if __name__ == "__main__":
    processor = BoundingBoxProcessor()
    processor.process_frames()
