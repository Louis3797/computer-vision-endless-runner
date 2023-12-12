from typing import Generator, Any

import cv2
import numpy as np
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from hog import HOG
import joblib
from skimage import color

model = joblib.load(
    '/Users/louis/PycharmProjects/cv_project/src/cv/svm/svm_detection_inria_with_flipped_and_anno_96_160_p_tt_500.dat')
size = (96, 160)
step_size = (10, 10)
downscale = 1.5
scale_factor = 0.2  # Define the scale factor
detection_threshold = 0.8


def sliding_window(image: np.ndarray, window_size: tuple[int, int], step_size: tuple[int, int]) -> Generator[
    tuple[int, int, np.ndarray], Any, None]:
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield x, y, image[y: y + window_size[1], x: x + window_size[0]]


def computeHOGFeatures(image: np.ndarray) -> np.ndarray:
    fd, _ = HOG().compute(image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(3, 3), sobel=False, visualize=False,
                          normalize_input=True, flatten=True)

    return fd




cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Gamma correction
    # gamma = 1.5
    # frame_corrected = np.power(frame / 255.0, gamma) * 255.0
    # frame_corrected = np.clip(frame_corrected, 0, 255).astype(np.uint8)

    # Gaussian smoothing
    frame_smoothed = cv2.GaussianBlur(frame, (5, 5), 0)

    cv2.imshow("gamme", frame_smoothed)

    image = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    grey_image = color.rgb2gray(image)

    detections = []
    # The current scale of the image
    scale = 0
    for im_scaled in pyramid_gaussian(grey_image, downscale=downscale):
        # The list contains detections at the current scale
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break
        for (x, y, window) in sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue

            fd = computeHOGFeatures(window).reshape(1, -1)

            pred = model.predict(fd)
            print(f"pred: {pred}")

            if pred == 1:

                decision = model.decision_function(fd)
                print(decision)
                # print(decision)
                if decision > detection_threshold:
                    x_orig = int(x * (downscale ** scale) / scale_factor)
                    y_orig = int(y * (downscale ** scale) / scale_factor)
                    w_orig = int(size[0] * (downscale ** scale) / scale_factor)
                    h_orig = int(size[1] * (downscale ** scale) / scale_factor)

                    detections.append((x_orig, y_orig, decision, w_orig, h_orig))
        scale += 1
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = np.array([score[0] for (x, y, score, w, h) in detections])
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

    for (x1, y1, x2, y2) in pick:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, 'Person : {:.2f}'.format(np.max(sc)), (x1 - 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 2)

    cv2.imshow('Person Detection', frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
