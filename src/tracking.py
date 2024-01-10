import numpy
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
from src.tracking import HOGDescriptor, PersonDetector


class Track:
    _count = 1

    def __init__(self, bbox, conf, max_inactive_frames=10):
        # Give auto id by with help pf static class attribute
        self.id = Track._count
        Track._count += 1

        self.bbox = bbox  # bounding box: [x, y, width, height]
        self.conf = conf  # confidence score
        self.age = 1  # age of the track
        self.isActive = True  #
        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.max_inactive_frames = max_inactive_frames

        self.last_observation = np.array([-1,-1,-1,-1,-1])
        self.velocity = None
        self.delta_t = 3  # Todo place in consturctor parameters

        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

        # Measurement function matrix: x, y, width, height
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        # Covariance matrix initialization
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # Measurement noise matrix
        self.kf.R[2:, 2:] *= 10.

        # Process noise matrix
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state with bounding box coordinates (x, y, width, height)
        self.kf.x[:4] = bbox.reshape((4, 1))

    def update(self, bbox):

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Update Kalman filter with the new measurement (x, y, width, height)
        self.kf.update(bbox.reshape((4, 1)))
        # Update bounding box with predicted values (x, y, width, height)
        self.bbox = [int(round(float(i))) for i in self.kf.x[:4]]

    def predict(self):

        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        if self.time_since_update >= self.max_inactive_frames:
            self.isActive = False

        # Predict next bounding box using Kalman filter

        predicted_bbox = [int(round(float(i))) for i in self.kf.x[:4]]  # Convert elements to float before rounding
        self.history.append(predicted_bbox)

        return self.history[-1]




# =======================================================================================

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


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


if __name__ == '__main__':
    hogDescriptor = HOGDescriptor(9, (8, 8), (3, 3), 4, False, False, True, True)

    scale_factor = 0.3
    size = (96, 160)
    stepSize = (10, 10)
    detection_threshold_1 = 0.3
    detection_threshold_2 = 0.3
    overlap_threshold = 0.6
    downscale = 1.15

    personDetector = PersonDetector(
        "/Users/louis/CLionProjects/Tracking/models/svm_model_inria_96_160_with_flipped.xml",
        "/Users/louis/CLionProjects/Tracking/models/svm_model_tt_96_160_with_cropped_5000.xml",
        hogDescriptor, scale_factor,
        size, stepSize, detection_threshold_1, detection_threshold_2, overlap_threshold,
        downscale)

    cap = cv2.VideoCapture(0)

    activeTracks = []
    removedTracks = []
    max_inactive_frames = 5


    while True:
        ret, frame = cap.read()
        if ret:
            output = frame.copy()

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_grey = cv2.resize(grey, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            # Update the tracker with detections for each frame
            detections = personDetector.detect(resized_grey)
            rects = numpy.array(detections[0])  # rect format is [x, y, width, height]
            confidenceScores = numpy.array(detections[1])  # array of float - example [0.87, 1.0, 0.9]

            # Create cost matrix based on IoU between tracks and detections
            cost_matrix = np.zeros((len(activeTracks), len(rects)))
            for i, track in enumerate(activeTracks):
                for j, bbox in enumerate(rects):
                    prediction = track.predict()
                    iou = calculate_iou(prediction, bbox)
                    cost_matrix[i, j] = -iou  # Negative IoU as the cost for Hungarian algorithm

            # Use Hungarian algorithm to find the optimal assignment between tracks and detections
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)

            # Update tracks based on the assignment
            for i, j in zip(track_indices, detection_indices):
                prediction = activeTracks[i].predict()  # Move prediction outside the inner loop
                if cost_matrix[i, j] < -0.5:  # Threshold for considering the assignment
                    new_bbox = rects[j]
                    activeTracks[i].update(new_bbox)  # Update the track with the associated detection
                    activeTracks[i].conf = confidenceScores[j]  # Update track confidence

            # Create new tracks for unmatched detections
            unmatched_detections = set(range(len(rects))) - set(detection_indices)
            for idx in unmatched_detections:
                new_track = Track(rects[idx], confidenceScores[idx])
                activeTracks.append(new_track)

            # Remove inactive tracks
            activeTracks = [track for track in activeTracks if track.isActive]
            removedTracks.extend([track for track in activeTracks if not track.isActive])

            # Display Tracks
            for track in activeTracks:
                x, y, w, h = track.bbox
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, f"ID: {track.id}\t conf: {track.conf:0.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 2)

            cv2.imshow("output", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
