import numpy
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
from src.tracking import HOGDescriptor, PersonDetector


class Track:
    _count = 1

    def __init__(self, bbox, conf , max_inactive_frames=10):
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

        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.velocity = None
        self.delta_t = 3  # Todo place in consturctor parameters

        self.hsv_hist = None

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

    def calculate_velocity(self):
        if len(self.history) >= 2:
            latest_position = np.array(self.history[-1])
            prev_position = np.array(self.history[-2])
            velocity = latest_position - prev_position

            # Apply smoothing using a simple moving average
            alpha = 0.2  # Smoothing factor (adjust as needed)
            smoothed_velocity = alpha * velocity + (
                        1 - alpha) * self.velocity if self.velocity is not None else velocity

            self.velocity = smoothed_velocity

            return smoothed_velocity
        else:
            return np.zeros(4)


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
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det

def normalize_feature(value, observed_data):
    max_value = np.max(observed_data)
    return value / (max_value + 1e-6)


def compute_color_histogram(bbox):
    # Assuming bbox is in the format [x, y, width, height]
    x, y, w, h = bbox
    roi = frame[y:y + h, x:x + w]

    # Convert the region of interest (ROI) to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define histogram parameters
    h_bins = 8
    s_bins = 8
    hist_size = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    channels = [0, 1]

    # Calculate the histogram
    hist = cv2.calcHist([hsv_roi], channels, None, hist_size, [h_ranges, s_ranges])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist)

    return hist.flatten()


def some_similarity_function(hist1, hist2):
    # Compute histogram intersection similarity
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

    return similarity

def compute_appearance_similarity(track, new_bbox):

    new_histogram = compute_color_histogram(new_bbox)
    similarity = some_similarity_function(track.hsv_hist, new_histogram)
    return similarity

def reidentify_track(new_bbox, new_conf, active_tracks):
    for track in active_tracks:
        # Compute similarity between appearance features of the track and the new detection
        appearance_similarity = compute_appearance_similarity(track, new_bbox)

        # Choose a threshold for similarity, e.g., 0.7
        if appearance_similarity > 0.7:
            # Reidentification successful, update the track
            track.update(new_bbox)
            track.conf = new_conf
            return track
    return None



if __name__ == '__main__':
    hogDescriptor = HOGDescriptor(9, (8, 8), (3, 3), 4, False, False, True, True)

    scale_factor = 0.3
    size = (96, 160)

    detection_threshold_1 = 0.4
    detection_threshold_2 = 0.4
    overlap_threshold = 0.3


    personDetector = PersonDetector(
        "/Users/louis/CLionProjects/Tracking/cmake-build-debug/svm_model_inria+tt_96_160_with_cropped_3pK_2nk_tt.xml",
        hogDescriptor, scale_factor,
        size, detection_threshold_1, detection_threshold_2, overlap_threshold)

    cap = cv2.VideoCapture(0)

    activeTracks = []
    removedTracks = []

    max_inactive_frames = 5

    all_velocities = []
    all_accelerations = []

    while True:
        ret, frame = cap.read()
        if ret:
            output = frame.copy()

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_grey = cv2.resize(grey, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            # Update the tracker with detections for each frame
            detections = personDetector.detect(resized_grey, 10000)
            rects = numpy.array(detections[0])  # rect format is [x, y, width, height]
            confidenceScores = numpy.array(detections[1])  # array of float - example [0.87, 1.0, 0.9]

            # Create cost matrix based on IoU between tracks and detections
            cost_matrix = np.zeros((len(activeTracks), len(rects)))
            for i, track in enumerate(activeTracks):
                for j, bbox in enumerate(rects):
                    velocity = track.calculate_velocity()  # Move this line outside the if block
                    prediction = track.predict()

                    iou = calculate_iou(prediction, bbox)

                    # Calculate velocity and acceleration based on the track's history

                    print(f"velocity: {velocity}")
                    if len(track.history) >= 2:
                        prev_position = np.array(track.history[-2])

                        # Check if there is sufficient history to compute acceleration
                        if len(track.history) >= 3:
                            acceleration = velocity - (prev_position - np.array(track.history[-3]))
                        else:
                            # If insufficient history, set acceleration to zero
                            acceleration = np.zeros_like(velocity)

                        # You may choose to use velocity or acceleration (or both) in your association logic
                        # Here, I'm using both velocity and acceleration to form the cost matrix
                        # You can modify this logic based on your requirements
                        # Normalize velocity and acceleration
                        norm_velocity = np.linalg.norm(velocity)
                        norm_acceleration = np.linalg.norm(acceleration)

                        all_velocities.append(norm_velocity)
                        all_accelerations.append(norm_acceleration)



                        norm_velocity = normalize_feature(norm_velocity, all_velocities)
                        norm_acceleration = normalize_feature(norm_acceleration, all_accelerations)
                        # print(-iou - norm_velocity - norm_acceleration)

                        # Combine IOU, normalized velocity, and normalized acceleration to form the cost for Hungarian algorithm
                        # You can adjust weights or combine them differently based on their importance
                        cost_matrix[i, j] = -iou - norm_velocity - norm_acceleration
                    else:
                        # If there's insufficient history, only consider IOU in the cost matrix
                        cost_matrix[i, j] = -iou

        # Use Hungarian algorithm to find the optimal assignment between tracks and detections
        print(cost_matrix)
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Update tracks based on the assignment
        for i, j in zip(track_indices, detection_indices):
            prediction = activeTracks[i].predict()  # Move prediction outside the inner loop
            # Todo make assignment as parameter
            if cost_matrix[i, j] < -0.5:  # Threshold for considering the assignment
                new_bbox = rects[j]
                activeTracks[i].update(new_bbox)  # Update the track with the associated detection
                activeTracks[i].conf = confidenceScores[j]  # Update track confidence

        # Create new tracks for unmatched detections
        unmatched_detections = set(range(len(rects))) - set(detection_indices)
        for idx in unmatched_detections:
            new_track = Track(rects[idx], confidenceScores[idx], max_inactive_frames=max_inactive_frames)
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
