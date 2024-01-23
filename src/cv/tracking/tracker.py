import numpy as np
import cv2
from src.cv.tracking.Metric import Metric
from src.cv.tracking.associate import associate_detections_to_trackers
from src.cv.tracking.track import Track
from src.bindings.tracking.build.tracking import HOGDescriptor, PersonDetector # ignore this error
from src.cv.utils.color import generate_bright_color
from src.cv.utils.crop import crop_bbox


class Tracker(object):

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, iou_metric_weight=1.0, orb_metric_weight=1.0):
        """
           Initialize the object tracker.

           Args:
               max_age: Maximum number of frames without an update before a track is considered dead.
               min_hits: Minimum number of hits to confirm a track.
               iou_threshold: IoU threshold for matching.
               iou_metric_weight: Weight for IoU metric.
               orb_metric_weight: Weight for ORB metric.
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
          Update the tracker with new frame information.

          Args:
              frame: Input image frame.
              new_detections: List of bounding box coordinates for new detections.
              confidenceScore: List of confidence scores for new detections.

          Returns:
              Updated list of tracks.
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
            orb_kp, orb_des = Metric.extract_orb_features(image_crop)
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

    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)

    hogDescriptor = HOGDescriptor(9, (8, 8), (3, 3), 4, False, False, True, True)

    scale_factor = 0.3
    size = (96, 160)
    detection_threshold = 0.3
    overlap_threshold = 0.3
    bgs_history = 500
    bgs_threshold = 15
    bgs_detectShadows = False
    bgs_learning_rate = 0.01
    bgs_shadow_threshold = 0.5

    personDetector = PersonDetector(
        "../svm_models/svm_model_inria+neg_tt+daimler_16Kpos_15Kneg_no_flipped.xml",

        hogDescriptor, scale_factor,
        size, detection_threshold, overlap_threshold, bgs_history, bgs_threshold, bgs_detectShadows,
        bgs_learning_rate, bgs_shadow_threshold)

    # cap = cv2.VideoCapture("/Users/louis/Downloads/dance_demo.mp4")
    cap = cv2.VideoCapture("/Users/louis/Downloads/PETS09-S2L1-raw.mp4")
    # cap = cv2.VideoCapture("/Users/louis/Downloads/tracking_vid_test1.mp4")
    # cap = cv2.VideoCapture("/Users/louis/Downloads/y2mate.is - Sidekick Dance Challenge 2020 Mastermind-tZPbPDEIaeE-1080pp-1705850872.mp4")
    # cap = cv2.VideoCapture("/Users/louis/Documents/Uni/Semester 5/Computer Vision/m3_crowded_bg_with_jan_in_video.mov")
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
            detections = personDetector.detect(resized_grey, 4000, 3,7,9,9)
            rects = np.array(detections[0])  # rect format is [x, y, width, height]
            confidenceScores = np.array(detections[1])  # array of float - example [0.87, 1.0, 0.9]


            # # if there are any detections
            if len(rects) > 0:
                rects[:, 2:4] += rects[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]


                tracks = tracker.update(frame, rects, confidenceScores)


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
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (track_color_r, track_color_b, track_color_g), 2)

            cv2.imshow("output", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
