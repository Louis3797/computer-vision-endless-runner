
from filterpy.kalman import KalmanFilter
from src.cv.tracking.bbox_utils import xysr_to_xxyy, xxyy_to_xysr
import numpy as np


class Track(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, conf, color,  orb_keypoints=None, orb_descriptors=None):
        """
        Initialises a tracker using initial bounding box.
        """

        self.id = Track.count
        Track.count += 1

        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.

        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = xxyy_to_xysr(bbox)
        self.time_since_update = 0

        self.conf = conf
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.velocity = np.array([0, 0])
        self.color = color
        self.orb_keypoints = orb_keypoints
        self.orb_descriptors = orb_descriptors

    def update(self, bbox, conf, orb_keypoints=None, orb_descriptors=None):
        """
        Updates the state vector with observed bbox.
        """
        self.kf.update(xxyy_to_xysr(bbox))
        self.orb_keypoints = orb_keypoints
        self.orb_descriptors = orb_descriptors
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.conf = conf

        current_position = xxyy_to_xysr(bbox)[:2]
        if len(self.history) > 0:
            previous_position = self.history[-1][:2]
            self.velocity = current_position - previous_position

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(xysr_to_xxyy(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return xysr_to_xxyy(self.kf.x)