import cv2
import numpy as np

from src.cv.utils.iou import calculate_iou


class Metric:

    def __init__(self):
        pass

    @staticmethod
    def extract_orb_features(image):
        """
          Extract ORB features from an image.

          Args:
              image: Input image.

          Returns:
              Tuple of keypoints and descriptors.
          """
        # Initialize ORB feature detector with recommended parameters (may need tweaks according to use case)
        patchSize = 7
        orb = cv2.ORB_create(edgeThreshold=7, patchSize=patchSize, nlevels=8, scaleFactor=1.2, WTA_K=2,
                             scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500, fastThreshold=20)

        # make sure the image shape is larger than the patchSize in ORB parameters
        if min(image.shape[:2]) < patchSize:
            return None, None

        # find the keypoints and descriptors with ORB
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if isinstance(descriptors, type(None)):
            return None, None

        return keypoints, descriptors

    @staticmethod
    def calculate_orb_score(img1, track_orb_features):
        """
        Calculate the ORB score between two images.

        Args:
            img1: First image.
            track_orb_features: Tuple of keypoints and descriptors of the second image.

        Returns:
            ORB score.
        """
        # Extract ORB features for both images
        kp1, des1 = Metric.extract_orb_features(img1)
        kp2, des2 = track_orb_features

        if kp1 is None or kp2 is None or des1 is None or des2 is None:
            return 0

        # consider the appropriate reference for keypoint matches
        num_kp = min(len(kp1), len(kp2))

        # create BFMatcher object and match descriptors
        bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # ratio test as per Lowe's paper
        good_matches = []
        for match in matches:
            if len(match) != 2:
                continue
            m, n = match
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)

        score = len(good_matches) / num_kp

        return score

    @staticmethod
    def calculate_iou_metric(detections, tracks):
        """
        Calculate the Intersection over Union (IoU) metric between detections and tracks.

        Args:
            detections: List of bounding box coordinates for detections.
            tracks: List of bounding box coordinates for tracks.

        Returns:
            IoU matrix.
        """
        iou_matrix = np.zeros(shape=(len(detections), len(tracks)), dtype=np.float32)

        # Calculate costs for iou matrix
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                iou_matrix[d, t] = calculate_iou(det, trk)

        return iou_matrix

    @staticmethod
    def calculate_orb_metric(detections, track_orb_features):
        """
        Calculate the ORB metric between detections and track ORB features.

        Args:
            detections: List of bounding box coordinates for detections.
            track_orb_features: List of ORB features for tracks.

        Returns:
            ORB metric matrix.
        """
        orb_matrix = np.zeros(shape=(len(detections), len(track_orb_features)), dtype=np.float32)

        # Calculate costs for iou matrix
        for d, det in enumerate(detections):
            for t, trk in enumerate(track_orb_features):
                orb_matrix[d, t] = Metric.calculate_orb_score(det, trk)

        return orb_matrix
