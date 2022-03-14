import cv2
import numpy as np
from enum import Enum

# Resource: https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html


FEATURE_COLOR = (0,0, 255)
MATCH_COLOR = (0,255,0)
FEATURE_SIZE = 2
NUMBER_OF_FEATURES = 2000
ORB_SCALE_FACTOR = 1.2
ORB_NUM_LEVELS = 3

#Shi-thomasi detector
QUALITY_LEVEL = 0.01
MIN_CORNER_DIST = 3
BLOCK_SIZE = 5


class FeatureDetectorType(Enum):
    SHI_THOMASI = 1
    ORB = 2


def feature_detector_factory(feature_detector_type: FeatureDetectorType):
    if feature_detector_type == FeatureDetectorType.SHI_THOMASI:
        return ShiTomasiDetector()
    if feature_detector_type == FeatureDetectorType.ORB:
        return ORBDetector()


class FeatureDetector:
    """Class for detecting features"""

    def __init__(self):
        pass

    @staticmethod
    def draw_feature_keypoints(image, kps, color=FEATURE_COLOR):
        """Draw keypoints on image"""
        return cv2.drawKeypoints(image, kps, None, color, flags=0)

    @staticmethod
    def draw_feature_points(image, pts, feat_size=FEATURE_SIZE, color=FEATURE_COLOR):
        """Draw feature image coordinate points on image"""
        for p in pts:
            a, b = p.ravel()
            cv2.circle(image, (a, b), feat_size, color, -1)
        return image

    @staticmethod
    def extract_image_coordinates(kps):
        """Extract pixel points from keypoints"""
        return np.array([k.pt for k in kps], dtype=np.float32)

    @staticmethod
    def convert_to_keypoints(pts, size=0.0):
        """Convert image coordinates to keypoints"""
        kps = []
        if pts is not None:
            kps = [cv2.KeyPoint(p[0], p[1], size) for p in pts]
        return np.array(kps)


class ORBDetector(FeatureDetector):
    def __init__(
        self,
        n_features=NUMBER_OF_FEATURES,
        scale_factor=ORB_SCALE_FACTOR,
        n_levels=ORB_NUM_LEVELS,
    ):
        super().__init__()
        """Initiate feature detector"""
        # ORB detector
        self.n_features = n_features
        self.detector = cv2.ORB_create(
            nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels
        )

    def detect_and_compute(self, image, mask=None):
        """Detect and compute keypoints and descriptors"""
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kps, des = self.detector.detectAndCompute(img_gray, None)
        assert len(kps) == self.n_features, "Did not detect desired number of features"
        return np.array(kps), np.array(des)


class ShiTomasiDetector(FeatureDetector):
    def __init__(
        self,
        n_features=NUMBER_OF_FEATURES,
        quality_level=QUALITY_LEVEL,
        min_corner_distance=MIN_CORNER_DIST,
        block_size=BLOCK_SIZE,
    ):
        super().__init__()
        self.num_features = n_features
        self.quality_level = quality_level
        self.min_corner_distance = min_corner_distance
        self.block_size = block_size

    def detect_and_compute(self, image, mask=None):
        """Detect and compute keypoints and descriptors.
        Descriptors for ShiTomashi features is None"""
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Uses cornerMinEigenVal as detector
        pts = cv2.goodFeaturesToTrack(
            img_gray,
            self.num_features,
            self.quality_level,
            self.min_corner_distance,
            blockSize=self.block_size,
            mask=mask,
        )
        kps, des = [], []
        if pts is not None:
            # Convert image coordinates into list of keypoints
            kps = [cv2.KeyPoint(p[0][0], p[0][1], self.block_size) for p in pts]
        return np.array(kps), np.array(des)
