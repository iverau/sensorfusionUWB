from Sensors.CameraSensor.featureDetector import *
import cv2
import numpy as np
from enum import Enum


# Resources:
#   https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
#   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html


FEATURE_COLOR = (0,0, 255)
MATCH_COLOR = (0,255,0)
FEATURE_SIZE = 2
NORM_TYPE = cv2.NORM_HAMMING
LOWEST_RATIO = 0.5
NUMBER_OF_FEATURES = 2000
GOOD_MATCHES_RATIO = 0.3
LUKAS_KANADE_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=max(3, 3),
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
) 


class FeatureMatcherType(Enum):
    BRUTE_FORCE = 1
    BRUTE_FORCE_RATIO_TEST = 2
    OPTICAL_FLOW = 3


def feature_matcher_factory(feature_matcher_type: FeatureMatcherType):
    if feature_matcher_type == FeatureMatcherType.BRUTE_FORCE:
        return BFMatcher()
    if feature_matcher_type == FeatureMatcherType.BRUTE_FORCE_RATIO_TEST:
        return BFRatioTestMatcher()
    if feature_matcher_type == FeatureMatcherType.OPTICAL_FLOW:
        return OFMatcher()


class FeatureMatcher:
    """Class for matching features"""

    def __init__(self):
        pass

    def match(kps_cur, kps_ref):
        return None, None, None

    @staticmethod
    def draw(kps_cur, kps_ref):
        pass

    @staticmethod
    def draw_matches_side_by_side(
        image_cur, image_ref, kps_cur, kps_ref, feat_color=FEATURE_COLOR
    ):
        # Concatenate ref image and cur image as left and right image
        img = np.concatenate([image_ref, image_cur], axis=1)
        # Exteact keypoint image coordinates
        pts_ref = FeatureDetector.extract_image_coordinates(kps_ref)
        pts_cur = FeatureDetector.extract_image_coordinates(kps_cur)

        for pt1, pt2 in zip(pts_ref, pts_cur):
            a1, b1 = pt1.ravel()
            a2, b2 = pt2.ravel()
            a2 += image_cur.shape[
                1
            ]  # add image size as offset for plotting on features on right image

            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img, (int(a1), int(b1)), (int(a2), int(b2)), color, 1)
            cv2.circle(img, (int(a1), int(b1)), 1, feat_color, -1)
            cv2.circle(img, (int(a2), int(b2)), 1, feat_color, -1)

        return img

    @staticmethod
    def draw_matches_side_by_side_opencv(
        image_cur, image_ref, kps_cur, kps_ref, matches
    ):
        """Draw matches given two images and feature keypoints"""
        return cv2.drawMatches(
            image_cur,
            kps_cur,
            image_ref,
            kps_ref,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    @staticmethod
    def extract_matched_keypoints(kps_cur, kps_ref, des_cur, des_ref, matches):
        """Extract keypoints that are matched"""
        kps_cur_matched = []
        des_cur_matched = []
        kps_ref_matched = []
        des_ref_matched = []
        for m in matches:
            kps_cur_matched.append(kps_cur[m.queryIdx])
            des_cur_matched.append(des_cur[m.queryIdx])

            kps_ref_matched.append(kps_ref[m.trainIdx])
            des_ref_matched.append(des_cur[m.trainIdx])

        return (
            np.array(kps_cur_matched),
            np.array(kps_ref_matched),
            np.array(des_cur_matched),
            np.array(des_ref_matched),
        )

    @staticmethod
    def draw_matches(
        image_ref,
        kps_ref,
        kps_cur,
        feat_color=FEATURE_COLOR,
        match_color=MATCH_COLOR,
        feat_size=FEATURE_SIZE,
    ):
        """Draw keypoint matches as gradient lines on a single image.
        Draw kps_cur as circles, and draw lines between points kps_cur and kps_ref"""
        # Exteact keypoint image coordinates
        pts_ref = FeatureDetector.extract_image_coordinates(kps_ref)
        pts_cur = FeatureDetector.extract_image_coordinates(kps_cur)

        for pt1, pt2 in zip(pts_ref, pts_cur):
            a1, b1 = pt1.ravel()
            a2, b2 = pt2.ravel()
            cv2.line(image_ref, (int(a1), int(b1)), (int(a2), int(b2)), match_color, 1)
            cv2.circle(image_ref, (int(a1), int(b1)), feat_size, feat_color, -1)

        return image_ref


class BFMatcher(FeatureMatcher):
    def __init__(
        self,
        norm_type=NORM_TYPE,
        lowes_ratio=LOWEST_RATIO,
    ):
        super().__init__()
        """Initiate feature matcher with ratio test"""
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.lowes_ratio = lowes_ratio

    def match(self, img_cur, img_ref, kps_cur, kps_ref, des_cur, des_ref):
        """Feature matching using Lowe's test.
        Images are not used for matching."""
        assert len(des_cur) == len(des_ref), "len(ref) and len(cur) features not equal"
        good_matches = self.matcher.match(des_cur, des_ref)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        good_matches = good_matches[: int(NUMBER_OF_FEATURES * GOOD_MATCHES_RATIO)]
        # Extract keypoints that are matched
        (
            kps_cur_matched,
            kps_ref_matched,
            des_cur_matched,
            des_ref_matched,
        ) = super().extract_matched_keypoints(
            kps_cur, kps_ref, des_cur, des_ref, good_matches
        )
        return (
            np.array(kps_cur_matched),
            np.array(kps_ref_matched),
            np.array(des_cur_matched),
            np.array(des_ref_matched),
        )


class BFRatioTestMatcher(FeatureMatcher):
    def __init__(
        self,
        norm_type=NORM_TYPE,
        lowes_ratio=LOWEST_RATIO,
    ):
        super().__init__()
        """Initiate feature matcher with ratio test"""
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        self.lowes_ratio = lowes_ratio

    def match(self, img_cur, img_ref, kps_cur, kps_ref, des_cur, des_ref):
        """Feature matching using Lowe's test.
        Images are not used for matching."""
        assert len(des_cur) == len(des_ref), "len(ref) and len(cur) features not equal"
        matches = self.matcher.knnMatch(des_cur, des_ref, k=2)

        good_matches = []
        for m, n in matches:
            # Apply ratio test
            if m.distance < self.lowes_ratio * n.distance:
                good_matches.append(m)

        # Extract keypoints that are matched
        (
            kps_cur_matched,
            kps_ref_matched,
            des_cur_matched,
            des_ref_matched,
        ) = super().extract_matched_keypoints(
            kps_cur, kps_ref, des_cur, des_ref, good_matches
        )
        return (
            np.array(kps_cur_matched),
            np.array(kps_ref_matched),
            np.array(des_cur_matched),
            np.array(des_ref_matched),
        )


class OFMatcher(FeatureMatcher):
    def __init__(self, lk_params=LUKAS_KANADE_PARAMS):
        super().__init__()
        self.lk_params = lk_params

    def match(self, img_cur, img_ref, kps_cur, kps_ref, des_cur, des_ref):
        """Calculates an optical flow for a sparse feature set using the iterative
        Lucas-Kanade (LK) method with pyramids. Descriptors are not used here for matching.
        NOTE: With LK we follow feature trails hence we can forget unmatched features"""
        # Gray images are used for matching
        img_gray_cur = cv2.cvtColor(img_cur, cv2.COLOR_RGB2GRAY)
        img_gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)

        # Extract keypoint image coordinates
        pts_ref = FeatureDetector.extract_image_coordinates(kps_ref)

        # Match features using optical flow
        pts_cur, mask_match, err = cv2.calcOpticalFlowPyrLK(
            img_gray_ref, img_gray_cur, pts_ref, None, **self.lk_params
        )
        # Extract matched keypoints
        mask_match = mask_match.reshape(mask_match.shape[0])
        idxs = [i for i, v in enumerate(mask_match) if v == 1]
        pts_ref_matched = pts_ref[idxs]
        pts_cur_matched = pts_cur[idxs]

        # Convert image coordinates into list of keypoints. Image coordinate points
        # are converted to keypoints for convenience. Size of keypoint is set to 0
        kps_ref_matched = FeatureDetector.convert_to_keypoints(pts_ref_matched, size=0)
        kps_cur_matched = FeatureDetector.convert_to_keypoints(pts_cur_matched, size=0)

        # No descriptors are used/processed here
        des_ref_matched = np.full(kps_ref_matched.shape, None)
        des_cur_matched = np.full(kps_cur_matched.shape, None)
        return (
            np.array(kps_cur_matched),
            np.array(kps_ref_matched),
            np.array(des_cur_matched),
            np.array(des_ref_matched),
        )
