from math import degrees
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation as R


MIN_NUM_FEATURES = 2000
SCALE_THRESHOLD = 0.1


class VOState(Enum):
    NO_IMAGES_YET = 0
    GOT_FIRST_IMAGE = 1


class VisualOdometry:
    def __init__(self, feature_detector, feature_matcher, epipolar_geometry):

        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.epipolar_geometry = epipolar_geometry

        # Initialize pose
        self.cur_R = np.eye(3, 3)
        self.cur_t = np.zeros((3, 1))
        self.R = np.eye(3, 3)
        self.t = np.zeros((3, 1))
        self.traj3d_est = []
        self.traj3d_gt = []

        # Initialize variables
        self.kps_cur = None
        self.kps_ref = None
        self.des_cur = None
        self.des_ref = None
        self.matches = None
        self.img_feat = None
        self.img_match = None

        self.pose_initialized = False
        self.state = VOState.NO_IMAGES_YET

    def track(self, image):
        self.img_cur = image
        #self.gt = 1

        if self.state == VOState.NO_IMAGES_YET:
            self.process_first_image()
        elif self.state == VOState.GOT_FIRST_IMAGE:
            self.process_image()

        self.img_ref = self.img_cur

    def process_first_image(self):
        # Detect reference features
        self.kps_ref, self.des_ref = self.feature_detector.detect_and_compute(
            self.img_cur
        )
        self.state = VOState.GOT_FIRST_IMAGE

    def process_image(self):
        self.match_features()
        self.estimate_pose()
        self.extract_inliers()
        self.draw_inlier_matches()
        self.calculate_pose()
        self.update_reference_features()

    def match_features(self):
        # Match current features to reference/previous features
        (
            self.kps_cur_matched,
            self.kps_ref_matched,
            self.des_cur_matched,
            self.des_ref_matched,
        ) = self.feature_matcher.match(
            self.img_cur,
            self.img_ref,
            self.kps_cur,
            self.kps_ref,
            self.des_cur,
            self.des_ref,
        )
        self.num_matches = len(self.kps_cur_matched)

    def estimate_pose(self):
        # Extract image coordinates and estimate pose of the cur frame relative to ref frame
        pts_cur_matched = self.feature_detector.extract_image_coordinates(
            self.kps_cur_matched
        )
        pts_ref_matched = self.feature_detector.extract_image_coordinates(
            self.kps_ref_matched
        )
        # The translation t is a unit vector.
        (
            _retval,
            self.R,
            self.t,
            self.mask_inlier,
            _mask_pose,
        ) = self.epipolar_geometry.estimate_pose(pts_cur_matched, pts_ref_matched)

    def extract_inliers(self):
        # Filter matches using mask from essential matrix calculation
        # These are defined as inlier features
        self.kps_cur_inliers = self.kps_cur_matched[
            [m[0] == 1 for m in self.mask_inlier]
        ]
        self.kps_ref_inliers = self.kps_ref_matched[
            [m[0] == 1 for m in self.mask_inlier]
        ]
        self.num_inliers = len(self.kps_cur_inliers)

    def draw_inlier_matches(self):
        img_match = self.img_cur.copy()
        self.img_match = self.feature_matcher.draw_matches(
            img_match, self.kps_ref_inliers, self.kps_cur_inliers
        )

    def update_reference_features(self):
        # If number of matched features are less than total number of
        # features to detect, then detect new features.
        if self.kps_cur_matched.shape[0] < MIN_NUM_FEATURES:
            self.kps_ref, self.des_ref = self.feature_detector.detect_and_compute(
                self.img_cur
            )
        else:
            self.kps_ref, self.des_ref = self.kps_cur_matched, self.kps_ref_matched

    def calculate_pose(self):
        assert 1 > SCALE_THRESHOLD, "Absolute scale not > 0.1"
        # compose absolute motion [Rwa,twa] with estimated relative motion [Rab,s*tab]
        # (s is the scale extracted from the ground truth)
        # [Rwb,twb] = [Rwa,twa]*[Rab,tab] = [Rwa*Rab|twa + Rwa*tab]

        # Predict pose
        self.cur_t = self.cur_t + 1 * self.cur_R.dot(self.t)
        self.cur_R = self.cur_R.dot(self.R)
        rotation = R.from_matrix(self.cur_R)
        print(rotation.as_euler("zyx", degrees=True))

        # Calculate start translation
        if not self.pose_initialized:
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])
            #self.t0_gt = np.array([self.gt.x, self.gt.y, self.gt.z])
            self.pose_initialized = True

        # Trajectory starts at translation=0
        if self.pose_initialized:
            # Estimate
            p_est = [
                self.cur_t[0] - self.t0_est[0],
                self.cur_t[1] - self.t0_est[1],
                self.cur_t[2] - self.t0_est[2],
            ]
            self.traj3d_est.append(p_est)
            # Groundtruth
            #p_gt = [
            #    self.gt.x - self.t0_gt[0],
            #    self.gt.y - self.t0_gt[1],
            #    self.gt.z - self.t0_gt[2],
            #]
            #self.traj3d_gt.append(p_gt)
