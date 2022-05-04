from Sensors.CameraSensor2.camera import PinholeCamera
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def getCommonImagePoints(imageIndexes, kp1, kp2):
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    imageIndexes = np.array(imageIndexes)

    relevantKp1 = kp1[imageIndexes[:, 0]]
    relevantKp2 = kp2[imageIndexes[:, 1]]

    uv1 = []
    uv2 = []
    for k1, k2 in zip(relevantKp1, relevantKp2):
        uv1.append([k1.pt[0], k1.pt[1], 1])
        uv2.append([k2.pt[0], k2.pt[1], 1])

    return np.array(uv1), np.array(uv2)


def get_num_ransac_trials(sample_size, confidence, inlier_fraction):
    return int(np.log(1 - confidence)/np.log(1 - inlier_fraction**sample_size))


def estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials):

    # Tip: The following snippet extracts a random subset of 8
    # correspondences (w/o replacement) and estimates E using them.
    #   sample = np.random.choice(xy1.shape[1], size=8, replace=False)
    #   E = estimate_E(xy1[:,sample], xy2[:,sample])

    uv1 = K@xy1
    uv2 = K@xy2

    best_num_inliers = -1
    for i in range(num_trials):
        sample = np.random.choice(xy1.shape[1], size=8, replace=False)
        E_i = estimate_E(xy1[:, sample], xy2[:, sample])
        d_i = epipolar_distance(F_from_E(E_i, K), uv1, uv2)
        inliers_i = np.absolute(d_i) < distance_threshold
        num_inliers_i = np.sum(inliers_i)
        if num_inliers_i > best_num_inliers:
            best_num_inliers = num_inliers_i
            E = E_i
            inliers = inliers_i

    return E, inliers


def F_from_E(E, K):
    K_inv = np.linalg.inv(K)
    F = K_inv.T@E@K_inv
    return F


def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    l2 = F@uv1
    l1 = F.T@uv2
    e = np.sum(uv2*l2, axis=0)
    norm1 = np.linalg.norm(l1[:2, :], axis=0)
    norm2 = np.linalg.norm(l2[:2, :], axis=0)
    return 0.5*e*(1/norm1 + 1/norm2)


def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.empty((n, 9))
    for i in range(n):
        x1, y1 = xy1[:2, i]
        x2, y2 = xy2[:2, i]
        A[i, :] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    _, _, VT = np.linalg.svd(A)
    return np.reshape(VT[-1, :], (3, 3))


def triangulate_many(xy1, xy2, P1, P2):
    """
    Arguments
        xy: Calibrated image coordinates in image 1 and 2
            [shape 3 x n]
        P:  Projection matrix for image 1 and 2
            [shape 3 x 4]
    Returns
        X:  Dehomogenized 3D points in world frame
            [shape 4 x n]
    """
    n = xy1.shape[1]
    X = np.empty((4, n))
    for i in range(n):
        A = np.empty((4, 4))
        A[0, :] = P1[0, :] - xy1[0, i]*P1[2, :]
        A[1, :] = P1[1, :] - xy1[1, i]*P1[2, :]
        A[2, :] = P2[0, :] - xy2[0, i]*P2[2, :]
        A[3, :] = P2[1, :] - xy2[1, i]*P2[2, :]
        U, s, VT = np.linalg.svd(A)
        X[:, i] = VT[3, :]/VT[3, 3]
    return X


def decompose_E(E):
    """
    Computes the four possible decompositions of E into a relative
    pose, as described in Szeliski 7.2.

    Returns a list of 4x4 transformation matrices.
    """
    U, _, VT = np.linalg.svd(E)
    R90 = np.array([[0, -1, 0], [+1, 0, 0], [0, 0, 1]])
    R1 = U @ R90 @ VT
    R2 = U @ R90.T @ VT
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    t1, t2 = U[:, 2], -U[:, 2]
    return [SE3(R1, t1), SE3(R1, t2), SE3(R2, t1), SE3(R2, t2)]


def SE3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class VisualOdometry:

    def __init__(self, rot_init, t_init, noise_values=0) -> None:
        self.noise_values_init = noise_values
        self.noise_values = noise_values
        self.camera = PinholeCamera()
        self.detector = cv2.ORB_create(2000, 1.2, 8)
        self.lk_params = dict(winSize=(21, 21), criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.old_image = None
        self.scale = 0.5
        self.n_features = 0
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        self.initial_rotation = rot_init
        self.initial_position = t_init

        # States
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.East = []
        self.North = []
        self.Down = []
        self.noise_counter = 1
        # -90 grader rundt z, -90 grader i x
        self.body_t_cam = Rot.from_euler('xyz', [0.823, -2.807, 8.303], degrees=True).as_matrix().T  @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    def detect(self, img):
        points = self.detector.detect(img)
        return np.array([x.pt for x in points], dtype=np.float32).reshape(-1, 1, 2)

    def get_best_point_corespondence(self):
        T4 = decompose_E(self.E)
        best_num_visible = 0
        for T in T4:
            P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
            P2 = T[:3, :]
            X1 = triangulate_many(self.xy1, self.xy2, P1, P2)
            X2 = T@X1
            num_visible = np.sum((X1[2, :] > 0) & (X2[2, :] > 0))
            if num_visible > best_num_visible:
                best_num_visible = num_visible
                best_T = T
                best_X1 = X1
        self.T = best_T
        X = best_X1
        return X, self.T

    def reset_initial_conditions(self, rot_init, t_init):
        self.R = self.body_t_cam @ rot_init
        self.t = self.body_t_cam @  rot_init @ np.asarray([t_init]).T
        self.noise_counter = 1

    def track_odometry2(self, image):
        image = np.array(image)
        # Track stuff
        if self.old_image is not None:

            self.kp1, self.des1 = self.detector.detectAndCompute(self.old_image, None)
            self.kp2, self.des2 = self.detector.detectAndCompute(image, None)

            matches = self.matcher.knnMatch(self.des1, self.des2, k=2)

            imageIndexes = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    imageIndexes.append([m.queryIdx, m.trainIdx])

            self.remove_outliers_with_ransac(imageIndexes)
            self.E = estimate_E(self.xy1, self.xy2)

            # Start extrating T
            self.X, T = self.get_best_point_corespondence()

            t = T[:3, 3]
            #t[0] = 0
            R = T[:3, :3]
            t = np.asarray([t]).T

            # Kinematic equations for VO in NED
            self.t += self.scale * self.R @ t
            self.R = R.dot(self.R)

            rotation = Rot.from_matrix(self.body_t_cam.T @ self.R)
            #rotation = Rot.from_matrix(rotation)
            rotation = rotation.as_euler("xyz")
            print("ROtation:", Rot.from_euler("xyz", [0, 0, rotation[2]]).as_euler("xyz"))
            rotation = Rot.from_euler("xyz", [0, 0, rotation[2]]).as_matrix()
            # Reset the variables to the new varaibles
            self.old_image = image

            keypoints = self.detector.detect(image, None)
            keypoints, descriptors = self.detector.compute(image, keypoints)
            new_img = cv2.drawKeypoints(
                image, keypoints, None, color=(0, 255, 0), flags=0)
            cv2.imshow("Frame", new_img)
            cv2.waitKey(1)
            self.noise_values = self.noise_counter * self.noise_values_init
            self.noise_counter += 1
            return rotation, self.body_t_cam.T @ self.t.copy()

        else:
            # Case for first image
            self.old_image = image
            self.old_points = self.detect(image)

            # Initial rotation and transelation set to ground truth values
            self.R = self.body_t_cam @ self.initial_rotation
            self.t = self.body_t_cam @  self.initial_rotation @ np.asarray(
                [self.initial_position]).T

            rotation = Rot.from_matrix(self.body_t_cam.T @ self.R)

    def track(self, image):
        image = np.array(image)
        # Track stuff
        if self.old_image is not None:

            self.kp1, self.des1 = self.detector.detectAndCompute(self.old_image, None)
            self.kp2, self.des2 = self.detector.detectAndCompute(image, None)

            matches = self.matcher.knnMatch(self.des1, self.des2, k=2)

            imageIndexes = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    imageIndexes.append([m.queryIdx, m.trainIdx])

            self.remove_outliers_with_ransac(imageIndexes)
            self.E = estimate_E(self.xy1, self.xy2)

            # Start extrating T
            self.X, T = self.get_best_point_corespondence()

            t = T[:3, 3]
            t[0] = 0
            R = T[:3, :3]
            t = np.asarray([t]).T

            # Kinematic equations for VO in NED
            self.t = self.scale * self.R @ t + self.t
            self.R = R @ self.R

            rotation = Rot.from_matrix(self.R).as_euler("xyz")
            #self.R = Rot.from_euler("xyz", [0, rotation[1], 0]).as_matrix()
            rotation = Rot.from_matrix(self.body_t_cam.T @ self.R)

            self.roll.append(rotation.as_euler("xyz", degrees=False)[0])
            self.pitch.append(rotation.as_euler("xyz", degrees=False)[1])
            self.yaw.append(rotation.as_euler("xyz", degrees=False)[2])

            self.North.append(self.body_t_cam.T.dot(self.t.copy())[0])
            self.East.append(self.body_t_cam.T.dot(self.t.copy())[1])
            self.Down.append(self.body_t_cam.T.dot(self.t.copy())[2])

            # Reset the variables to the new varaibles
            self.old_image = image

            keypoints = self.detector.detect(image, None)
            keypoints, descriptors = self.detector.compute(image, keypoints)
            new_img = cv2.drawKeypoints(
                image, keypoints, None, color=(0, 255, 0), flags=0)
            cv2.imshow("Frame", new_img)
            cv2.waitKey(1)

        else:
            # Case for first image
            self.old_image = image
            self.old_points = self.detect(image)

            # Initial rotation and transelation set to ground truth values
            self.R = self.body_t_cam @ self.initial_rotation
            self.t = self.body_t_cam @  self.initial_rotation @ np.asarray([self.initial_position]).T

            rotation = Rot.from_matrix(self.body_t_cam.T @ self.R)
            self.roll.append(rotation.as_euler("xyz", degrees=False)[0])
            self.pitch.append(rotation.as_euler("xyz", degrees=False)[1])
            self.yaw.append(rotation.as_euler("xyz", degrees=False)[2])

            self.North.append(self.body_t_cam.T.dot(self.t.copy())[0])
            self.East.append(self.body_t_cam.T.dot(self.t.copy())[1])
            self.Down.append(self.body_t_cam.T.dot(self.t.copy())[2])

    def update_scale(self):
        # Update the scale parameter
        pass

    def remove_outliers_with_ransac(self, imageIndexes):
        # Extract the uv points and their projection from the images
        uv1, uv2 = getCommonImagePoints(imageIndexes, self.kp1, self.kp2)
        xy1 = self.camera.Kinv @ uv1.T
        xy2 = self.camera.Kinv @ uv2.T

        # Calculate amount of the ransac trials and run ransac on the matches
        num_trials = get_num_ransac_trials(8, 0.99, 0.50)
        _, inliers = estimate_E_ransac(
            xy1, xy2, self.camera.K, 4.0, num_trials)

        # Remove outliers from the image coordinates
        self.uv1 = uv1.T[:, inliers]
        self.uv2 = uv2.T[:, inliers]
        self.xy1 = xy1[:, inliers]
        self.xy2 = xy2[:, inliers]

        # Save the descriptor and keypoints for the model for later usage
        kp1 = np.array(self.kp1)
        imageIndexes = np.array(imageIndexes)
        des1 = np.array(self.des1)
        self.model_descriptor = des1[imageIndexes[:, 0]]
        self.model_descriptor = self.model_descriptor[inliers]
        self.model_keypoints = kp1[imageIndexes[:, 0]]
        self.model_keypoints = self.model_keypoints[inliers]
