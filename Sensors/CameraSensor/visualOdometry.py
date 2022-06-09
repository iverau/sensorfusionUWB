from unittest import TextTestRunner
from Sensors.CameraSensor.camera import PinholeCamera
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])


def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])


def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])


def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    uvw = K@X[:3, :]
    uvw /= uvw[2, :]
    return uvw[:2, :]


def levenberg_marquardt(residualsfun, p0, num_iterations=100, finite_difference_epsilon=1e-5, mu=1e-3):
    eps = finite_difference_epsilon
    p = p0.copy()
    mu_temp = mu
    for iteration in range(num_iterations):
        # Calculating the Jacobian of p using finite differences for the left part of J
        J = []
        for i in range(len(p)):
            p_temp_plus = p.copy()
            p_temp_minus = p.copy()
            p_temp_plus[i] += eps
            p_temp_minus[i] -= eps

            J_i = (residualsfun(p_temp_plus) - residualsfun(p_temp_minus)) / (2*eps)
            J.append(J_i)
        J = np.array(J)
        J = J.T

        # Calculating the normal equations
        A = J.T @ J
        A += mu_temp * np.eye(A.shape[0])
        b = - J.T @ residualsfun(p)

        # Calculating delta by solving the modified normal equations
        delta = np.linalg.solve(A, b)
        if costFuncLinearization(residualsfun(p), J, delta) < costFunc(residualsfun(p)):
            mu_temp /= 3

        while(costFuncLinearization(residualsfun(p), J, delta) > costFunc(residualsfun(p))):
            mu_temp *= 2
            A = J.T @ J
            A += mu_temp * np.eye(len(A))
            b = - J.T @ residualsfun(p)

            delta = np.linalg.solve(A, b)

        # Improving p based on the desired search direction
        p = p + delta
        if (np.linalg.norm(delta) < 0.00001):
            break

    return p

# Calculating the cost given the residuals (sum of squares)


def costFunc(residuals):
    cost = 0
    for element in residuals:
        cost += element**2
    return cost

# Calculating the cost given the residuals and the step based on delta (sum of squares)


def costFuncLinearization(residuals, J, delta):
    cost = 0
    for i in range(len(residuals)):
        cost += (residuals[i] + J[i, :] @ delta)**2
    return cost


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
    R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
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
    T[:3, 3] = t.flatten().T
    return T


class VisualOdometry:

    def __init__(self, noise_values=0) -> None:
        self.noise_values_init = noise_values
        self.noise_values = noise_values
        self.camera = PinholeCamera()
        self.detector = cv2.ORB_create(nfeatures=2500)
        self.old_image = None
        self.scale = 1.0
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)

        # States
        self.states = []
        self.noise_counter = 1

        self.body_t_cam = Rot.from_euler('xyz', [0.823, -2.807, 8.303], degrees=True).as_matrix()  @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

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
        T = best_T
        self.T = best_T
        X = best_X1
        return X, T

    def reset_initial_conditions(self):
        #self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.noise_counter = 1

    def residualFunction(self, uv, p, XY01):
        r = np.empty(2*uv.shape[1])
        uv_hat = project(self.camera.K, self.rotationMatrix(p)[:3] @ XY01)
        r[:uv.shape[1]] = (uv_hat[0] - uv[0])
        r[uv.shape[1]:] = (uv_hat[1] - uv[1])
        return r**2

    # Create a rotation matrix based on pose p

    def rotationMatrix(self, p):
        rotation = rotate_x(p[0]) @ rotate_y(p[1]) @ rotate_z(p[2]) @ translate(p[3], p[4], p[5]) @ self.T
        return rotation

    def track(self, image):
        image = self.camera.undistort_image(np.array(image))

        # Track stuff
        if self.old_image is not None:

            self.kp1, self.des1 = self.detector.detectAndCompute(self.old_image, None)
            self.kp2, self.des2 = self.detector.detectAndCompute(image, None)
            matches = self.matcher.knnMatch(self.des1, self.des2, k=2)

            imageIndexes = []
            good = []
            for m, n in matches:
                if m.distance < 0.8*n.distance:
                    imageIndexes.append([m.queryIdx, m.trainIdx])
                    good.append([m])

            img3 = cv2.drawMatchesKnn(cv2.cvtColor(self.old_image, cv2.COLOR_BGR2RGB), self.kp1, cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB), self.kp2, good[:5], None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            plt.imshow(img3)
            plt.show()
            exit()
            self.remove_outliers_with_ransac(imageIndexes)
            self.E = estimate_E(self.xy1, self.xy2)
            # Start extrating T
            self.X, T = self.get_best_point_corespondence()
            p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            def residualsfun(p):
                return self.residualFunction(self.uv2, p, self.X)

            p = levenberg_marquardt(residualsfun, p0)
            T = rotate_x(p[0]) @ rotate_y(p[1]) @ rotate_z(p[2])  @ translate(p[3], p[4], p[5]) @ self.T

            t = -T[:3, 3].reshape((3, 1))
            R = T[:3, :3].T
            #rotation = self.createYawRotation(R)

            # Kinematic equations for VO in camera frame
            self.t = self.scale * self.R @ t + self.t
            self.R = self.R @ R
            rotation = self.body_t_cam @ self.R @ self.body_t_cam.T
            rotation = self.createYawRotation(rotation)

            self.states.append(SE3(rotation, self.body_t_cam @ self.t))

            # Reset the variables to the new varaibles
            self.old_image = image

            # Show the images at each iteration
            keypoints = self.detector.detect(image, None)
            keypoints, descriptors = self.detector.compute(image, keypoints)
            new_img = cv2.drawKeypoints(
                image, keypoints, None, color=(0, 255, 0), flags=0)
            cv2.imshow("Frame", new_img)
            cv2.waitKey(1)
            return rotation, self.body_t_cam @ self.t

        else:
            # Case for first image
            self.old_image = image

            self.R = np.eye(3)
            self.t = np.zeros((3, 1))
            rotation = self.body_t_cam @ self.R @ self.body_t_cam.T
            rotation = self.createYawRotation(rotation)

            self.states.append(SE3(rotation, self.body_t_cam @ self.t))
            return rotation, self.body_t_cam @ self.t

    def update_scale(self, scale):
        self.scale = scale

    def createYawRotation(self, rotation):
        temp_rot = Rot.from_matrix(rotation).as_euler("xyz")
        return Rot.from_euler("xyz", [0, 0, temp_rot[2]]).as_matrix()

    def remove_outliers_with_ransac(self, imageIndexes):
        # Extract the uv points and their projection from the images
        uv1, uv2 = getCommonImagePoints(imageIndexes, self.kp1, self.kp2)
        xy1 = self.camera.Kinv @ uv1.T
        xy2 = self.camera.Kinv @ uv2.T

        # Calculate amount of the ransac trials and run ransac on the matches
        num_trials = get_num_ransac_trials(8, 0.999, 0.50)
        _, inliers = estimate_E_ransac(xy1, xy2, self.camera.K, 1.0, num_trials)
        # Remove outliers from the image coordinates
        self.xy1 = xy1[:, inliers]
        self.xy2 = xy2[:, inliers]
        self.uv1 = uv1.T[:, inliers]
        self.uv2 = uv2.T[:, inliers]
