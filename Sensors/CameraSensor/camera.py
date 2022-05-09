import numpy as np
import cv2


class PinholeCamera:

    def __init__(self) -> None:
        self.width = 1920
        self.height = 1200
        self.freq = 10
        self.K = np.array([[1995.4, 0, 965.5],
                           [0, 1995.2, 605.6],
                           [0, 0, 1]])
        self.Kinv = np.linalg.inv(self.K)
        self.dist = np.array([-0.14964, 0.13337, 0.0, 0.0, 0.0])

    def undistort_image(self, img):
        optimalMatrix, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (self.width, self.height), 1, (self.width, self.height))
        undistorted_image = cv2.undistort(img, self.K, self.dist, None, optimalMatrix)
        x, y, w, h = roi
        return undistorted_image[y:y+h, x:x+w]

    def undistort_points(self, uv):
        uvs_undistorted = cv2.undistortPoints(uv, self.K, self.dist, None, self.K)
        return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)

    def normalize_image_coordinates(self, uv):
        """Normalize image pixel coordinates using camera intrinsics
        and homogenous coordinates
        Args:
            uvs : [Nx2]
        Returns:
            xcs: [Nx3] of normalized pixel coordinates
        """
        return np.dot(self.Kinv, add_ones(uv).T).T[:, 0:2]


def add_ones(x):
    """Add ones to work in homogenous coordinates"""
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
