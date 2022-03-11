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
        self.dist = np.array([-0.14964, 0.13337, 0.0, 0.0, 0.0])

    def undistort_image(self, img):
        optimalMatrix, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (self.width, self.height), 1, (self.width, self.height))
        undistorted_image = cv2.undistort(img, self.K, self.dist, None, optimalMatrix)
        x,y,w,h = roi
        return undistorted_image[y:y+h, x:x+w]
        
    def undistort_points(self, uv):
        uvs_undistorted = cv2.undistortPoints(uv, self.K, self.D, None, self.K)
        return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)