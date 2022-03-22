from Sensors.CameraSensor2.camera import PinholeCamera
import cv2
import matplotlib.pyplot as plt
import numpy as np
class VisualOdometry:

    def __init__(self) -> None:
        self.camera = PinholeCamera()
        self.detector = cv2.ORB_create(2000, 1.2, 8)
        self.lk_params = dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.image = None

    def detect(self, img):
        points = self.detector.detect(img)
        return np.array([x.pt for x in points], dtype=np.float32).reshape(-1, 1, 2)

    def track(self, image):
        # Track stuff
        if self.image is not None:
            
            new_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_image, image, self.old_points, None, **self.lk_params)


            self.good_old = self.old_points[st==1]
            self.good_new = new_points[st==1]

            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.camera.K, cv2.RANSAC, 0.999, 1.0, None)
            _, self.R, self.t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.camera.K self.R.copy(), self.t.copy(), None)

            # Reset the variables to the new varaibles

            """
            keypoints = self.detector.detect(image, None)
            keypoints, descriptors = self.detector.compute(image, keypoints)
            new_img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
            plt.imshow(new_img)
            plt.show()
            """

        else:
            # Case for first image
            self.old_image = image
            self.old_points = self.detect(image)

    def update_scale(self):
        # Update the scale parameter 
        pass