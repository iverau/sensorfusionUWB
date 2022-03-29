from math import degrees
from Sensors.CameraSensor2.camera import PinholeCamera
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
class VisualOdometry:

    def __init__(self, rot_init, t_init) -> None:
        self.camera = PinholeCamera()
        self.detector = cv2.ORB_create(2000, 1.2, 8)
        self.lk_params = dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.old_image = None
        self.scale = 1.0
        self.n_features = 0

        self.initial_rotation = rot_init
        self.initial_position = t_init

        # States
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.x = []
        self.y = []
        self.z = []

        # -90 grader rundt z, -90 grader i x
        self.cam_t_body = Rot.from_euler('xyz', [-90, -90, 0], degrees=True).as_matrix()

    def detect(self, img):
        points = self.detector.detect(img)
        return np.array([x.pt for x in points], dtype=np.float32).reshape(-1, 1, 2)

    def track(self, image):
        image = np.array(image)
        # Track stuff
        if self.old_image is not None:
            
            if self.n_features < 2000:
                self.old_points = self.detect(self.old_image)

            new_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_image, image, self.old_points, None, **self.lk_params)


            self.good_old = self.old_points[st==1]
            self.good_new = new_points[st==1]

            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.camera.K, cv2.RANSAC, 0.999, 1.0, None)
            _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R)
            
            # Transform the rotation from camera coordinates to 
            R = self.cam_t_body @ R
            t = self.cam_t_body @ t



            #Kinematic equations for VO in NED
            self.t = self.t +  self.R @ t
            self.R = self.R.dot(R)

            print(self.t)


            rotation = Rot.from_matrix(self.R)

            self.roll.append( rotation.as_euler("xyz", degrees=True)[0])
            self.pitch.append( rotation.as_euler("xyz", degrees=True)[1])
            self.yaw.append( rotation.as_euler("xyz", degrees=True)[2])

            self.x.append(self.t.copy()[0])
            self.y.append(self.t.copy()[1])
            self.z.append(self.t.copy()[2])



            # Reset the variables to the new varaibles
            self.old_image = image
            self.old_points = new_points
            self.n_features = self.good_new.shape[0]

            

            cv2.imshow("Frame", image)
            cv2.waitKey(1)

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

            # Initial rotation and transelation set to ground truth values
            self.R = self.initial_rotation
            self.t = np.asarray([self.initial_position]).T

    def update_scale(self):
        # Update the scale parameter 
        pass