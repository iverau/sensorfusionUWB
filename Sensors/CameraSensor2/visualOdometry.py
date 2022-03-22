from Sensors.CameraSensor2.camera import PinholeCamera
import cv2
import matplotlib.pyplot as plt
class VisualOdometry:

    def __init__(self) -> None:
        self.camera = PinholeCamera()
        self.detector = cv2.ORB_create(2000, 1.2, 8)
        self.lk_params = None
        self.image = None

    def track(self, image):
        # Track stuff
        if self.image is not None:
            keypoints = self.detector.detect(image, None)
            keypoints, descriptors = self.detector.compute(image, keypoints)
            new_img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
            plt.imshow(new_img)
            plt.show()

        else:
            # Case for first image
            self.image = image

    def update_scale(self):
        # Update the scale parameter 
        pass