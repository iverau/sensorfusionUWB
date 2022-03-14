import cv2
import numpy as np

RANSAC_PROBABILITY = 0.999
RANSAC_THRESHOLD =  0.0003

class EpipolarGeometry:

    def __init__(self, cam) -> None:
        self.camera = cam

    def undistort_and_normalize_image_coordinates(self, points):
        """Undistort and normalize image coordinates"""
        points = self.cam.undistort_points(points)
        points = self.cam.normalize_image_coordinates(points)
        return points 

    def calculate_essential_matrix(self, pts_normalized_cur, pts_normalized_ref):
        """Calculate essential matrix using normalized image coordinates"""
        E, mask = cv2.findEssentialMat(
            pts_normalized_cur,
            pts_normalized_ref,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=RANSAC_PROBABILITY,
            threshold=RANSAC_THRESHOLD
        )
        return E, mask

    def calculate_fundemental_matrix_using_essential(self, pts_normalized_cur, pts_normalized_ref):
        """Calculates fundemental matrix using essential matrix"""
        E, mask = self.calculate_essential_matrix(
            pts_normalized_cur, pts_normalized_ref
        )
        # Equation: E = (K)^T     * F * K
        #         : F = (K_inv)^T * E * K_inv
        F = np.transpose(self.cam.Kinv) @ E @ self.cam.Kinv
        return F, mask

    def estimate_pose(self, pts_cur, pts_ref):
        """Estimate pose using normalized image coordinates and essential matrix
        NOTE: The translation t is a unit vector."""

        # Undistort and normalize image coordinates
        pts_normalized_cur = self.undistort_and_normalize_image_coordinates(pts_cur)
        pts_normalized_ref = self.undistort_and_normalize_image_coordinates(pts_ref)

        # Calculate essential matrix
        E, mask_essential = self.calculate_essential_matrix(
            pts_normalized_cur, pts_normalized_ref
        )
        # Extract pose from essential matrix
        retval, R, t, mask_pose = cv2.recoverPose(
            E, pts_normalized_cur, pts_normalized_ref, focal=1.0, pp=(0.0, 0.0)
        )
        # Mask_essential is used to extract inlier features
        # Unsure of the use of mask_pose?
        return retval, R, t, mask_essential, mask_pose

    @staticmethod
    def draw_lines(image1, image2, lines, pts1, pts2):
        """image1 on which we draw the lines. Corresponding points are drawn
        on corresponding images"""
        r, c, _ = image1.shape
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            # Generate random color for every line
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # Calculate image start and end coordinates
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            image1 = cv2.line(image1, (x0, y0), (x1, y1), color, 1)
            # Draw circles on each edges of the lines
            image1 = cv2.circle(image1, tuple(pt1), 5, color, -1)
            image2 = cv2.circle(image2, tuple(pt2), 5, color, -1)
        return image1, image2

    def draw_epilines(self, image_cur, image_ref, pts_cur_matched, pts_ref_matched):
        """Find epilines corresponding to points in image1 and
        drawing its lines on image2. It uses matches image coordinates to draw epilines"""
        # Undistort and normalize image coordinates
        pts_normalized_cur = self.undistort_and_normalize_image_coordinates(
            pts_cur_matched
        )
        pts_normalized_ref = self.undistort_and_normalize_image_coordinates(
            pts_ref_matched
        )
        # Calculate fundemental matrix
        F, mask = self.calculate_fundemental_matrix_using_essential(
            pts_normalized_cur, pts_normalized_ref
        )
        # Select only inlier points for plotting
        pts_cur_matched = pts_cur_matched[mask.ravel() == 1]
        pts_ref_matched = pts_ref_matched[mask.ravel() == 1]

        # Find epilines corresponding to points in left image (image1) and
        # drawing its lines on right image (image2)
        lines = cv2.computeCorrespondEpilines(pts_cur_matched.reshape(-1, 1, 2), 1, F)
        lines = lines.reshape(-1, 3)
        image_cur, image_ref = self.draw_lines(
            image_ref, image_cur, lines, pts_ref_matched, pts_cur_matched
        )
        return image_cur, image_ref
