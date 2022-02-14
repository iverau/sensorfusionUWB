import gtsam
import numpy as np

def gtsam_pose_from_result(gtsam_result):
    poses = gtsam.utilities.allPose3s(gtsam_result)
    keys = gtsam.KeyVector(poses.keys())

    positions, eulers = [], []
    for key in keys:
        if gtsam_result.exists(key):
            pose = gtsam_result.atPose3(key)
            pos, euler = gtsam_pose_to_numpy(pose)
            positions.append(pos)
            eulers.append(euler)
    positions = np.array(positions)
    eulers = np.array(eulers)
    return positions, eulers


def gtsam_landmark_from_results(gtsam_result, landmark_keys):

    poses = {}
    for key in landmark_keys:
        if gtsam_result.exists(key):
            pose = gtsam_result.atVector(key)
            poses[key] = pose
    return poses

def gtsam_pose_to_numpy(gtsam_pose):
    """Convert GTSAM pose to numpy arrays 
    (position, orientation)"""
    position = np.array([
        gtsam_pose.x(),
        gtsam_pose.y(),
        gtsam_pose.z()])
    euler = np.array([
        gtsam_pose.rotation().roll(),
        gtsam_pose.rotation().pitch(),
        gtsam_pose.rotation().yaw()])
    return position, euler

