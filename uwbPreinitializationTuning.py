import numpy as np

# Prior noises
PRIOR_POSE_SIGMAS = np.array([1e-15, 1e-15, 1e-5, 1e-3, 1e-3, 1e-10]) # rx, ry, rz, x, y, z
PRIOR_VEL_SIGMAS = np.array([0.01, 0.01, 0.01])
PRIOR_BIAS_SIGMAS = np.array([5e-12, 5e-12, 5e-12, 5e-5, 5e-5, 5e-5])

# Prior states
DOWN_INITIAL_VALUE = -0.7
BIAS_INITIAL_VALUE =  np.array([0, 0, 0, -15e-3, -5e-3, -7e-3])

# UWB noises
UWB_NOISE = 0.2
UWB_PRIOR_POSITIONING_NOISE = 1e-32

# UWB-stage IMU tuning
VELOCITY_SIGMAS = np.array([0.1, 0.1, 1])