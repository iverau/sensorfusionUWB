import numpy as np

# Prior noises
PRIOR_POSE_SIGMAS = np.array([1e-15, 1e-15, 1e-5, 1e-3, 1e-3, 1e-10]) # rx, ry, rz, x, y, z
PRIOR_VEL_SIGMAS = np.array([0.0001, 0.0001, 0.0001])
PRIOR_BIAS_SIGMAS = np.array([5e-12, 5e-12, 5e-12, 5e-7, 5e-7, 5e-7])

# Prior states
DOWN_INITIAL_VALUE = -0.7
BIAS_INITIAL_VALUE =  np.array([0, 0, 0, -15e-3, -5e-3, -7e-3])

# UWB noises
UWB_NOISE = 0.3
UWB_PRIOR_POSITIONING_NOISE = 1e-32

# UWB-stage IMU tuning
VELOCITY_SIGMAS = np.array([0.5, 0.5, 0.01]) # It is scalled with number of measurements
POSE_SIGMAS = np.array([0.000175, 0.000175, 0.01, 1, 1, 0.001]) # It is scalled with number of measurements
# pre-integration parameters can be found in the file Sensors/Imu.py

#GNSS tuning
GNSS_PREINIT_ENABLED = True
GNSS_NOISE = np.array([0.8, 0.8, 0.5, 4, 4, 5])
GNSS_VELOCITY_SIGMAS = np.array([0.1, 0.1, 0.01])

#Other constants
NUMBER_OF_RUNNING_ITERATIONS = 2000 # Full traj is about 3000