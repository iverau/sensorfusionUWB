import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d


def ATE(traj1, traj2):
    residualTraj = traj1 - traj2
    sum_value = 0
    for element, t in zip(residualTraj.as_coordinates().T, residualTraj.time):
        sum_value += np.inner(element, element)
    return np.sqrt(sum_value/len(residualTraj))


def absoluteError(ground_truth, estimate):
    residual = ground_truth - estimate
    # print(residual)
    return sum([abs(r) for r in residual])/len(residual)


def get_common_time_frame(time_gt, time_orb, resolution):
    return np.linspace(max(time_gt[0], time_orb[0] + 10), min(time_orb[-1] - 6, time_gt[-1]), resolution)


def interpolate_1D_arrays(value_gt, value_orb, time_gt, time_orb, resolution=1000):
    f1 = interp1d(time_gt, value_gt)
    f2 = interp1d(time_orb, value_orb)

    time_frame = get_common_time_frame(time_gt, time_orb, resolution)

    return f1(time_frame), f2(time_frame), time_frame - time_frame[0]


def plot_horizontal_trajectory(position_estimates, x_lim, y_lim, uwb_beacons, ground_truth):
    plt.suptitle("Horizontal trajectory")
    #uwb_beacons = {7782220156096217088: [-0.545153  , -0.04282936,  1.49997155], 7782220156096217089: [-92.77930304,  -6.67807677,   0.91031529], 7782220156096217090: [-22.33896783, -22.76350421,   1.49969375], 7782220156096217091: [-55.43239422,  35.36695921,  -1.19977753], 7782220156096217092: [-82.20259541,   8.18208795,   0.91097657]}
    x_list = []
    y_list = []
    print("UWB beacons", uwb_beacons)
    for p in uwb_beacons.values():
        x_list.append(p[1])
        y_list.append(p[0])

    plt.scatter(x_list, y_list)
    plt.plot(position_estimates[:, 1], position_estimates[:, 0], color="blue")
    plt.plot(ground_truth.gt_transelation[1],
             ground_truth.gt_transelation[0], color="red")

    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend(["Ground truth", "Estimates", "UWB beacons"])
    plt.grid()


def plot_horizontal_trajectory_old(position_estimates, x_lim, y_lim, landmark_variables):
    plt.suptitle("Horizontal trajectory")
    uwb_beacons = {7782220156096217088: [-0.545153, -0.04282936,  1.49997155], 7782220156096217089: [-92.77930304,  -6.67807677,   0.91031529], 7782220156096217090: [
        -22.33896783, -22.76350421,   1.49969375], 7782220156096217091: [-55.43239422,  35.36695921,  -1.19977753], 7782220156096217092: [-82.20259541,   8.18208795,   0.91097657]}
    x_list = []
    y_list = []
    for p in uwb_beacons.values():
        x_list.append(p[1])
        y_list.append(p[0])

    plt.scatter(x_list, y_list)
    plt.plot(position_estimates[:, 1], position_estimates[:, 0], color="blue")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend(["Ground truth", "Estimates", "UWB beacons"])
    plt.grid()


def find_index_closest(time_array, start_time):
    #temp_array = time_array - (time_array[0])
    return (np.abs(time_array - start_time)).argmin()


def convert_to_NED(ground_truth, position_estimates, time_steps):
    ned_positions = []
    for position, time in zip(position_estimates, time_steps):
        gt_index = find_index_closest(ground_truth.time, time)
        print("Index", gt_index, time, ground_truth.time[gt_index])
        gt_angels = ground_truth.gt_angels[gt_index, :]
        rotation = Rot.from_euler(
            "xyz", [gt_angels[0], gt_angels[1], gt_angels[2]])
        ned_positions.append(rotation.as_matrix() @ position.T)

    return np.array(ned_positions)


def plot_position(position_estimates, ground_truth, time_steps, convert_NED=False):
    time_steps = np.array(time_steps)
    time_steps[1:] -= time_steps[1] - time_steps[0]

    if convert_NED:
        position_estimates = convert_to_NED(
            ground_truth, position_estimates, time_steps)

    plt.suptitle("Positions")
    plt.subplot(311)
    plt.plot(time_steps, position_estimates[:, 0])
    plt.plot(ground_truth.time, ground_truth.gt_transelation[0, :])
    plt.legend(["Estimate", "Ground truth"])

    plt.grid()
    plt.ylabel("North [m]")

    plt.subplot(312)
    plt.plot(time_steps, position_estimates[:, 1])
    plt.plot(ground_truth.time, ground_truth.gt_transelation[1, :])
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("East [m]")

    plt.subplot(313)
    plt.plot(time_steps, position_estimates[:, 2])
    plt.plot(ground_truth.time, ground_truth.gt_transelation[2, :])
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Down [m]")

    plt.tight_layout()

# TODO: Finne ut om zyx eller xyz


def convert_to_body(ground_truth):
    body_pos = []
    for angel, position in zip(ground_truth.gt_angels, ground_truth.gt_transelation.T):
        rotation = Rot.from_euler("xyz", [angel[0], angel[1], angel[2]])
        body_pos.append(rotation.as_matrix().T @ position.T)
    return np.array(body_pos)


def plot_threedof2(position, euler_angels, ground_truth, time_steps):
    time_steps = np.array(time_steps)
    time_steps[1:] -= time_steps[1] - time_steps[0]
    time_steps -= time_steps[0] + 1

    gt_time = ground_truth.time
    #index = ground_truth.find_index_closest(ground_truth.time, 2*ground_truth.datasetSettings.gt_time_offset)
    #gt_time = gt_time[index:]
    gt_time -= gt_time[0]

    r2d = 180/np.pi

    gt, est, time = interpolate_1D_arrays(ground_truth.gt_transelation[0], position[:, 0], gt_time, time_steps)
    print("Error North:", absoluteError(gt, est))
    plt.suptitle("Pose Estimate and Ground Truth")
    plt.subplot(311)
    plt.plot(time, est)
    plt.plot(time, gt)
    plt.legend(["Estimate", "Ground truth"])

    plt.grid()
    plt.ylabel("North [m]")
    gt, est, time = interpolate_1D_arrays(ground_truth.gt_transelation[1], position[:, 1], gt_time, time_steps)
    print("Error East:", absoluteError(gt, est))

    plt.subplot(312)
    plt.plot(time, est)
    plt.plot(time, gt)
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("East [m]")

    gt, est, time = interpolate_1D_arrays(r2d * ground_truth.gt_angels[:, 2], r2d * euler_angels[:, 2], gt_time, time_steps)
    print("Error Yaw:", absoluteError(gt, est))

    plt.subplot(313)
    plt.plot(time, est)
    plt.plot(time, gt)
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Yaw [deg]")


def plot_threedof(position, euler_angels, ground_truth, time_steps):
    time_steps = np.array(time_steps)
    time_steps[1:] -= time_steps[1] - time_steps[0]
    time_steps -= time_steps[0]

    gt_time = ground_truth.time
    index = ground_truth.find_index_closest(ground_truth.time, 2*ground_truth.datasetSettings.gt_time_offset)
    gt_time = gt_time[index:]
    gt_time -= gt_time[0]

    r2d = 180/np.pi

    gt, est, time = interpolate_1D_arrays(ground_truth.gt_transelation[0, index:], position[:, 0], gt_time, time_steps)
    print("Error North:", absoluteError(gt, est))
    plt.suptitle("Pose")
    plt.subplot(311)
    plt.plot(time, est)
    plt.plot(time, gt)
    plt.legend(["Estimate", "Ground truth"])

    plt.grid()
    plt.ylabel("North [m]")
    gt, est, time = interpolate_1D_arrays(ground_truth.gt_transelation[1, index:], position[:, 1], gt_time, time_steps)
    print("Error East:", absoluteError(gt, est))

    plt.subplot(312)
    plt.plot(time, est)
    plt.plot(time, gt)
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("East [m]")

    gt, est, time = interpolate_1D_arrays(r2d * ground_truth.gt_angels[index:, 2], r2d * euler_angels[:, 2], gt_time, time_steps)
    print("Error Yaw:", absoluteError(gt, est))

    plt.subplot(313)
    plt.plot(time, est)
    plt.plot(time, gt)
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Yaw [deg]")


def plot_threedof_error(position, euler_angels, ground_truth, time_steps):
    time_steps = np.array(time_steps)
    time_steps[1:] -= time_steps[1] - time_steps[0]
    time_steps -= time_steps[0] + 1

    gt_time = ground_truth.time
    #index = ground_truth.find_index_closest(ground_truth.time, 2*ground_truth.datasetSettings.gt_time_offset)
    #gt_time = gt_time[index:]
    gt_time -= gt_time[0]

    r2d = 180/np.pi

    gt, est, time = interpolate_1D_arrays(ground_truth.gt_transelation[0], position[:, 0], gt_time, time_steps)
    gtx, estx, time = interpolate_1D_arrays(ground_truth.gt_transelation[0], position[:, 0], gt_time, time_steps)
    gty, esty, time = interpolate_1D_arrays(ground_truth.gt_transelation[1], position[:, 1], gt_time, time_steps)

    estimate_traj = np.array([estx, esty])
    gt_traj = np.array([gtx, gty])
    residual_traj = gt_traj - estimate_traj
    residual_traj = np.array([np.linalg.norm(element) for element in residual_traj.T])
    #print("Error North:", absoluteError(gt, est))
    est = abs(est - gt)
    plt.suptitle("Pose Error")
    plt.subplot(311)
    plt.plot(time, residual_traj)
    #plt.plot(time, gt)
    plt.legend(["Absolute error"])

    plt.grid()
    plt.ylabel("Absolute error [m]")
    #gt, est, time = interpolate_1D_arrays(ground_truth.gt_transelation[1], position[:, 1], gt_time, time_steps)
    #print("Error East:", absoluteError(gt, est))
    estx = abs(estx - gtx)
    esty = abs(esty - gty)

    plt.subplot(312)
    plt.plot(time, estx)
    plt.plot(time, esty)
    #plt.plot(time, gt)
    plt.legend(["Error in North", "Error in East"])
    plt.grid()
    plt.ylabel("Error [m]")

    gt, est, time = interpolate_1D_arrays(r2d * ground_truth.gt_angels[:, 2], r2d * euler_angels[:, 2], gt_time, time_steps)
    #print("Error Yaw:", absoluteError(gt, est))
    est = abs(est - gt)
    plt.subplot(313)
    plt.plot(time, est)
    #plt.plot(time, gt)
    plt.legend(["Error in Yaw"])
    plt.grid()
    plt.ylabel("Error [deg]")


def ATE(position, ground_truth, time_steps):
    time_steps = np.array(time_steps)
    time_steps[1:] -= time_steps[1] - time_steps[0]
    time_steps -= time_steps[0] + 1

    gt_time = ground_truth.time
    #index = ground_truth.find_index_closest(ground_truth.time, 2*ground_truth.datasetSettings.gt_time_offset)
    #gt_time = gt_time[index:]
    gt_time -= gt_time[0]
    gtx, estx, time = interpolate_1D_arrays(ground_truth.gt_transelation[0], position[:, 0], gt_time, time_steps)
    gty, esty, time = interpolate_1D_arrays(ground_truth.gt_transelation[1], position[:, 1], gt_time, time_steps)

    estimate_traj = np.array([estx, esty])
    gt_traj = np.array([gtx, gty])

    residual_traj = estimate_traj - gt_traj
    sum_value = 0
    for element in residual_traj.T:
        sum_value += np.inner(element, element)
    return np.sqrt(sum_value/len(residual_traj.T))


def new_xy_plot(position, euler_angels, ground_truth, time_steps):
    time_steps = np.array(time_steps)
    time_steps[1:] -= time_steps[1] - time_steps[0]
    time_steps -= time_steps[0] + 1

    gt_time = ground_truth.time
    #index = ground_truth.find_index_closest(ground_truth.time, 2*ground_truth.datasetSettings.gt_time_offset)
    #gt_time = gt_time[index:]
    gt_time -= gt_time[0]
    gtx, estx, time = interpolate_1D_arrays(ground_truth.gt_transelation[0], position[:, 0], gt_time, time_steps)
    gty, esty, time = interpolate_1D_arrays(ground_truth.gt_transelation[1], position[:, 1], gt_time, time_steps)

    plt.suptitle("Horizontal Trajectory")
    plt.plot(estx, esty)
    plt.plot(gtx, gty)
    plt.legend(["Estimate", "Ground truth"])


def plot_angels(euler_angels, ground_truth, time_steps):
    time_steps = np.array(time_steps)
    time_steps[1:] -= time_steps[1] - time_steps[0]
    #
    # time_steps[:]
    r2d = 180/np.pi

    plt.suptitle("Angels")
    plt.subplot(311)
    plt.plot(time_steps,  r2d * euler_angels[:, 0])
    plt.plot(ground_truth.time, r2d * ground_truth.gt_angels[:, 0])
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Roll [deg]")

    plt.subplot(312)
    plt.plot(time_steps, r2d * euler_angels[:, 1])
    plt.plot(ground_truth.time, r2d * ground_truth.gt_angels[:, 1])
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Pitch [deg]")

    plt.subplot(313)
    plt.plot(time_steps, r2d * euler_angels[:, 2])
    plt.plot(ground_truth.time, r2d * ground_truth.gt_angels[:, 2])
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Yaw [deg]")

    plt.tight_layout()


def plot_bias(bias):
    plt.suptitle("Biases")
    plt.subplot(311)
    plt.plot(range(len(bias)), bias[:, 0])
    plt.ylabel("North bias")
    plt.subplot(312)
    plt.plot(range(len(bias)), bias[:, 1])
    plt.ylabel("East bias")
    plt.subplot(313)
    plt.plot(range(len(bias)), bias[:, 2])
    plt.ylabel("Down bias")


def plot_vel(velocities, time_steps, ground_truth):
    time_steps = np.array(time_steps)
    time_steps[1:] -= time_steps[1] - time_steps[0]
    plt.suptitle("Velocities")
    plt.subplot(311)
    plt.plot(time_steps, velocities[:, 0])
    plt.plot(ground_truth.time, ground_truth.v_north)
    plt.ylabel("North velocity")
    plt.subplot(312)
    plt.plot(time_steps, velocities[:, 1])
    plt.plot(ground_truth.time, ground_truth.v_east)

    plt.ylabel("East velocity")
    plt.subplot(313)
    plt.plot(time_steps, velocities[:, 2])
    plt.plot(ground_truth.time, ground_truth.v_down)

    plt.ylabel("Down velocity")
