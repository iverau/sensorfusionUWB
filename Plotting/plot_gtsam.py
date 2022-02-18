import matplotlib.pyplot as plt
import numpy as np

def plot_horizontal_trajectory(position_estimates, x_lim, y_lim, uwb_beacons, ground_truth):
    plt.suptitle("Horizontal trajectory")
    uwb_beacons = {7782220156096217088: [-0.545153  , -0.04282936,  1.49997155], 7782220156096217089: [-92.77930304,  -6.67807677,   0.91031529], 7782220156096217090: [-22.33896783, -22.76350421,   1.49969375], 7782220156096217091: [-55.43239422,  35.36695921,  -1.19977753], 7782220156096217092: [-82.20259541,   8.18208795,   0.91097657]}
    x_list = []
    y_list = []
    for p in uwb_beacons.values():
        x_list.append(p[0])
        y_list.append(p[1])

    plt.scatter(x_list, y_list)
    plt.plot(ground_truth.gt_transelation[0], ground_truth.gt_transelation[1], color="red")
    plt.plot(position_estimates[:, 1], position_estimates[:, 0], color="blue")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend(["Ground truth", "Estimates", "UWB beacons"])
    plt.grid()

def plot_horizontal_trajectory_old(position_estimates, x_lim, y_lim, landmark_variables):
    plt.suptitle("Horizontal trajectory")
    uwb_beacons = {7782220156096217088: [-0.545153  , -0.04282936,  1.49997155], 7782220156096217089: [-92.77930304,  -6.67807677,   0.91031529], 7782220156096217090: [-22.33896783, -22.76350421,   1.49969375], 7782220156096217091: [-55.43239422,  35.36695921,  -1.19977753], 7782220156096217092: [-82.20259541,   8.18208795,   0.91097657]}
    x_list = []
    y_list = []
    for p in uwb_beacons.values():
        x_list.append(p[0])
        y_list.append(p[1])

    plt.scatter(x_list, y_list)
    plt.plot(position_estimates[:, 1], position_estimates[:, 0], color="blue")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend(["Ground truth", "Estimates", "UWB beacons"])
    plt.grid()

def plot_position(position_estimates, ground_truth, time_steps):
    time_steps[1:] -= time_steps[1] - time_steps[0]

    plt.suptitle("Positions")
    plt.subplot(311)
    plt.plot(time_steps, position_estimates[:, 0])
    plt.plot(ground_truth.time  , ground_truth.gt_transelation[0, :])
    plt.legend(["Estimate", "Ground truth"])

    plt.grid()
    plt.ylabel("X [m]")

    plt.subplot(312)
    plt.plot(time_steps, position_estimates[:, 1])
    plt.plot(ground_truth.time  , ground_truth.gt_transelation[1, :])
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Y [m]")

    plt.subplot(313)
    plt.plot(time_steps, position_estimates[:, 2])
    plt.plot(ground_truth.time  , ground_truth.gt_transelation[2, :])
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Z [m]")

    plt.tight_layout()


def plot_angels(euler_angels, ground_truth, time_steps):
    time_steps[1:] -= time_steps[1] - time_steps[0]
    r2d = 180/np.pi

    plt.suptitle("Angels")
    plt.subplot(311)
    plt.plot(time_steps, r2d * euler_angels[:, 0])
    plt.plot(ground_truth.time  , r2d * ground_truth.gt_angels[:, 0])
    plt.legend(["Estimate", "Ground truth"])
    plt.grid()
    plt.ylabel("Roll [deg]")

    plt.subplot(312)
    plt.plot(time_steps, r2d * euler_angels[:, 1])
    plt.plot(ground_truth.time  , r2d * ground_truth.gt_angels[:, 1])
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

    