
import matplotlib.pyplot as plt

def plot_horizontal_trajectory(pose_estimate, x_lim, y_lim, uwb_beacons):
    plt.suptitle("Horizontal trajectory")
    uwb_beacons = {7782220156096217088: [-0.545153  , -0.04282936,  1.49997155], 7782220156096217089: [-92.77930304,  -6.67807677,   0.91031529], 7782220156096217090: [-22.33896783, -22.76350421,   1.49969375], 7782220156096217091: [-55.43239422,  35.36695921,  -1.19977753], 7782220156096217092: [-82.20259541,   8.18208795,   0.91097657]}
    x_list = []
    y_list = []
    for p in uwb_beacons.values():
        x_list.append(p[0])
        y_list.append(p[1])

    plt.scatter(x_list, y_list)
    plt.plot(pose_estimate[:, 1], pose_estimate[:, 0], color="blue")
    plt.xlabel("y [m]")
    plt.ylabel("x [m]")
    plt.legend(["estimate", "ground truth"])
    #plt.xlim(x_lim)
    #plt.ylim(y_lim)
    plt.grid()


def plot_poses(pose_estimates):
    print(pose_estimates[:, 0], len(pose_estimates[:, 0]))
    plt.suptitle("X Y Z")
    plt.subplot(311)
    plt.plot([i for i in range(len(pose_estimates[:, 0]))], pose_estimates[:, 0])
    plt.grid()
    plt.ylabel("X")

    plt.subplot(312)
    plt.plot([i for i in range(len(pose_estimates[:, 0]))], pose_estimates[:, 1])
    plt.grid()
    plt.ylabel("Y")

    plt.subplot(313)
    plt.plot([i for i in range(len(pose_estimates[:, 0]))], pose_estimates[:, 2])
    plt.grid()
    plt.ylabel("Z")

    plt.tight_layout()