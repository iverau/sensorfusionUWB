
import matplotlib.pyplot as plt

def plot_horizontal_trajectory(pose_estimate, x_lim, y_lim, uwb_beacons):
    plt.suptitle("Horizontal trajectory")
    
    x_list = []
    y_list = []
    for p in uwb_beacons.values():
        x_list.append(p[0])
        y_list.append(p[1])
    #print(positions)
    plt.scatter(x_list, y_list)
    #print("Pose estimates:", pose_estimate)
    plt.plot(pose_estimate[:, 1], pose_estimate[:, 0], color="blue")
    #plt.plot(ground_truth[:, 1], ground_truth[:, 0], color="gray", linestyle="dashed")
    plt.xlabel("y [m]")
    plt.ylabel("x [m]")
    plt.legend(["estimate", "ground truth"])
    #plt.xlim(x_lim)
    #plt.ylim(y_lim)
    plt.grid()