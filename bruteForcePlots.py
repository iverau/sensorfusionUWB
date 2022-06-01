import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def extractData(number):
    string = f"vo__gnss_loss_{number}.pickle"

    with open(string, "rb") as f:
        position, angle, time = pickle.load(f)
        position = position[:-100]
        angle = angle[:-100]
        time = time[:-100]

    return position, angle, time


position_0, angle_0, time_0 = extractData(0)
position_5, angle_5, time_5 = extractData(5)
position_10, angle_10, time_10 = extractData(10)
position_20, angle_20, time_20 = extractData(20)
position_30, angle_30, time_30 = extractData(30)
status = "deg"

plt.suptitle("Heading Error")
plt.subplot(511)
plt.plot(time_0,  angle_0)
plt.legend(["Error with 0 seconds"])
plt.grid()
plt.ylabel(f"Error [{status}]")

plt.subplot(512)
plt.plot(time_5,  angle_5)
plt.legend(["Error with 5 seconds"])
plt.axvspan(20, 25, color='orange', alpha=0.5)

plt.grid()
plt.ylabel(f"Error [{status}]")

plt.subplot(513)
plt.plot(time_10,  angle_10)
plt.legend(["Error with 10 seconds"])
plt.axvspan(20, 30, color='orange', alpha=0.5)

plt.grid()
plt.ylabel(f"Error [{status}]")

plt.subplot(514)
plt.plot(time_20,  angle_20)
plt.legend(["Error with 20 seconds"])
plt.axvspan(20, 40, color='orange', alpha=0.5)

plt.grid()
plt.ylabel(f"Error [{status}]")

plt.subplot(515)
plt.plot(time_30,  angle_30)
plt.legend(["Error with 30 seconds"])
plt.axvspan(20, 50, color='orange', alpha=0.5)

plt.grid()
plt.ylabel(f"Error [{status}]")
plt.xlabel("Time [s]")

plt.show()
