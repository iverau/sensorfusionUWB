import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

time_values = [250, 500, 1000, 2000, 3500, 5000]
values = [12.2, 12.0, 10.7, 9.8, 10.1, 10.7]

plt.plot(time_values, values)
plt.suptitle("Mean Yaw Error Given Number of Features")
plt.xlabel("Features")
plt.ylabel("Mean Yaw Error [deg]")
plt.show()
