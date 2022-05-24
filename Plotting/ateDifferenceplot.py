import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set()
gnss_variables = [0.33, 0.41, 1.0, 3.2, 7.01]
uwb_variables = [0.42, 0.43, 0.47, 1.75, 17.2]
time = [0, 5, 10, 20, 30]

plt.title("ATE for different amount of dropout")
plt.plot(time, gnss_variables)
plt.plot(time, uwb_variables)
plt.xlabel("Time [s]")
plt.ylabel("ATE [m]")
plt.legend(["GNSS", "UWB"])
plt.show()
