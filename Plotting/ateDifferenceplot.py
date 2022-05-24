import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set()
gnss_variables = [0.62, 0.55, 0.77, 1.13, 2.4]
#uwb_variables = [0.42, 0.43, 0.47, 1.75, 17.2]
time = [0, 2, 5, 10, 20]

plt.title("ATE For Reduced Framerate")
plt.plot(time, gnss_variables)
#plt.plot(time, uwb_variables)
plt.xlabel("Frames dropped")
plt.ylabel("ATE [m]")
plt.legend(["ATE"])
plt.show()
