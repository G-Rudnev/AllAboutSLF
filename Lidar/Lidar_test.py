import sys
sys.path.append('./')
import numpy as np
import time
from Lidar import*
from Config import*

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes([0.07, 0.25, 0.45, 0.7])
ax.set(xlim = (-5, 5), ylim = (-5, 5))
ax.set_aspect('equal')

lid0 = Lidar.Create(mainDevicesPars['lidarSN_0'], ax, fig)

while not lid0.Start():
    time.sleep(1.0)

plt.show()

# time.sleep(0.5)

# try:

#     while 1:
#         # lid0.acquireXY()
#         # # dx_dy_alpha += lid0.dx_dy_alpha
#         # print(lid0.xy[:2, :2])
#         # lid0.releaseXY()
#         time.sleep(0.15)
# except:
#     lid0.Stop()

exit()