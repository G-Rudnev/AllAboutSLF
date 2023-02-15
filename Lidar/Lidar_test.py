import sys
sys.path.append('./')
import numpy as np
import time
import threading
import Lidar
from Config import*
from RoboMath import*

import matplotlib.pyplot as plt

def foo(lid, ax, fig):

    time.sleep(0.2)

    obb_L2G = Transform2D_mat(0.2, 0.1, pi / 6, 1)
    obb_half_length = 0.6
    obb_half_width = 0.4

    obb_pnts = np.array([
        [obb_L2G[0, 2] + obb_L2G[0, 0] * obb_half_length + obb_L2G[0, 1] * obb_half_width, \
            obb_L2G[0, 2] + obb_L2G[0, 0] * obb_half_length - obb_L2G[0, 1] * obb_half_width, \
                obb_L2G[0, 2] - obb_L2G[0, 0] * obb_half_length - obb_L2G[0, 1] * obb_half_width, \
                    obb_L2G[0, 2] - obb_L2G[0, 0] * obb_half_length + obb_L2G[0, 1] * obb_half_width, \
                        obb_L2G[0, 2] + obb_L2G[0, 0] * obb_half_length + obb_L2G[0, 1] * obb_half_width], \
            \
        [obb_L2G[1, 2] + obb_L2G[1, 0] * obb_half_length + obb_L2G[1, 1] * obb_half_width, \
            obb_L2G[1, 2] + obb_L2G[1, 0] * obb_half_length - obb_L2G[1, 1] * obb_half_width, \
                obb_L2G[1, 2] - obb_L2G[1, 0] * obb_half_length - obb_L2G[1, 1] * obb_half_width, \
                    obb_L2G[1, 2] - obb_L2G[1, 0] * obb_half_length + obb_L2G[1, 1] * obb_half_width, \
                        obb_L2G[1, 2] + obb_L2G[1, 0] * obb_half_length + obb_L2G[1, 1] * obb_half_width] \
        ])

    # seg = np.array([[0.6, 10.0], [0.4, 0.8]])

    # T = 0.0
    # nT = 0

    while plt.get_fignums():

        Nlines = 0

        while Nlines == 0:
            Nlines = lid.GetLinesXY()
        
        # tt = time.time()  #контроль времени, можно убрать

        # intersected = Is_segment_intersects_lines(seg[:, 0], seg[:, 1], lid.linesXY, Nlines)
        intersected = Is_obb_intersects_lines(obb_L2G, obb_half_length, obb_half_width, lid.linesXY, Nlines)

        # T += (time.time() - tt)
        # nT += 1
        # if (nT == 100):
        #     print(f"Intersection time: {T / 100.0}")
        #     T = 0.0
        #     nT = 0

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.cla()

        ax.set(xlim = xlim, ylim = ylim)
        ax.set_aspect('equal')

        ax.plot([0.0, lid.linesXY[0, 0]], [0.0, lid.linesXY[1, 0]], color = 'black', linewidth = 3.0)

        v = 0
        while v < Nlines:
            
            u = v

            while (v < Nlines and (abs(lid.linesXY[0, v]) > 0.01 or abs(lid.linesXY[1, v]) > 0.01)):
                v += 1

            if (v > u + 1):
                ax.plot(lid.linesXY[0, u : v], lid.linesXY[1, u : v], linewidth = 3.0)

            if (v < Nlines): 
                if (lid.linesXY[0, v] != 0.0 and lid.linesXY[1, v] != 0.0):
                    ax.plot([lid.linesXY[0, v - 1], lid.linesXY[0, v + 1]], [lid.linesXY[1, v - 1], lid.linesXY[1, v + 1]], color = 'black', linewidth = 3.0)
            else:
                ax.plot([lid.linesXY[0, v - 1], 0.0], [lid.linesXY[1, v - 1], 0.0], color = 'black', linewidth = 3.0)

            v += 1

        if (intersected >= 0):
            # ax.plot(seg[0, :], seg[1, :], color = 'red', linewidth = 4.0)
            ax.plot(obb_pnts[0, :], obb_pnts[1, :], color = 'red', linewidth = 4.0)
        else:
            # ax.plot(seg[0, :], seg[1, :], color = 'blue', linewidth = 4.0)
            ax.plot(obb_pnts[0, :], obb_pnts[1, :], color = 'blue', linewidth = 4.0)

        fig.canvas.draw_idle()

def main():

    fig = plt.figure()
    ax = plt.axes([0.07, 0.25, 0.45, 0.7])
    ax.set(xlim = (-1, 5), ylim = (-5, 5))
    ax.set_aspect('equal')

    lid0 = Lidar.Lidar.Create(0)

    while not lid0.Start():
        time.sleep(1.0)

    threading.Thread(target=foo, args = (lid0, ax, fig)).start()

    plt.show()

    exit()

if __name__ == "__main__":
    main()