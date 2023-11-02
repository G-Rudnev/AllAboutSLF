import matplotlib.pyplot as plt
import time
import threading

from RealSense import *
from RoboMath import *
from Config import *

def plot(cam: RealSense):
    while True:
        plot_output(cam)
        time.sleep(0.1)


def plot_output(cam: RealSense):

    if cam._is_visual:
        for ln in cam.ax:
            for cl in ln:
                cl.clear()

        cam.ax[0][0].imshow(cam._color_image)
        cam.ax[0][1].imshow(cam._depth_image, cmap='jet', norm=Normalize(0, 65535 * 0.075))

        for a in cam.ax[1]:
            a.scatter(0, 0, s=10, c='black')

        # ax[1, 0].scatter(angle_sorted_projected_points[0, :], angle_sorted_projected_points[1, :], s=1, color='gray')
        cam.ax[1][1].scatter(cam._xy[0, :], cam._xy[1, :], s=1, color='gray')

        # Рисование ломаной

        linewidth = 1.5
        v = 0
        while v < cam._Nlines[0]:
            
            u = v

            while (v < cam._Nlines[0] and (abs(cam._linesXY[0, v]) > 1e-5 or abs(cam._linesXY[1, v]) > 1e-5)):
                v += 1

            if (v > u + 1):
                cam.ax[1][1].plot(cam._linesXY[0, u : v], cam._linesXY[1, u : v], linewidth = linewidth, color = 'red')

            if (v < cam._Nlines[0] - 1): 
                if (cam._linesXY[0, v] != 0.0 and cam._linesXY[1, v] != 0.0):
                    cam.ax[1][1].plot([cam._linesXY[0, v - 1], cam._linesXY[0, v + 1]], [cam._linesXY[1, v - 1], cam._linesXY[1, v + 1]], color = 'black', linewidth = linewidth)
            else:
                cam.ax[1][1].plot([cam._linesXY[0, cam._Nlines[0] - 1], cam._linesXY[0, 0]], [cam._linesXY[1, cam._Nlines[0] - 1], cam._linesXY[1, 0]], color = 'red', linewidth = linewidth)

            v += 1

    else:
        cam.ax.clear()

        cam.ax.set(xlim = (-1, cam._trimming_pars[1]), ylim = (-cam._trimming_pars[1], cam._trimming_pars[1]))
        cam.ax.set_aspect('equal')

        cam.ax.scatter(0, 0, s=10, c='black')
        cam.ax.scatter(cam._xy[0, :], cam._xy[1, :], s=1, color='gray')

        # Рисование ломаной
        linewidth = 1.5
        v = 0
        while v < cam._Nlines[0]:
            
            u = v

            while (v < cam._Nlines[0] and (abs(cam._linesXY[0, v]) > 1e-5 or abs(cam._linesXY[1, v]) > 1e-5)):
                v += 1
            if (v > u + 1):
                cam.ax.plot(cam._linesXY[0, u : v], cam._linesXY[1, u : v], linewidth = linewidth, color = 'red')

            if (v < cam._Nlines[0] - 1): 
                if (cam._linesXY[0, v] != 0.0 and cam._linesXY[1, v] != 0.0):
                    cam.ax.plot([cam._linesXY[0, v - 1], cam._linesXY[0, v + 1]], [cam._linesXY[1, v - 1], cam._linesXY[1, v + 1]], color = 'black', linewidth = linewidth)
            else:
                cam.ax.plot([cam._linesXY[0, cam._Nlines[0] - 1], cam._linesXY[0, 0]], [cam._linesXY[1, cam._Nlines[0] - 1], cam._linesXY[1, 0]], color = 'red', linewidth = linewidth)

            v += 1

    cam.fig.canvas.draw_idle()



def main():
    RS_ID = 0
    vis = False
    do_plot = False
    cam0 = RealSense.Create(RS_ID, vis=vis, do_plot=do_plot)
    # while not cam0.Start():
    #     time.sleep(0.1)

    polyline = Polyline(cam0.N)

    if cam0._do_plot:
        threading.Thread(target=plot, args=(cam0,)).start()
        while not cam0.Start():
            time.sleep(0.1)
        plt.show()
    else:
        while True:
            while not cam0.Start():
                time.sleep(0.1)
            print(cam0.GetLinesXY(polyline))
            time.sleep(3.0)
            cam0.Stop()
            time.sleep(2.0)

    exit()

if __name__ == '__main__':
    main()