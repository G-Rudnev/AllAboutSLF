import math
from math import inf as inf
from math import pi as pi
from math import tan as tan
from math import sin as sin
from math import cos as cos
from math import atan as atan
from math import atan2 as atan2
from math import sqrt as sqrt

import numpy as np
from numpy.linalg import norm as norm

import threading
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from sklearn.cluster import KMeans, DBSCAN

N = 100

pntsXY = [] #теперь можно импортировать, память выделяется только в главном вызове
pntsPhi = []
pntsBuf = []

Nlines = 0
linesXY = []

deep = 5.0
cont = 0.6
half_dphi = 0.03
tol = 0.1

mess = 0.1
shape = 0

fig = []    #и плоттер запускается только в главном вызове
ax = []

mutex = threading.RLock()


from random import randint, uniform, normalvariate
def create_lidar_pnts(pnts: np.ndarray, pntsPhi: np.ndarray, pntsBuf: np.ndarray, N, mess=0.1, lidar_angle=(pi - 0.001), lidar_ax=(0.0, 0.0)) -> np.ndarray:
    """ генерация лидар-подобных данных для проверки решений """

    # TODO
    #  1. РЕАЛИЗОВАТЬ ВОЗМОЖНОСТЬ ГЕНЕРАЦИИ ЗАМКНУТОГО ОБЛАКА ТОЧЕК (ТО БИШЬ УГОЛ ОБЗОРА ЛИДАРА РАВЕН 360),
    #  РЕШИТЬ ВОПРОС С ЛУЧАМИ НА ОДНОЙ ПРЯМОЙ, НО В ПРОТИВОПОЛОЖНЫХ НАПРАВЛЕНИЯХ, Т.Е. КАК СГЕНЕРИРОВАТЬ ТОЧКИ НА НИХ
    #  2. ДОБАВИТЬ ЯВНЫЕ РАЗРЫВЫ, Т.Е. ПРОБЕЛЫ В ОБЛАКЕ, БУДТО ДАННЫЕ УТЕРЯНЫ ВОВСЕ

    lidar_ax_x = lidar_ax[0]
    lidar_ax_y = lidar_ax[1]
    R = 2  # радиус окружности, на которой выбираются центры окрестностей
    w = 1.5  # ширина окрестности (половина стороны квадрата)
    d_max = 4
    seg_num = randint(10, 20)  # число сегментов (стены и препятствия)
    segs = np.zeros([4, seg_num], dtype=float)
    c_x = R * cos((pi + lidar_angle) / 2)
    c_y = R * sin((pi + lidar_angle) / 2)

    st_x = c_x + uniform(-w, w)
    st_y = c_y + uniform(-w, w)
    for i in range(seg_num):
        c_x = R * cos((pi + lidar_angle) / 2 - ((i + 1) * lidar_angle / seg_num))
        c_y = R * sin((pi + lidar_angle) / 2 - ((i + 1) * lidar_angle / seg_num))

        end_x = c_x + uniform(-w, w)
        end_y = c_y + uniform(-w, w)

        k = (end_y - st_y) / (end_x - st_x)
        b = st_y - k * st_x

        segs[0, i] = k
        segs[1, i] = b
        if st_x < end_x:
            segs[2, i] = st_x
            segs[3, i] = end_x
        else:
            segs[2, i] = end_x
            segs[3, i] = st_x

        st_x = end_x
        st_y = end_y

    delta_angle = lidar_angle / (N - 1)  # период измерения расстояния при вращении
    # beam_angle = 0
    y_min = min(lidar_ax_y, d_max * sin((pi + lidar_angle) / 2))
    for i in range(N):
        dist_noise = normalvariate(0.0, mess / 15)
        beam_angle = (pi + lidar_angle) / 2 - i * delta_angle
        beam_k = tan(beam_angle)
        beam_b = lidar_ax_y - beam_k * lidar_ax_x
        d = inf
        for j in range(seg_num):
            intsec_x = (beam_b - segs[1, j]) / (segs[0, j] - beam_k)
            intsec_y = beam_k * (beam_b - segs[1, j]) / (segs[0, j] - beam_k) + beam_b
            if segs[2, j] <= intsec_x <= segs[3, j] and intsec_y > y_min:
                d_new = sqrt((intsec_x - lidar_ax_x) ** 2 + (intsec_y - lidar_ax_y) ** 2)
                if d_new < d:
                    d = d_new
                    nrst_intsec_x = intsec_x
                    nrst_intsec_y = intsec_y

        if math.isinf(d):
            pnts[0, i] = d_max * cos(beam_angle)
            pnts[1, i] = d_max * sin(beam_angle)
        else:
            pnts[0, i] = nrst_intsec_x  # Добавить шум
            pnts[1, i] = nrst_intsec_y  # Добавить шум

        pnts[0, i] = pnts[0, i] - dist_noise * sin(beam_angle - pi / 2)
        pnts[1, i] = pnts[1, i] + dist_noise * cos(beam_angle - pi / 2)
        pntsPhi[i] = atan2(pnts[1, i], pnts[0, i])

    pntsBuf = pnts[:2, :].copy()

    # data = pntsBuf.T
    # model.fit(data)
    # cl_fig, cl_ax = plt.subplots()
    # cl_ax.scatter(data[:, 0], data[:, 1], c=model.labels_)
    # cl_fig.show()

    # with open('lidar_data.csv', 'w', newline='') as csvfile:
    #     # fieldnames = ['lidar_angle', 'point_x', 'point_y']
    #     fieldnames = ['point_x', 'point_y']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    #     # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
    #     # writer.writeheader()
    #     for i in range(N):
    #         # writer.writerow({'lidar_angle': pnts_phi[i], 'point_x': pnts[0, i], 'point_y': pnts[1, i]})
    #         writer.writerow({'point_x': pnts[0, i], 'point_y': pnts[1, i]})

    return pntsBuf


def plot_clustering(pntsXY: np.ndarray):
    cl_fig, cl_ax = plt.subplots()
    # model = KMeans(n_clusters=10)
    model = DBSCAN(eps=0.1, min_samples=2)
    # data = np.zeros([N, 3])
    data = np.zeros([N, 2], dtype=float)
    data[:, :2] = pntsXY[:2, :].T
    # data[:, 2] = pntsPhi.T
    # data[:, 2] = np.asarray([i for i in range(N)], dtype=float).T
    model.fit(data)
    print(model.labels_)
    print(len(set(model.labels_)))
    cl_ax.scatter(data[:, 0], data[:, 1], s=12, c=model.labels_, cmap='rainbow')
    for i in range(N):
        cl_ax.plot([0.0, pntsXY[0, i]], [0.0, pntsXY[1, i]], color='cyan', linewidth=0.05)
    cl_ax.set_aspect('equal')
    cl_fig.show()


def drawLoad(xlim=(-4.5, 4.5), ylim=(-0.5, 3.5)):
    ax.cla()

    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect('equal')

    ax.scatter(pntsXY[0, 0], pntsXY[1, 0], s=7, marker='o', color='red')
    ax.scatter(pntsXY[0, 1:], pntsXY[1, 1:], s=7, marker='o', color='gray')

    for i in range(N):
        ax.plot([0.0, pntsXY[0, i]], [0.0, pntsXY[1, i]], color='cyan', linewidth=0.05)

    # linewidth = 2

    # ax.plot([0.0, linesXY[0, 0]], [0.0, linesXY[1, 0]], color='black', linewidth=linewidth)
    #
    # v = 0
    # while v < Nlines:
    #
    #     u = v
    #
    #     while (v < Nlines and (abs(linesXY[0, v]) > 0.01 or abs(linesXY[1, v]) > 0.01)):
    #         v += 1
    #
    #     if (v > u + 1):
    #         ax.plot(linesXY[0, u: v], linesXY[1, u: v], linewidth=linewidth)
    #
    #     if (v < Nlines):
    #         if (linesXY[0, v] != 0.0 and linesXY[1, v] != 0.0):
    #             ax.plot([linesXY[0, v - 1], linesXY[0, v + 1]], [linesXY[1, v - 1], linesXY[1, v + 1]], color='black',
    #                     linewidth=linewidth)
    #     else:
    #         ax.plot([linesXY[0, v - 1], 0.0], [linesXY[1, v - 1], 0.0], color='black', linewidth=linewidth)
    #
    #     v += 1

    fig.canvas.draw_idle()


def nextPnts(event):
    global Nlines, pntsBuf
    with mutex:

        pntsBuf = create_lidar_pnts(pntsXY, pntsPhi, pntsBuf, N)
        # Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)

        drawLoad()
        plot_clustering(pntsXY)


def updatePnts(val):
    global Nlines, mess, pntsBuf
    with mutex:
        mess = val

        pntsBuf = create_lidar_pnts(pntsXY, pntsPhi, pntsBuf, N)
        # Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
        drawLoad()


def updateLinesTolerance(val):
    global Nlines, tol
    with mutex:
        tol = val
        # Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)

    drawLoad(ax.get_xlim(), ax.get_ylim())
    plot_clustering(pntsXY)


def updatePntsShape(event):
    global Nlines, shape, pntsBuf
    with mutex:
        shape += 1
        if shape > 1:
            shape = 0
        pntsBuf = create_lidar_pnts(pntsXY, pntsPhi, pntsBuf, N)
        # Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
        drawLoad()


import numpy.random as np_random

jit = False
def lidar_jitter(event):
    """ функция, которая генерации дрожания и выбросов ВДОЛЬ ЛУЧА ЛИДАРА """
    global jit

    def foo():
        global Nlines
        rns = np.zeros([2, N])
        while jit and plt.get_fignums():
            with mutex:
                rns[:] = 0.0
                for i in range(N):
                    beam_angle = atan2(pntsXY[1, i], pntsXY[0, i])
                    if np_random.rand() > 0.9:
                        outlier = 0.1 * np_random.rand() - 0.05
                    else:
                        outlier = 0.02 * np_random.rand() - 0.01
                    rns[0, i] = - outlier * sin(beam_angle - pi / 2)
                    rns[1, i] = outlier * cos(beam_angle - pi / 2)

                pntsXY[:2, :N] = pntsBuf + rns
                pntsPhi[:N] = np.arctan2(pntsXY[1, :N], pntsXY[0, :N])
                # Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
                drawLoad(ax.get_xlim(), ax.get_ylim())
            time.sleep(0.1)

    with mutex:
        jit = not jit
        threading.Thread(target=foo).start()


def main():
    global Nlines, ax, fig
    global pntsXY, pntsPhi, pntsBuf, linesXY

    pntsXY = np.zeros([3, N])
    pntsPhi = np.zeros([N])
    pntsBuf = np.zeros([2, N])

    linesXY = np.zeros([3, N])
    linesXY[2, :] = 1.0

    fig = plt.figure()
    ax = plt.axes([0.07, 0.25, 0.45, 0.7])

    pntsBuf = create_lidar_pnts(pntsXY, pntsPhi, pntsBuf, N)

    # Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
    drawLoad()

    ax1 = plt.axes([0.15, 0.17, 0.45, 0.03])
    ax2 = plt.axes([0.15, 0.14, 0.45, 0.03])
    ax3 = plt.axes([0.55, 0.28, 0.1, 0.04])
    ax4 = plt.axes([0.55, 0.35, 0.1, 0.04])
    ax5 = plt.axes([0.55, 0.42, 0.1, 0.04])

    sz1 = Slider(ax1, 'tolerance', 0.0, 0.8, tol, valstep=0.02)
    sz1.on_changed(updateLinesTolerance)

    sz2 = Slider(ax2, 'mess', 0.0, 1.0, mess, valstep=0.02)
    sz2.on_changed(updatePnts)

    btn1 = Button(ax3, 'Jitter', hovercolor='0.975')
    btn1.on_clicked(lidar_jitter)

    btn2 = Button(ax4, 'Shape', hovercolor='0.975')
    btn2.on_clicked(updatePntsShape)

    btn3 = Button(ax5, 'Next', hovercolor='0.975')
    btn3.on_clicked(nextPnts)

    plot_clustering(pntsXY)

    plt.show()


if __name__ == "__main__":
    main()
