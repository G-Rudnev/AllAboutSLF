import math as m
import numpy as np
import numpy.random as random

import threading
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

N = 50
Nlines = 0
pnts = np.zeros([3, N])
pntsBuf = np.zeros([2, N])
lines = np.zeros([4, N - 1])

tol = 0.1
mess = 0.1
shape = 0
mode = 'avg'  # метод аппроксимации
timing = np.zeros(2, dtype=float)  # значения времени исполнения аппроксимации avg и lsm

fig = plt.figure()
ax = plt.axes([0.07, 0.25, 0.45, 0.7])

mutex = threading.RLock()


def lineApproxAveraging(pnts : np.ndarray, fr : int, to : int):

    b0 = np.exp(np.linspace(-0.25, 1 - 0.25, (to - fr))**2 / -0.03)
    # b0 = np.linspace(1.0, 0.0, N)**2.0
    b0 /= np.sum(b0)

    b1 = np.flip(b0)

    pnt0 = [np.sum(b0 * pnts[0, fr : to]), np.sum(b0 * pnts[1, fr : to])]
    pnt1 = [np.sum(b1 * pnts[0, fr : to]), np.sum(b1 * pnts[1, fr : to])]

    a = (pnt1[1] - pnt0[1]) / (pnt1[0] - pnt0[0])
    b = pnt0[1] - a * pnt0[0]
    return (a, b, np.mean(((a * pnts[0, fr : to] - pnts[1, fr : to] + b)**2) / (b**2 + 1)))


# ====== РЕАЛИЗАЦИЯ ФУНКЦИЙ МНК С ИСПОЛЬЗОВАНИЕМ NUMPY =======

def calc_loss_func(seg_pnts: np.ndarray, A: float, C: float):
    loss_func_value = 0
    for i in range(seg_pnts.shape[1]):
        # Расстояние до текущей прямой
        dist = abs(seg_pnts[0, i] * A - seg_pnts[1, i] + C) / m.sqrt(A ** 2 + 1)
        loss_func_value += dist ** 2
    return loss_func_value


def calc_sums(seg_pnts: np.ndarray):
    x_sum = np.sum(seg_pnts[0])  # сумма всех х координат
    y_sum = np.sum(seg_pnts[1])  # сумма всех у координат
    xy_sum = np.dot(seg_pnts[0], seg_pnts[1].T)  # сумма произведений всех х и всех у координат соответственно
    x_sq_sum = np.dot(seg_pnts[0], seg_pnts[0].T)  # сумма квадратов всех координат х
    y_sq_sum = np.dot(seg_pnts[1], seg_pnts[1].T)  # сумма квадратов всех координат у

    return x_sum, y_sum, xy_sum, x_sq_sum, y_sq_sum


def lineApproxLSM(pnts: np.ndarray, fr: int, to: int):
    # data_x, data_y = segment_data
    pts_num = to - fr
    x_sum, y_sum, xy_sum, x_sq_sum, y_sq_sum = calc_sums(pnts[:2, fr:to])
    # Вычисление A для минимумов функции потерь
    phi = xy_sum - x_sum * y_sum / pts_num
    theta = (x_sq_sum - y_sq_sum) / phi + (y_sum ** 2 - x_sum ** 2) / (pts_num * phi)
    D = theta ** 2 + 4  # дискриминант
    A1 = (-theta + m.sqrt(D)) / 2
    A2 = (-theta - m.sqrt(D)) / 2
    # Вычисление С для минимумов функции потерь
    C1 = (y_sum - x_sum * A1) / pts_num
    C2 = (y_sum - x_sum * A2) / pts_num
    # Подстановка в функцию потерь, выявление лучшего
    lf1 = calc_loss_func(pnts[:2, fr:to], A1, C1)
    lf2 = calc_loss_func(pnts[:2, fr:to], A2, C2)
    # Выбор наименьшего значения функции потерь, возврат соответствующих ему параметров А и С
    if lf1 < lf2:
        return A1, C1, np.mean(((A1 * pnts[0, fr:to] - pnts[1, fr:to] + C1) ** 2) / (A1 ** 2 + 1))
    else:
        return A2, C2, np.mean(((A2 * pnts[0, fr:to] - pnts[1, fr:to] + C2) ** 2) / (A2 ** 2 + 1))

# =====================================================================


def firstPnt(pnts : np.ndarray) -> None:
    pnts[0, 0] = 0.5 * random.rand() - 0.25
    pnts[1, 0] = 0.5 * random.rand() - 0.25
    pnts[2, 0] = 2.0 * m.pi * random.rand() - m.pi

def createPnts(pnts : np.ndarray, N, d0 = 0.1, shape = 0, mess = 0.1) -> None:
    global pntsBuf

    i_ang = 0
    deltaAng = 0.2 * random.rand() - 0.

    for i in range(1, N):
        d = d0 * (1 + random.randn() * mess)
        pnts[0, i] = pnts[0, i - 1] + d * m.cos(pnts[2, i - 1])
        pnts[1, i] = pnts[1, i - 1] + d * m.sin(pnts[2, i - 1])

        if (shape == 0):    #polyline
            if (random.rand() > 1 - 5.0 / N): # 5 fractures in average
                pnts[2, i] = pnts[2, i - 1] + m.pi * random.rand() - m.pi/2
                i_ang = i
            else:
                pnts[2, i] = pnts[2, i_ang] * (1 + random.randn() * mess)
        elif (shape == 1):  #circle
            pnts[2, i] = pnts[2, i - 1] + deltaAng

    pntsBuf = pnts[:2, :].copy()


def getLines(lines: np.ndarray, pnts: np.ndarray, Npnts, tolerance=0.1, mode='avg') -> int:
    """#returns the number of the gotten lines in lines"""
    t0 = time.time()
    global Nlines

    global timing

    line = np.zeros([4])
    pcross = np.array([0.0, 0.0])

    i = 1
    Nlines = 0

    while i < Npnts:
        gap = tolerance
        i0 = i

        while True:

            line[0] = (pnts[1, i] - pnts[1, i - 1]) / (pnts[0, i] - pnts[0, i - 1])
            line[1] = pnts[1, i] - line[0] * pnts[0, i]
            byNpnts = 2

            while True:
                
                i += 1

                if (i < N and abs(line[0] * pnts[0, i] - pnts[1, i] + line[1]) / m.sqrt(line[0]**2 + 1) < gap):
                    if (not byNpnts % 2):
                        line[0] = (pnts[1, i - byNpnts // 2] - pnts[1, i - byNpnts]) / (pnts[0, i - byNpnts // 2] - pnts[0, i - byNpnts])
                        line[1] = pnts[1, i - byNpnts] - line[0] * pnts[0, i - byNpnts]
                    byNpnts += 1
                else:
                    # аппроксимация в зависимости от режима mode (avg - усреднение, lsm - МНК)
                    if mode == 'avg':
                        (line[0], line[1], q0) = lineApproxAveraging(pnts, i - byNpnts, i)
                    elif mode == 'lsm':
                        (line[0], line[1], q0) = lineApproxLSM(pnts, i - byNpnts, i)

                    while (q0 > 0.0001):
                        if mode == 'avg':
                            (line_0, line_1, q) = lineApproxAveraging(pnts, i - byNpnts, i - 1)
                        elif mode == 'lsm':
                            (line_0, line_1, q) = lineApproxLSM(pnts, i - byNpnts, i - 1)
                        if (q > q0):
                            break
                        else:
                            i -= 1
                            byNpnts -= 1
                            line[0] = line_0
                            line[1] = line_1
                            q0 = q

                    if (Nlines > 0):
                        pcross[0] = (line[1] - lines[1, Nlines - 1]) / (lines[0, Nlines - 1] - line[0])
                        pcross[1] = line[0] * pcross[0] + line[1]

                        if (np.linalg.norm(pnts[:2, i - byNpnts] - pcross) > tolerance or m.isnan(pcross[0]) or m.isinf(pcross[0])):
                            if (byNpnts <= 2):
                                pcross[0] = (pnts[0, i - 2] + lines[0, Nlines - 1] * pnts[1, i - 2] - lines[0, Nlines - 1] * lines[1, Nlines - 1]) / (lines[0, Nlines - 1]**2 + 1)
                                pcross[1] = lines[0, Nlines - 1] * pcross[0] + lines[1, Nlines - 1]

                                line[0] = (pnts[1, i - 1] - pcross[1]) / (pnts[0, i - 1] - pcross[0])
                                line[1] = pcross[1] - line[0] * pcross[0]
                                lines[3, Nlines - 1] = pcross[0]
                                line[2] = pcross[0]
                            else:
                                i = i0
                                gap *= 0.75
                                break
                        else:
                            lines[3, Nlines - 1] = pcross[0]
                            line[2] = pcross[0]

                    else:
                        line[2] = (pnts[0, 0] + line[0] * pnts[1, 0] - line[0] * line[1]) / (line[0]**2 + 1)

                    if (i > N - 1):
                        line[3] = (pnts[0, N - 1] + line[0] * pnts[1, N - 1] - line[0] * line[1]) / (line[0]**2 + 1)

                    break

            if (i > i0):
                break
            else:
                continue

        lines[:, Nlines] = line
        Nlines += 1

    t1 = time.time()

    # сохранение значений времени выполнения аппроксимации всей траектории
    if mode == 'avg':
        timing[0] = t1 - t0
    elif mode == 'lsm':
        timing[1] = t1 - t0

    return Nlines


def drawLoad(xlim=(-4, 4), ylim=(-4, 4)):

    ax.cla()

    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect('equal')

    ax.scatter(pnts[0, 0], pnts[1, 0], s=20, marker='o', color='red')
    ax.scatter(pnts[0, 1:], pnts[1, 1:], s=20, marker='o', color='gray')
    ax.set(title=mode)
    ax.axis('scaled')

    for i in range(Nlines):
        ax.plot([lines[2, i], lines[3, i]], [lines[0, i] * lines[2, i] + lines[1, i], lines[0, i] * lines[3, i] + lines[1, i]], linewidth=3)
    
    fig.canvas.draw_idle()


def nextPnts(event):
    with mutex:
        firstPnt(pnts)

        createPnts(pnts, N, shape=shape, mess=mess)
        getLines(lines, pnts, N, tol, mode=mode)

        drawLoad()


def updatePnts(val):
    global mess
    with mutex:
        mess = val
        createPnts(pnts, N, shape=shape, mess=mess)
        getLines(lines, pnts, N, tol, mode=mode)
        drawLoad()


def updateLinesTolerance(val):
    global tol
    
    with mutex:
        tol = val
        getLines(lines, pnts, N, tol, mode=mode)
    
    drawLoad(ax.get_xlim(), ax.get_ylim())


def updatePntsShape(event):
    global shape
    with mutex:
        shape += 1
        if shape > 1:
            shape = 0
        createPnts(pnts, N, shape=shape, mess=mess)
        getLines(lines, pnts, N, tol, mode=mode)
        drawLoad()


jit = False
def jitter(event):
    global jit
    
    def foo():
        while jit and plt.get_fignums():
            with mutex:
                rns = np.zeros([2, N])
                for i in range(N):
                    if random.rand() > 0.9:
                        rns[:, i] += 0.5 * random.rand(2) - 0.25
                pnts[:2, :] = pntsBuf + 0.02 * random.rand(2, N) - 0.01 + rns

                getLines(lines, pnts, N, tol, mode=mode)
                drawLoad(ax.get_xlim(), ax.get_ylim())
            time.sleep(0.5)

    with mutex:
        jit = not jit
        threading.Thread(target=foo).start()


# Смена метода аппроксимации: avg - усреднение, lsm - МНК
def change_mode(event):
    global mode
    with mutex:
        if mode == 'avg':
            mode = 'lsm'
        elif mode == 'lsm':
            mode = 'avg'
        getLines(lines, pnts, N, tol, mode=mode)
        drawLoad()


# Вывод графиков времени аппроксимации каждым алгоритмом
def plot_timing_cumsum(event):
    tm_fig, tm_ax = plt.subplots()
    t_avg_cumsum = 0  # кумулятивная сумма для усреднения
    t_lsm_cumsum = 0  # кумулятивная сумма для МНК
    n_samples = 10  # число различных наборов точек
    n_tests = 10  # число запусков на одном наборе
    with mutex:
        for i in range(n_samples):
            nextPnts(None)
            for j in range(n_tests):
                # дважды меняем метод аппроксимации
                change_mode(None)
                change_mode(None)
                tm_ax.plot([j + n_tests * i, j + n_tests * i + 1],
                           [t_avg_cumsum, t_avg_cumsum + timing[0]],
                           color='red')
                tm_ax.plot([j + n_tests * i, j + n_tests * i + 1],
                           [t_lsm_cumsum, t_lsm_cumsum + timing[1]],
                           color='blue')
                t_avg_cumsum += timing[0]
                t_lsm_cumsum += timing[1]

        tm_fig.show()


def main():

    firstPnt(pnts)
    createPnts(pnts, N, shape=shape, mess=mess)

    getLines(lines, pnts, N, tol, mode=mode)
    drawLoad()

    ax1 = plt.axes([0.15, 0.17, 0.45, 0.03])
    ax2 = plt.axes([0.15, 0.14, 0.45, 0.03])
    ax3 = plt.axes([0.55, 0.28, 0.1, 0.04])
    ax4 = plt.axes([0.55, 0.35, 0.1, 0.04])
    ax5 = plt.axes([0.55, 0.42, 0.1, 0.04])
    ax6 = plt.axes([0.55, 0.49, 0.1, 0.04])
    ax7 = plt.axes([0.55, 0.6, 0.1, 0.1])

    sz1 = Slider(ax1, 'tolerance', 0.0, 0.8, tol, valstep=0.02)
    sz1.on_changed(updateLinesTolerance)

    sz2 = Slider(ax2, 'mess', 0.0, 1.0, mess, valstep=0.02)
    sz2.on_changed(updatePnts)

    btn1 = Button(ax3, 'Jitter', hovercolor='0.975')
    btn1.on_clicked(jitter)

    btn2 = Button(ax4, 'Shape', hovercolor='0.975')
    btn2.on_clicked(updatePntsShape)

    btn3 = Button(ax5, 'Next', hovercolor='0.975')
    btn3.on_clicked(nextPnts)

    btn4 = Button(ax6, 'Mode', hovercolor='0.975')
    btn4.on_clicked(change_mode)

    btn5 = Button(ax7, 'Start\ntest', hovercolor='0.975')
    btn5.on_clicked(plot_timing_cumsum)

    plt.show()


if __name__ == "__main__":
    main()
