import math
from math import inf as inf
from math import pi as pi
from math import tan as tan
from math import sin as sin
from math import cos as cos
from math import atan as atan
from math import atan2 as atan2

import numpy as np
from numpy.linalg import norm as norm 
import numpy.random as random

import threading
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

N = 50

pnts = np.zeros([3, N])
pntsBuf = []

Nlines = 0
lines = np.zeros([3, N])
lines[2, :] = 1.0

cont = 0.5
half_dphi = 0.03
tol = 0.1

mess = 0.1
shape = 0

fig = plt.figure()
ax = plt.axes([0.07, 0.25, 0.45, 0.7])

mutex = threading.RLock()

class Line():

    def LMS(X : np.ndarray, Y : np.ndarray, N : int, x_sum, y_sum, xx_sum, yy_sum, xy_sum):
        """sums are of: x, y, xx, yy, xy.\n
        X and Y must have the same length, numpy exception will be thrown if not"""

        phi = xy_sum - x_sum * y_sum / N

        if abs(phi) > 0.0001:
            # Вычисление A для минимумов функции потерь
            theta = (yy_sum - xx_sum) / phi + (x_sum ** 2 - y_sum ** 2) / N / phi
            D = theta ** 2 + 4.0  # дискриминант
            A1 = (theta + math.sqrt(D)) / 2.0
            A2 = (theta - math.sqrt(D)) / 2.0
            # Вычисление С для минимумов функции потерь
            C1 = (y_sum - x_sum * A1) / N
            C2 = (y_sum - x_sum * A2) / N
            # Подстановка в функцию потерь, выявление лучшего

            distsSum1 = np.sum(np.abs(X[:N] * A1 - Y[:N] + C1)) / math.sqrt(A1 ** 2 + 1.0)
            distsSum2 = np.sum(np.abs(X[:N] * A2 - Y[:N] + C2)) / math.sqrt(A2 ** 2 + 1.0)
        else:
            A1 = math.inf  #вертикальная прямая
            A2 = 0.0    #горизонтальная прямая

            C1 = x_sum / N    #вообще говоря, должен быть с минусом
            C2 = y_sum / N
            
            distsSum1 = np.sum(np.abs(X[:N] - C1))
            distsSum2 = np.sum(np.abs(Y[:N] - C2))

        # Выбор наименьшего значения функции потерь, возврат соответствующих ему параметров А и С
        if distsSum1 < distsSum2:
            return A1, C1, distsSum1 / N
        else:
            return A2, C2, distsSum2 / N

    def __init__(self, qOk = 0.005):
        self.line = np.zeros([2])
        """A and C"""

        self.isGap = True
        self.isSingle = False

        self.qOk = qOk
        self._q = qOk

        self._sums = ()

    def copy(self):
        line = Line()
        line.line[0], line.line[1], line.isGap, line.isSingle, line.qOk, line._q = \
            self.line[0], self.line[1], self.isGap, self.isSingle, self.qOk, self._q
        line._sums = self._sums #immutable
        return line

    def set_as_tangent_with_1_pnt(self, p : np.ndarray):
        if p[1] != 0.0:
            self.line[0] = -p[0] / p[1]
            self.line[1] = (p[0] * p[0] + p[1] * p[1]) / p[1]
        else:
            self.line[0] = math.inf
            self.line[1] = p[0]
        self.isSingle = True

    def set_with_2_pnts(self, p1 : np.ndarray, p2 : np.ndarray):
        dx = p2[0] - p1[0]
        if dx != 0.0:
            self.line[0] = (p2[1] - p1[1]) / dx
            self.line[1] = p1[1] - self.line[0] * p1[0]
        else:
            self.line[0] = math.inf
            self.line[1] = p1[0] #без минуса удобнее
        self.isSingle = False

    def set_best_with_LMS(self, pnts : np.ndarray):
        """Движемся по одной точке назад до первого увеличения q (функции ошибки) или до в целом норм вписывания.\n
        Возвращает смещение с конца (<=0 !!!) до последней точки, взятой в прямую."""
        N = pnts.shape[1]

        X = pnts[0, :]
        Y = pnts[1, :]

        sums = np.sum(X), np.sum(Y), np.dot(X, X), np.dot(Y, Y), np.dot(X, Y)

        self.line[0], self.line[1], self._q = Line.LMS(X, Y, N, *sums)

        shift = 0
        while self._q >= self.qOk:
            shift -= 1
            sums = sums[0] - X[shift],  sums[1] - Y[shift],  sums[2] - X[shift] * X[shift],  sums[3] - Y[shift] * Y[shift],  sums[4] - X[shift] * Y[shift]
            A, B, q = Line.LMS(X, Y, N + shift, *sums)
            if (q > self._q):
                shift += 1
                break
            else:
                self.line[0], self.line[1], self._q = A, B, q

        self.isSingle = False
        return shift

    def get_distance_to_pnt(self, p : np.ndarray, signed = False):
        if signed:
            if not math.isinf(self.line[0]):
                return (self.line[0] * p[0] - p[1] + self.line[1]) / math.sqrt(self.line[0] ** 2 + 1)
            else:
                return (p[0] - self.line[1])
        else:
            if not math.isinf(self.line[0]):
                return abs(self.line[0] * p[0] - p[1] + self.line[1]) / math.sqrt(self.line[0] ** 2 + 1)
            else:
                return abs(p[0] - self.line[1])

    def get_projection_of_pnt(self, p : np.ndarray, pout : np.ndarray):
        if not math.isinf(self.line[0]):
            pout[0] = (p[0] + self.line[0] * p[1] - self.line[0] * self.line[1]) / (self.line[0] ** 2 + 1.0)
            pout[1] = self.line[0] * pout[0] + self.line[1]
        else:
            pout[0], pout[1] = self.line[1], p[1]

    def get_projection_of_pnt_Ex(self, p : np.ndarray, pout : np.ndarray, half_dphi, direction):
        """Для линий, построенных по одной точке (self.isSingle = True) и считающихся касательными, находит точку,
        отстоящую влево или вправо (direction = True или False, соответственно) от точки касания на угол half_dphi"""
        if (not self.isSingle):
            if (not math.isinf(self.line[0])):
                pout[0] = (p[0] + self.line[0] * p[1] - self.line[0] * self.line[1]) / (self.line[0] ** 2 + 1.0)
                pout[1] = self.line[0] * pout[0] + self.line[1]
            else:
                pout[0], pout[1] = self.line[1], p[1]
        else:
            l = norm(p[:2]) * tan(half_dphi) #находим точку на линии, отстоящую вправо от точки касания на угол half_dphi
            alpha = atan(self.line[0])
            if (direction):
                pout[0], pout[1] = p[0] + l * cos(alpha), p[1] + l * sin(alpha)
            else:
                pout[0], pout[1] = p[0] - l * cos(alpha), p[1] - l * sin(alpha)
    
    def get_intersection(self, line, pout : np.ndarray):
        if (self.line[0] == line.line[0]):
            pout[0], pout[1] = math.inf, math.inf #прямые параллельны, в т.ч. и если обе вертикальные
        else:
            if math.isinf(self.line[0]):
                pout[0], pout[1] = self.line[1], line.line[0] * self.line[1] + line.line[1] #вертикальная исходная
            elif math.isinf(line.line[0]):
                pout[0], pout[1] = line.line[1], self.line[0] * line.line[1] + self.line[1] #вертикальная подставляемая
            else:
                pout[0] = (line.line[1] - self.line[1]) / (self.line[0] - line.line[0])
                pout[1] = self.line[0] * pout[0] + self.line[1]

def getLines(lines : np.ndarray, pnts : np.ndarray, Npnts : int, continuity : float, half_dphi : float, tolerance : float) -> int:
    """#returns the number of the gotten points in lines"""

    half_dphi *= 0.0174532925199432957692369

    fr = 0
    to = 0
    Nlines = 0

    line = Line()
    prev_line = Line()

    while fr < Npnts:

        if (pnts[0, fr] == 0.0 and pnts[1, fr] == 0.0):
            line.isGap = True   #не очень консистентно, но самую-самую малость быстрее
            to = fr + 1
            while (to < Npnts) and (pnts[0, to] == 0.0 and pnts[1, to] == 0.0):
                to += 1
        else:
            line.isGap = False  #не очень консистентно, но самую-самую малость быстрее
            line.set_as_tangent_with_1_pnt(pnts[:, fr])
            to = fr + 1
            if (to < Npnts) and (pnts[0, to] != 0.0 or pnts[1, to] != 0.0) and (norm(pnts[:2, to] - pnts[:2, fr]) <= continuity):   #допуск неразрывности пары точек
                line.set_with_2_pnts(pnts[:, fr], pnts[:, to])
                to += 1
                while (to < Npnts) and (pnts[0, to] != 0.0 or pnts[1, to] != 0.0) and (line.get_distance_to_pnt(pnts[:, to]) <= tolerance): #допуск близости к прямой, когда точек уже больше 2-х
                    to += 1
                    if (not (to - fr) % 2):   #делится на 2
                        line.set_with_2_pnts(pnts[:, fr], pnts[:, fr + (to - fr) // 2])

            if (to - fr > 2):
                to += line.set_best_with_LMS(pnts[:, fr : to])

        if (not line.isGap):

            if (not prev_line.isGap):   #предыдущая есть, можем, попробовать продолжить

                prev_line.get_intersection(line, lines[:, Nlines])  #получаем точку пересечения текущей и предыдудщей
                if (norm(pnts[:2, fr - 1] - lines[:2, Nlines]) > tolerance and norm(pnts[:2, fr] - lines[:2, Nlines]) > tolerance): #точка пересечения далеко, значит, не продолжаем

                    prev_line.get_projection_of_pnt_Ex(pnts[:, fr - 1], lines[:, Nlines], half_dphi, False)  #закрываем предыдущую
                    Nlines += 1

                    if (norm(pnts[:2, fr] - pnts[:2, fr - 1]) > continuity): #если необходимая непрерывность нарушена, добавляем неявный пробел
                        lines[:2, Nlines] = 0.001
                        Nlines += 1

                    prev_line.isGap = True

                else:
                    Nlines += 1

            if (prev_line.isGap):   #открываем текущую
                line.get_projection_of_pnt_Ex(pnts[:, fr], lines[:, Nlines], half_dphi, True)
                Nlines += 1

            if (to >= Npnts): #закрываем последнюю
                line.get_projection_of_pnt_Ex(pnts[:, to - 1], lines[:, Nlines], half_dphi, False)
                Nlines += 1

        else:

            if (not prev_line.isGap): #сперва закрываем предыдущую, если она есть
                prev_line.get_projection_of_pnt_Ex(pnts[:, fr - 1], lines[:, Nlines], half_dphi, False) 
                Nlines += 1

            if (fr == 0 or to >= Npnts or norm(pnts[:2, to] - pnts[:2, fr - 1]) > continuity):
                lines[:2, Nlines] = 0.0 #добавляем пробел
                Nlines += 1


        prev_line = line.copy()
        fr = to
    
    return Nlines

def firstPnt(pnts : np.ndarray) -> None:
    pnts[0, 0] = 50.0 + 0.5 * random.rand() - 0.25
    pnts[1, 0] = 50.0 + 0.5 * random.rand() - 0.25
    pnts[2, 0] = 2.0 * math.pi * random.rand() - math.pi

def createPnts(pnts : np.ndarray, N, d0 = 0.1, shape = 0, mess = 0.1) -> None:
    global pntsBuf

    i_ang = 0
    deltaAng = 0.2 * random.rand() - 0.

    for i in range(1, N):
        d = d0 * (1 + random.randn() * mess)
        pnts[0, i] = pnts[0, i - 1] + d * cos(pnts[2, i - 1])
        pnts[1, i] = pnts[1, i - 1] + d * sin(pnts[2, i - 1])

        if (shape == 0):    #polyline
            if (random.rand() > 1 - 5.0 / N): # 5 fractures in average
                pnts[2, i] = pnts[2, i - 1] + math.pi * random.rand() - math.pi/2
                i_ang = i
            else:
                pnts[2, i] = pnts[2, i_ang] * (1 + random.randn() * mess)
        elif (shape == 1):  #circle
            pnts[2, i] = pnts[2, i - 1] + deltaAng

    pntsBuf = pnts[:2, :].copy()

def drawLoad(xlim = (46, 54), ylim = (46, 54)):

    ax.cla()

    ax.set(xlim = xlim, ylim = ylim)
    ax.set_aspect('equal')

    ax.scatter(pnts[0, 0], pnts[1, 0], s = 30, marker = 'o', Color = 'red')
    ax.scatter(pnts[0, 1:], pnts[1, 1:], s = 30, marker = 'o', Color = 'gray')

    # ax.plot(lines[0, :Nlines], lines[1, :Nlines], linewidth = 4.0)

    v = 0
    while v < Nlines:
        
        u = v

        while (v < Nlines and (lines[0, v] > 0.01 or lines[1, v] > 0.01)):
            v += 1

        if (v != u):
            ax.plot(lines[0, u : v], lines[1, u : v], linewidth = 4.0)

        if (v < (Nlines - 1) and lines[0, v] != 0.0 and lines[1, v] != 0.0):
            ax.plot([lines[0, v - 1], lines[0, v + 1]], [lines[1, v - 1], lines[1, v + 1]], color = 'black', linewidth = 4.0)

        v += 1
    
    fig.canvas.draw_idle()

def nextPnts(event):
    global Nlines
    with mutex:
        firstPnt(pnts)

        createPnts(pnts, N, shape = shape, mess = mess)
        Nlines = getLines(lines, pnts, N, cont, half_dphi, tol)

        drawLoad()

def updatePnts(val):
    global Nlines, mess
    with mutex:
        mess = val
        createPnts(pnts, N, shape = shape, mess = mess)
        Nlines = getLines(lines, pnts, N, cont, half_dphi, tol)
        drawLoad()

def updateLinesTolerance(val):
    global Nlines, tol
    with mutex:
        tol = val
        Nlines = getLines(lines, pnts, N, cont, half_dphi, tol)
    
    drawLoad(ax.get_xlim(), ax.get_ylim())

def updatePntsShape(event):
    global Nlines, shape
    with mutex:
        shape += 1
        if shape > 1:
            shape = 0
        createPnts(pnts, N, shape = shape, mess = mess)
        Nlines = getLines(lines, pnts, N, cont, half_dphi, tol)
        drawLoad()

jit = False
def jitter(event):
    global jit
    def foo():
        global Nlines
        rns = np.zeros([2, N])
        while jit and plt.get_fignums():
            with mutex:
                rns[:] = 0.0
                for i in range(N):
                    if random.rand() > 0.9:
                        rns[:2, i] = 0.5 * random.rand(2) - 0.25

                pnts[:2, :N] = pntsBuf + 0.02 * random.rand(2, N) - 0.01 + rns

                Nlines = getLines(lines, pnts, N, cont, half_dphi, tol)
                drawLoad(ax.get_xlim(), ax.get_ylim())
            time.sleep(0.5)

    with mutex:
        jit = not jit
        threading.Thread(target=foo).start()

def main():

    global Nlines
    
    firstPnt(pnts)
    createPnts(pnts, N, shape = shape, mess = mess)

    Nlines = getLines(lines, pnts, N, cont, half_dphi, tol)
    drawLoad()

    ax1 = plt.axes([0.15, 0.17, 0.45, 0.03])
    ax2 = plt.axes([0.15, 0.14, 0.45, 0.03])
    ax3 = plt.axes([0.55, 0.28, 0.1, 0.04])
    ax4 = plt.axes([0.55, 0.35, 0.1, 0.04])
    ax5 = plt.axes([0.55, 0.42, 0.1, 0.04])

    sz1 = Slider(ax1, 'tolerance', 0.0, 0.8, tol, valstep = 0.02)
    sz1.on_changed(updateLinesTolerance)

    sz2 = Slider(ax2, 'mess', 0.0, 1.0, mess, valstep = 0.02)
    sz2.on_changed(updatePnts)

    btn1 = Button(ax3, 'Jitter', hovercolor='0.975')
    btn1.on_clicked(jitter)

    btn2 = Button(ax4, 'Shape', hovercolor='0.975')
    btn2.on_clicked(updatePntsShape)

    btn3 = Button(ax5, 'Next', hovercolor='0.975')
    btn3.on_clicked(nextPnts)

    plt.show()

if __name__ == "__main__":
    main()
