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

class Line():

    def LMS(X : np.ndarray, Y : np.ndarray, fr : int, to : int, x_sum, y_sum, xx_sum, yy_sum, xy_sum):
        """sums are of: x, y, xx, yy, xy.\n
        X and Y must have the same length, numpy exception will be thrown if not"""

        N = to - fr
        phi = xy_sum - x_sum * y_sum / N

        if abs(phi) > 0.000001:
            # Вычисление A для минимумов функции потерь
            theta = (yy_sum - xx_sum) / phi + (x_sum ** 2 - y_sum ** 2) / N / phi
            D = theta ** 2 + 4.0  # дискриминант
            A1 = (theta + math.sqrt(D)) / 2.0
            A2 = (theta - math.sqrt(D)) / 2.0
            # Вычисление С для минимумов функции потерь
            C1 = (y_sum - x_sum * A1) / N
            C2 = (y_sum - x_sum * A2) / N
            # Подстановка в функцию потерь, выявление лучшего

            distsSum1 = np.sum(np.abs(X[fr : to] * A1 - Y[fr : to] + C1)) / math.sqrt(A1 ** 2 + 1.0)
            distsSum2 = np.sum(np.abs(X[fr : to] * A2 - Y[fr : to] + C2)) / math.sqrt(A2 ** 2 + 1.0)
        else:

            A1 = inf  #вертикальная прямая
            A2 = 0.0    #горизонтальная прямая

            C1 = x_sum / N    #вообще говоря, должен быть с минусом
            C2 = y_sum / N
            
            distsSum1 = np.sum(np.abs(X[fr : to] - C1))
            distsSum2 = np.sum(np.abs(Y[fr : to] - C2))

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
            self.line[0] = inf
            self.line[1] = p[0]
        self.isSingle = True

    def set_with_2_pnts(self, p1 : np.ndarray, p2 : np.ndarray):
        dx = p2[0] - p1[0]
        if dx != 0.0:
            self.line[0] = (p2[1] - p1[1]) / dx
            self.line[1] = p1[1] - self.line[0] * p1[0]
        else:
            self.line[0] = inf
            self.line[1] = p1[0] #без минуса удобнее
        self.isSingle = False

    def set_with_LMS(self, pnts : np.ndarray, best = True):
        """Если best = True, то сначала движемся по одной точке от конца назад до первого увеличения q (функции ошибки),\n
        а потом аналогично по одной точке от начала вперёд или до в целом норм вписывания.\n
        Возвращает смещение с начала, с конца (<=0 !!!) до наилучше вписываемых точек."""
        N = pnts.shape[1]

        X = pnts[0, :]
        Y = pnts[1, :]

        sums = np.sum(X), np.sum(Y), np.dot(X, X), np.dot(Y, Y), np.dot(X, Y)

        self.line[0], self.line[1], self._q = Line.LMS(X, Y, 0, N, *sums)

        self.isSingle = False

        if best:

            beg = 0
            end = N

            direction = True
            while self._q >= self.qOk:
                if direction:
                    end -= 1
                    sums = sums[0] - X[end],  sums[1] - Y[end],  sums[2] - X[end] * X[end],  sums[3] - Y[end] * Y[end],  sums[4] - X[end] * Y[end]
                else:
                    sums = sums[0] - X[beg],  sums[1] - Y[beg],  sums[2] - X[beg] * X[beg],  sums[3] - Y[beg] * Y[beg],  sums[4] - X[beg] * Y[beg]
                    beg += 1
                A, B, q = Line.LMS(X, Y, beg, end, *sums)
                if (q > self._q):   #если последнее укорочение оказалось хуже, берём предпоследнее
                    if direction:
                        end += 1
                        direction = False
                    else:
                        beg -= 1
                        break
                else:
                    self.line[0], self.line[1], self._q = A, B, q

            return beg, end - N
        else:
            return 0, 0

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
            alpha = atan2(p[1], p[0])   # + pi / 2.0
            if (direction):
                pout[0], pout[1] = p[0] - l * sin(alpha), p[1] + l * cos(alpha) #cos(x) = sin(x + pi/2),  -sin(x) = cos(x + pi/2)
            else:
                pout[0], pout[1] = p[0] + l * sin(alpha), p[1] - l * cos(alpha)
    
    def get_intersection(self, line, pout : np.ndarray):
        if (self.line[0] == line.line[0]):
            pout[0], pout[1] = inf, inf #прямые параллельны, в т.ч. и если обе вертикальные
        else:
            if math.isinf(self.line[0]):
                pout[0], pout[1] = self.line[1], line.line[0] * self.line[1] + line.line[1] #вертикальная исходная
            elif math.isinf(line.line[0]):
                pout[0], pout[1] = line.line[1], self.line[0] * line.line[1] + self.line[1] #вертикальная подставляемая
            else:
                pout[0] = (line.line[1] - self.line[1]) / (self.line[0] - line.line[0])
                pout[1] = self.line[0] * pout[0] + self.line[1]

def getLines(linesXY : np.ndarray, pntsXY : np.ndarray, pntsPhi : np.ndarray, Npnts : int, deep : float, continuity : float, half_dphi : float, tolerance : float) -> int:
    """#returns the number of the gotten points in lines"""

    half_dphi *= 0.0174532925199432957692369

    fr = 0
    to = 0
    ex_fr = 0
    ex_to = 0
    Nlines = 0

    line = Line()
    prev_line = Line()
    ex_line = Line()

    extra = False

    ex_pnt = np.zeros([2])

    while fr < Npnts:

        #ФОРМИРОВАНИЕ СЕГМЕНТА
        if (not extra):
            if (pntsXY[0, fr] == 0.0 and pntsXY[1, fr] == 0.0):
                line.isGap = True   #не очень консистентно, но самую-самую малость быстрее
                to = fr + 1
                while (to < Npnts) and (pntsXY[0, to] == 0.0 and pntsXY[1, to] == 0.0):
                    to += 1
            else:
                line.isGap = False  #не очень консистентно, но самую-самую малость быстрее
                line.set_as_tangent_with_1_pnt(pntsXY[:, fr])
                to = fr + 1
                if (to < Npnts) and (pntsXY[0, to] != 0.0 or pntsXY[1, to] != 0.0) and (norm(pntsXY[:2, to] - pntsXY[:2, fr]) <= continuity):   #допуск неразрывности пары точек
                    line.set_with_2_pnts(pntsXY[:, fr], pntsXY[:, to])
                    to += 1
                    while (to < Npnts) and (pntsXY[0, to] != 0.0 or pntsXY[1, to] != 0.0) and (line.get_distance_to_pnt(pntsXY[:, to]) <= tolerance): #допуск близости к прямой, когда точек уже больше 2-х
                        to += 1
                        if (not (to - fr) % 2):   #делится на 2
                            line.set_with_2_pnts(pntsXY[:, fr], pntsXY[:, fr + (to - fr) // 2])

                if (to - fr > 2):
                    beg, end = line.set_with_LMS(pntsXY[:, fr : to])

                    if beg != 0:    #если от начала точки убрали, начинается добалвение экстра линии
                        ex_line = line.copy()   #ex_line в интеграции используется только если текущая - разрыв
                        ex_fr = fr + beg # начало ex_line
                        ex_to = to + end # начало следующей за ex_line 
                        extra = True
                        if beg == 1: #1 точка сначала выпала
                            line.line[0], line.line[1] = pntsXY[1, fr] / pntsXY[0, fr], 0.0
                            fr = fr - 1
                            to = fr + 1
                        elif beg == 2:   #2 точки
                            line.set_with_2_pnts(pntsXY[:, fr], pntsXY[:, fr + 1])
                            to = fr + 1
                        else:
                            line.set_with_LMS(pntsXY[:, fr : fr + beg], False)
                            to = fr + beg - 1
                    else:
                        to += end

        else:
            line = ex_line.copy()
            fr = ex_fr
            to = ex_to
            extra = False

        #ИНТЕГРАЦИЯ СЕГМЕНТА
        if (not line.isGap):    #текущая - сплошная

            if (not prev_line.isGap):   #безразрывно по точкам 2 линии друг за другом

                prev_line.get_intersection(line, linesXY[:, Nlines])  #получаем точку пересечения текущей и предыдудщей
                step_dist = norm(pntsXY[:2, fr] - pntsXY[:2, fr - 1])
                #Далее как раз outliers filter происходит: если точка пересечения двух прямых ближе и к началу следующей и к концу предыущей, 
                #чем расстояние между этими началом и концом, то берём в набор точку пересечения (написано же полностью обратное условие: x > a or y > b = not (x <= a and y <= b))
                #А вкратце непосредственно читаем: если точка пересечения далеко, не берём её 
                if (norm(pntsXY[:2, fr - 1] - linesXY[:2, Nlines]) > step_dist or norm(pntsXY[:2, fr] - linesXY[:2, Nlines]) > step_dist): 

                    prev_line.get_projection_of_pnt_Ex(pntsXY[:, fr - 1], linesXY[:, Nlines], half_dphi, False)  #закрываем предыдущую
                    Nlines += 1

                    if (step_dist > continuity): #если заданная непрерывность превышена, добавляем неявный разрыв, а если нет, то ломаная просто неразрывно подолжается
                        linesXY[:2, Nlines] = 0.001
                        Nlines += 1

                    prev_line.isGap = True

                else:
                    Nlines += 1

            if (prev_line.isGap):   #открываем текущую ... НАЧАЛО ЛИНИЕЙ ТОЖЕ ОКАЗЫВАЕТСЯ ЗДЕСЬ
                line.get_projection_of_pnt_Ex(pntsXY[:, fr], linesXY[:, Nlines], half_dphi, True)
                Nlines += 1

            if (to >= Npnts): #закрываем последнюю
                line.get_projection_of_pnt_Ex(pntsXY[:, to - 1], linesXY[:, Nlines], half_dphi, False)
                Nlines += 1

        else:   #текущая - разрыв

            if (fr == 0):   #начало с разрыва
                if (to < Npnts):
                    ex_line.line[0], ex_line.line[1] = tan(pntsPhi[0]), 0.0
                    ex_line.get_projection_of_pnt(pntsXY[:, to], linesXY[:, 0])
                    linesXY[:2, 1] = 0.0    #здесь разрыв может быть любой длины
                    Nlines = 2
                else:   #облако пустое
                    linesXY[0, 0], linesXY[1, 0] = deep * cos(pntsPhi[0]), deep * sin(pntsPhi[0])
                    linesXY[:2, 1] = 0.0
                    linesXY[0, 2], linesXY[1, 2] = deep * cos(pntsPhi[Npnts - 1]), deep * sin(pntsPhi[Npnts - 1])
                    Nlines = 3
                    #####   КОНЕЦ   #####
            else:
                #не может быть двух подряд разрывов, только в начале, которое уже отработано, поэтому           
                #всегда сперва закрываем предыдущую
                prev_line.get_projection_of_pnt_Ex(pntsXY[:, fr - 1], linesXY[:, Nlines], half_dphi, False) 
                Nlines += 1

                if (to >= Npnts):
                    linesXY[:2, Nlines] = 0.0 #здесь разрыв также может быть любой длины
                    Nlines += 1

                    ex_line.line[0], ex_line.line[1] = tan(pntsPhi[Npnts - 1]), 0.0
                    ex_line.get_projection_of_pnt(pntsXY[:, fr - 1], linesXY[:, Nlines])
                    Nlines += 1
                    ######  КОНЕЦ   ######

                elif (norm(pntsXY[:2, to] - pntsXY[:2, fr - 1]) > continuity): #разрыв достаточно большой и представляет интерес
                    #здесь до и после есть точка, по ним и будем работать
                        
                    ex_line.line[0], ex_line.line[1] = tan(pntsPhi[fr - 1]), 0.0
                    ex_line.get_projection_of_pnt(pntsXY[:, to], ex_pnt)
                    if (norm(ex_pnt) > norm([pntsXY[:2, fr - 1]]) and (ex_pnt[0] * pntsXY[0, fr - 1] > 0.0 or ex_pnt[1] * pntsXY[1, fr - 1] > 0.0)): #если ex_pnt дальше вдоль взгляда, чем pntsXY; вторая половина компенсирует нахождние с двух сторон от (0, 0)
                        linesXY[:2, Nlines] = 0.001
                        Nlines += 1
                        linesXY[:2, Nlines] = ex_pnt    #так копирует
                        Nlines += 1
                        linesXY[:2, Nlines] = 0.0   #разрыв до или после неявного разрыва может быть любой длины
                        Nlines += 1                            
                    else:   #оба перпендикуляра от краев до противоположных лучей не могут быть "за" краями 
                        ex_line.line[0], ex_line.line[1] = tan(pntsPhi[to]), 0.0
                        ex_line.get_projection_of_pnt(pntsXY[:, fr - 1], ex_pnt)
                        if (norm(ex_pnt) > norm([pntsXY[:2, to]]) and (ex_pnt[0] * pntsXY[0, to] > 0.0 or ex_pnt[1] * pntsXY[1, to] > 0.0)):    #аналогично, что и выше
                            linesXY[:2, Nlines] = 0.0
                            Nlines += 1  
                            linesXY[:2, Nlines] = ex_pnt    #так копирует
                            Nlines += 1
                            linesXY[:2, Nlines] = 0.001
                            Nlines += 1
                        else:
                            linesXY[:2, Nlines] = 0.0
                            Nlines += 1

        prev_line = line.copy()
        fr = to
    
    return Nlines

def firstPnt(pnts : np.ndarray, pntsPhi : np.ndarray) -> None:
    pnts[0, 0] = 50.0 + 0.5 * random.rand() - 0.25
    pnts[1, 0] = 50.0 + 0.5 * random.rand() - 0.25
    pnts[2, 0] = 2.0 * pi * random.rand() - pi
    pntsPhi[0] = atan2(pnts[1, 0], pnts[0, 0])

def createPnts(pnts : np.ndarray, pntsPhi : np.ndarray, pntsBuf : np.ndarray, N, d0 = 0.1, shape = 0, mess = 0.1) -> None:

    i_ang = 0
    deltaAng = 0.2 * random.rand() - 0.1

    for i in range(1, N):
        d = d0 * (1 + random.randn() * mess)
        pnts[0, i] = pnts[0, i - 1] + d * cos(pnts[2, i - 1])
        pnts[1, i] = pnts[1, i - 1] + d * sin(pnts[2, i - 1])

        if (shape == 0):    #polyline
            if (random.rand() > 1 - 5.0 / N): # 5 fractures in average
                pnts[2, i] = pnts[2, i - 1] + pi * random.rand() - pi/2
                i_ang = i
            else:
                pnts[2, i] = pnts[2, i_ang] * (1 + random.randn() * mess)
        elif (shape == 1):  #circle
            pnts[2, i] = pnts[2, i - 1] + deltaAng
        
        pntsPhi[i] = atan2(pnts[1, i], pnts[0, i])

    pntsBuf[:2, :] = pnts[:2, :]

def drawLoad(xlim = (46, 54), ylim = (46, 54)):

    ax.cla()

    ax.set(xlim = xlim, ylim = ylim)
    ax.set_aspect('equal')

    ax.scatter(pntsXY[0, 0], pntsXY[1, 0], s = 30, marker = 'o', color = 'red')
    ax.scatter(pntsXY[0, 1:], pntsXY[1, 1:], s = 30, marker = 'o', color = 'gray')

    ax.plot([0.0, linesXY[0, 0]], [0.0, linesXY[1, 0]], color = 'black', linewidth = 4.0)

    v = 0
    while v < Nlines:
        
        u = v

        while (v < Nlines and (abs(linesXY[0, v]) > 0.01 or abs(linesXY[1, v]) > 0.01)):
            v += 1

        if (v > u + 1):
            ax.plot(linesXY[0, u : v], linesXY[1, u : v], linewidth = 4.0)

        if (v < Nlines): 
            if (linesXY[0, v] != 0.0 and linesXY[1, v] != 0.0):
                ax.plot([linesXY[0, v - 1], linesXY[0, v + 1]], [linesXY[1, v - 1], linesXY[1, v + 1]], color = 'black', linewidth = 4.0)
        else:
            ax.plot([linesXY[0, v - 1], 0.0], [linesXY[1, v - 1], 0.0], color = 'black', linewidth = 4.0)

        v += 1
    
    fig.canvas.draw_idle()

def nextPnts(event):
    global Nlines
    with mutex:
        firstPnt(pntsXY, pntsPhi)

        createPnts(pntsXY, pntsPhi, pntsBuf, N, shape = shape, mess = mess)
        Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)

        drawLoad()

def updatePnts(val):
    global Nlines, mess
    with mutex:
        mess = val
        createPnts(pntsXY, pntsPhi, pntsBuf, N, shape = shape, mess = mess)
        Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
        drawLoad()

def updateLinesTolerance(val):
    global Nlines, tol
    with mutex:
        tol = val
        Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
    
    drawLoad(ax.get_xlim(), ax.get_ylim())

def updatePntsShape(event):
    global Nlines, shape
    with mutex:
        shape += 1
        if shape > 1:
            shape = 0
        createPnts(pntsXY, pntsPhi, pntsBuf, N, shape = shape, mess = mess)
        Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
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

                pntsXY[:2, :N] = pntsBuf + 0.02 * random.rand(2, N) - 0.01 + rns
                pntsPhi[:N] = np.arctan2(pntsXY[1, :N], pntsXY[0, :N])

                Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
                drawLoad(ax.get_xlim(), ax.get_ylim())
            time.sleep(0.1)

    with mutex:
        jit = not jit
        threading.Thread(target=foo).start()

def main():

    global Nlines, ax, fig

    pntsXY = np.zeros([3, N])
    pntsPhi = np.zeros([N])
    pntsBuf = np.zeros([2, N])

    linesXY = np.zeros([3, N])
    linesXY[2, :] = 1.0

    fig = plt.figure()
    ax = plt.axes([0.07, 0.25, 0.45, 0.7])
    
    firstPnt(pntsXY, pntsPhi)
    createPnts(pntsXY, pntsPhi, pntsBuf, N, shape = shape, mess = mess)

    Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
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
