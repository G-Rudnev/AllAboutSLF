
import time
import numpy as np
import lidarVector
from importlib import reload
from random import randint, uniform, normalvariate

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
from matplotlib.widgets import Button


N = 100

pntsXY = [] #теперь можно импортировать, память выделяется только в главном вызове
pntsPhi = []
pntsBuf = []

Nlines = 0
linesXY = []

deep = 5.0
cont = 0.6
half_dphi = 0.3
tol = 0.1

mess = 0.1

fig = []    #и плоттер запускается только в главном вызове
ax = []

curr_id = 0  # типа id девайса, с которого последним приледели данные?)

mutex = threading.RLock()



class Line():
    @staticmethod
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
                A, C, q = Line.LMS(X, Y, beg, end, *sums)
                if (q > self._q):   #если последнее укорочение оказалось хуже, берём предпоследнее
                    if direction:
                        end += 1
                        direction = False
                    else:
                        beg -= 1
                        break
                else:
                    self.line[0], self.line[1], self._q = A, C, q

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
    t0 = time.time()
    #half_dphi *= 0.0174532925199432957692369
    #print(half_dphi)
    #half_dphi = 0.00523599

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

            #print("PYTHON", fr, to)

        else:
            line = ex_line.copy()
            fr = ex_fr
            to = ex_to
            extra = False

        #print("PYTHON", fr, to)

        #ИНТЕГРАЦИЯ СЕГМЕНТА
        if (not line.isGap):    #текущая - сплошная
            
            if (not prev_line.isGap):   #безразрывно по точкам 2 линии друг за другом
                
                prev_line.get_intersection(line, linesXY[:, Nlines])  #получаем точку пересечения текущей и предыдудщей
                interAngle = atan2(linesXY[1, Nlines], linesXY[0, Nlines])
                #Фильтр взятия/исключения точки пересечения подготовленных прямых. Если угол этой точки между углов кревых точек - берём (в код вписано обратное условие)
                #Условие составное, т.к. лидары могут крутиться в разные стороны.
                if ((interAngle > pntsPhi[fr - 1] or interAngle < pntsPhi[fr]) and (interAngle > pntsPhi[fr] or interAngle < pntsPhi[fr - 1])):

                    prev_line.get_projection_of_pnt_Ex(pntsXY[:, fr - 1], linesXY[:, Nlines], half_dphi, False)  #закрываем предыдущую
                    Nlines += 1

                    if (norm(pntsXY[:2, fr] - pntsXY[:2, fr - 1]) > continuity): #если заданная непрерывность превышена, добавляем неявный разрыв, а если нет, то ломаная просто неразрывно подолжается
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
                    
                    #отрабатываем разрывы через контроль нормалей к направлениям краевых лучей
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
    print("PYTHON time", time.time() - t0)
    return Nlines


def create_lidar_pnts(pnts: np.ndarray, pntsPhi: np.ndarray, pntsBuf: np.ndarray, N, mess=0.1, lidar_angle=(pi - 0.001), lidar_ax=(0.0, 0.0)) -> np.ndarray:
    """ генерация лидар-подобных данных для проверки решений """

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


def drawLoad(xlim=(-4.5, 4.5), ylim=(-0.5, 3.5)):
    ''' рисование результатов '''

    ax.cla()

    ax.set(xlim = xlim, ylim = ylim)
    ax.set_aspect('equal')

    ax.scatter(pntsXY[0, 0], pntsXY[1, 0], s = 7, marker = 'o', color = 'red')
    ax.scatter(pntsXY[0, 1:], pntsXY[1, 1:], s = 7, marker = 'o', color = 'gray')
    
    # рисование лучей
    for i in range(N): 
        ax.plot([0.0, pntsXY[0, i]], [0.0, pntsXY[1, i]], color='cyan', linewidth=0.05)

    linewidth = 2
    
    # рисование ломаной
    ax.plot([0.0, linesXY[0, 0]], [0.0, linesXY[1, 0]], color = 'black', linewidth = linewidth)
    #for i in range(Nlines):
    #    ax.plot([linesXY[0][i], linesXY[0][i+1]], [linesXY[1][i], linesXY[1][i+1]], linewidth=2)
    v = 0
    while v < Nlines:
        
        u = v

        while (v < Nlines and (abs(linesXY[0, v]) > 0.01 or abs(linesXY[1, v]) > 0.01)):
            v += 1

        if (v > u + 1):
            ax.plot(linesXY[0, u : v], linesXY[1, u : v], linewidth = linewidth)

        if (v < Nlines): 
            if (linesXY[0, v] != 0.0 and linesXY[1, v] != 0.0):
                ax.plot([linesXY[0, v - 1], linesXY[0, v + 1]], [linesXY[1, v - 1], linesXY[1, v + 1]], color = 'black', linewidth = linewidth)
        else:
            ax.plot([linesXY[0, v - 1], 0.0], [linesXY[1, v - 1], 0.0], color = 'black', linewidth = linewidth)

        v += 1
    
    fig.canvas.draw_idle()
    #fig.show()



#N = 10000000
#pntsXY = np.ones([3, N])
#pntsPhi = np.ones([N]) * 1.5
#linesXY = np.zeros([3, N])

#pntsXY1 = np.ones([3, N])
#pntsPhi1 = np.ones([N]) * 1.5
#linesXY1 = np.zeros([3, N])

#pntsXY2 = np.ones([3, N])
#pntsPhi2 = np.ones([N]) * 1.5
#linesXY2 = np.zeros([3, N])

#id0 = lidarVector.init(pntsXY, pntsPhi, linesXY, N, (0.0, ))
#id1 = lidarVector.init(pntsXY1, pntsPhi1, linesXY1, N, (0.0, ))
#id2 = lidarVector.init(pntsXY2, pntsPhi2, linesXY2, N, (0.0, ))

#t0 = time.time()
#linesXY[0, :] = np.exp(pntsXY[0, :] * pntsXY[0, :] * pntsPhi)
#linesXY[1, :] = np.exp(pntsXY[1, :] * pntsXY[1, :] * pntsPhi)
#print('PY calc time:', time.time() - t0)

#t0 = time.time()
#lidarVector.pushPnts(id0)   #вызовы push и calc не блокирующие
#lidarVector.synchronize(id0)    #только вызов sync блокирует до завершения выполнения функций в очереди по объекту со своим id
#print('CPP Push only id0 time:', time.time() - t0)

#t0 = time.time()
#lidarVector.calcLines(id0)  #если по данному id не идет работа, вызов не блокирует
#lidarVector.synchronize(id0)
#print('CPP Calc id0 time:', time.time() - t0)

#print('lines beg:')
#print(linesXY[:, :1])
#print('lines end:')
#print(linesXY[:, N - 1:])

#print('__NEXT DATA__')
#pntsXY[:] = 2.0 #обновили данные

#t0 = time.time()
#linesXY[0, :] = np.exp(pntsXY[0, :] * pntsXY[0, :] * pntsPhi)
#linesXY[1, :] = np.exp(pntsXY[1, :] * pntsXY[1, :] * pntsPhi)
#print('PY Calc id0 time:', time.time() - t0)

#t0 = time.time()
#lidarVector.pushPnts(id0) #если по данному id не идет работа, вызов не блокирует
#lidarVector.pushPnts(id1)
#lidarVector.synchronize(id0)
#print('CPP Push id0 and id1 async (no sync for id1) time:', time.time() - t0) #он ускоряется при следующих вызовах push, видимо, кэширует

#t0 = time.time()
#lidarVector.calcLines(id0)
#lidarVector.synchronize(id0)
#print('CPP Calc id0 time:', time.time() - t0)

#print('lines beg:')
#print(linesXY[:, :1])
#print('lines end:')
#print(linesXY[:, N - 1:])

#print('__NEXT DATA__')
#pntsXY[:] = 3.0 #обновили данные

#t0 = time.time()
#lidarVector.pushPnts(id0)
#lidarVector.calcLines(id0)
#lidarVector.pushPnts(id1)
#lidarVector.calcLines(id1)
#lidarVector.synchronize(id0)
#lidarVector.synchronize(id1)
#print('CPP Push and Calc id0, id1 async time:', time.time() - t0)

#t0 = time.time()
#lidarVector.pushPnts(id0)
#lidarVector.calcLines(id0)
#lidarVector.pushPnts(id1)
#lidarVector.calcLines(id1)
#lidarVector.pushPnts(id2)
#lidarVector.calcLines(id2)
#lidarVector.synchronize(id0)
#lidarVector.synchronize(id1)
#lidarVector.synchronize(id2)
#print('CPP Push and Calc id0, id1, id2 async time:', time.time() - t0)

#print('lines beg:')
#print(linesXY[:, :1])
#print('lines end:')
#print(linesXY[:, N - 1:])

#plt.show()


def nextPnts(event):
    global Nlines, pntsBuf
    
    Nlines = 0
    
    with mutex:
        # firstPnt(pntsXY, pntsPhi)
        # createPnts(pntsXY, pntsPhi, pntsBuf, N, shape = shape, mess = mess)

        pntsBuf = create_lidar_pnts(pntsXY, pntsPhi, pntsBuf, N)
        #Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
        #t0 = time.time()
        ##print("calc", lidarVector.calcLines(curr_id))
        ##print("synch", lidarVector.synchronize(curr_id))
        #print("Python Nlines PREV", Nlines)
        #lidarVector.calcLines(curr_id)
        #Nlines = lidarVector.synchronize(curr_id)
        #print("Python Nlines NEW", Nlines)
        ##Nlines = lidarVector.calcLines(curr_id)
        ##lidarVector.synchronize(curr_id)
        ##Nlines = linesXY[0][-1]
        #print("Time", time.time() - t0)
        ##print("PYTHON Nlines", Nlines)
        ##print("PYTHON linesXY")
        ##print(linesXY)
        Nlines = getNlines(curr_id)
        print("Nlines CPP", Nlines)

        drawLoad()
        Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
        print("Nlines PYTHON", Nlines)


def getNlines(device_id):
    t0 = time.time()
    #print("Python Nlines PREV", Nlines)
    lidarVector.calcLines(device_id)
    lns_num = lidarVector.synchronize(device_id)
    #lns_num = lidarVector.getNlines(device_id)
    #print(linesXY)
    print("CPP time", time.time() - t0)

    return lns_num


def main():

    global Nlines, ax, fig
    global pntsXY, pntsPhi, pntsBuf, linesXY
    global curr_id

    pntsXY = np.zeros([3, N])
    pntsPhi = np.zeros([N])
    pntsBuf = np.zeros([2, N])

    linesXY = np.zeros([3, N])
    linesXY[2, :] = 1.0

    fig = plt.figure()
    ax = plt.axes([0.07, 0.25, 0.45, 0.7])
    
    pntsBuf = create_lidar_pnts(pntsXY, pntsPhi, pntsBuf, N)
    
    # ======== Блок вот этого вот движения ========
    deep = 5.0;
    continuity = 0.6;
    half_dPhi = 0.3;
    tolerance = 0.1;

    id0 = lidarVector.init(pntsXY, pntsPhi, linesXY, N, (deep, continuity, half_dPhi, tolerance))
    curr_id = id0

    Nlines = getNlines(curr_id)
    print("Nlines CPP", Nlines)
    drawLoad()

    fig.show()

    Nlines = getLines(linesXY, pntsXY, pntsPhi, N, deep, cont, half_dphi, tol)
    print("Nlines PYTHON", Nlines)

    

    #lidarVector.pushPnts(id0)
    #t0 = time.time()
    ##print("calc", lidarVector.calcLines(curr_id))
    ##print("synch", lidarVector.synchronize(curr_id))
    #print("Python Nlines PREV", Nlines)
    #lidarVector.calcLines(curr_id)
    #a = input()
    #lidarVector.synchronize(curr_id)
    #a = input()
    #Nlines = lidarVector.getNlines(curr_id)
    ##lidarVector.synchronize()
    #print("Python Nlines NEW", Nlines)
    #print("Time", time.time() - t0)
    #print("PYTHON Nlines", Nlines)
    #print("PYTHON linesXY")
    #print(linesXY)

    # =============== Конец блока =================

    #ax1 = plt.axes([0.15, 0.17, 0.45, 0.03])
    #ax2 = plt.axes([0.15, 0.14, 0.45, 0.03])
    #ax3 = plt.axes([0.55, 0.28, 0.1, 0.04])
    #ax4 = plt.axes([0.55, 0.35, 0.1, 0.04])
    ax_btn = plt.axes([0.55, 0.42, 0.1, 0.04])

    #sz1 = Slider(ax1, 'tolerance', 0.0, 0.8, tol, valstep = 0.02)
    #sz1.on_changed(updateLinesTolerance)

    #sz2 = Slider(ax2, 'mess', 0.0, 1.0, mess, valstep = 0.02)
    #sz2.on_changed(updatePnts)

    #btn1 = Button(ax3, 'Jitter', hovercolor='0.975')
    #btn1.on_clicked(lidar_jitter)

    #btn2 = Button(ax4, 'Shape', hovercolor='0.975')
    #btn2.on_clicked(updatePntsShape)

    btn = Button(ax_btn, 'Next', hovercolor='0.975')
    btn.on_clicked(nextPnts)

    a = input()

if __name__ == "__main__":
    main()

