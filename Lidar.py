import sys
import serial
import time
import threading
import numpy as np
from numpy.linalg import norm as norm 
import math
from math import tan as tan
from math import sin as sin
from math import cos as cos
from math import atan as atan
from math import atan2 as atan2
from RoboMath import*   #always needed
from Config import*     #always needed

__all__ = ['Lidar']

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

class Lidar():
    """
        SINGLE LIDAR OBJECT (non static, for instances use Create function).\n
        Each instance executes in new thread (when Start call)
    """

    _rndKey = 9846    #it is just a random number
    _IDs = 0

    def __init__(self, rndKey, lidarSN, ax, fig):
        "Use create function only!!!"

        if (rndKey != Lidar._rndKey):
            print("Use Lidar.Create function only!!!", exc = True)

        self.ax = ax
        self.fig = fig

        #PORT
        self.ser = serial.Serial(         
            baudrate = 128000,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            exclusive = True,
            timeout = 1.0
            )

        #IDs
        self.lidarSN = 'ffffffff'
        self.lidarID = -1

        #MAIN DATA
        self.half_dphi = 0.3
        """in degrees"""
        self.phiFrom = float(mainDevicesPars["lidarPhiFrom"])
        self.phiTo = float(mainDevicesPars["lidarPhiTo"])
        self.mountPhi = float(mainDevicesPars["lidarMountPhi"])
        """counterclockwise is positive"""

        self.N = round(1.1 * (self.phiTo - self.phiFrom) / (self.half_dphi * 2.0) ) + 40
        
        self._rphi = np.zeros([2, self.N])
        self._xy = np.zeros([3, self.N])
        self._xy[2, :] = 1.0
        self._NinRound = 0
        self._dx_dy_alpha = np.zeros([3])
        self._dx_dy_alphaE = np.zeros([3])

        self._Nlines = 0
        self._lines = np.zeros([3, self.N])
        self._lines[2, :] = 1.0

        #PUBLIC
        self.xy = np.zeros([3, self.N])
        self.xy[2, :] = 1.0
        #self.dxy = np.zeros([2, self.N])
        self.dx_dy_alpha = np.zeros([3])
        #self.dx_dy_alphaE = np.array([0.0, 0.0, 0.0])
        self.LT = np.zeros([4, 4])
        self.GC_v = np.zeros([4])

        #LOCKS
        self._thread = threading.Thread()
        self._lockXY = threading.Event()
        self._mutexXY = threading.Lock()
        self._mutex_dx_dy_alpha = threading.Lock()

        #EVENTS AND HANDLES
        self.ready = threading.Event()
        self.ready.clear()
        self._isFree = threading.Event()
        self._isFree.set()
        
        #IDs initialization
        self.lidarID = Lidar._IDs
        Lidar._IDs += 1
        self.lidarSN = lidarSN

        self.lidarLC = np.zeros([2])    #local coordinates
        for i, el in enumerate(mainDevicesPars["lidarLC_" + str(self.lidarID)].split()):
            if i > 1:
                break
            self.lidarLC[i] = float(el)

    def _Start_wrp(self, f):
        def func(*args, **kwargs):
            if (self._isFree.wait(timeout = 0.25)):  #for the case of delayed stopping
                self._thread = threading.Thread(target = f, args = args, kwargs = kwargs)
                self._thread.start()
                while self._thread.isAlive() and not self.ready.is_set():
                    time.sleep(0.01)
                if self.ready.is_set():
                    time.sleep(0.5) #wait for rotation to become stable
                self.Get_dx_dy_alpha() #to prevent the first guaranted miss
            return self.ready.is_set()
        return func

    def Start(self):
        """
            Starts the lidar forever.\n
            Gets wrapped by self._Start_wrp func which invokes for that a new thread and returns True if succeed, otherwise returns False.\n
            If pinsProcessing = False it blocks pins transform calculations. Memory for pins was allocated while creating, so you can change processing flag further on using Stop/Start\n
            Safe for multiple call.
        """

        self._isFree.clear()

        try:
            port_n = 7
            portDetected = False
            while not portDetected: #finding port with Lidar
                if (port_n > 8):
                    print(f"Cannot find Lidar {self.lidarID} ({self.lidarSN}) on COM", log = True, exc = True)
                try:
                    self.ser.port='COM' + str(port_n)
                    if self.ser.is_open:
                        raise Exception(msg='Busy port')
                    else:
                        self.ser.open()

                    portDetected = True
                    self.ser.write(b'\xa5\x65')
                    self.ser.flush()
                    self.ser.reset_input_buffer()
                    time.sleep(0.1)
                    self.ser.reset_input_buffer()   #old data confusely remains on the device
                    time.sleep(0.1)
                    self.ser.reset_input_buffer()
                    self.ser.write(b'\xa5\x90')
                    self.ser.flush()
                    t0 = time.time()
                    while(not self.ser.read(11) or self.ser.read(16).hex() != self.lidarSN):
                        if (time.time() - t0 > 0.3):
                            portDetected = False
                            self.ser.close()
                            break
                except:
                    portDetected = False
                finally:
                    port_n += 1
            #Время на раскручивание головки. Сразу после открытия порта (это автоматически запускает съемку и идут полезные пакеты облаков) идет команда остановить скан (головка продолжает вращаться), 
            #потом через время команда запустить скан и пропускаются 19 байт (7 + 12), относящиеся к ответу на запуск. 
            self.ser.write(b'\xa5\x65')
            self.ser.flush()
            time.sleep(0.2)
            self.ser.reset_output_buffer()
            self.ser.reset_input_buffer()
            self.ser.write(b'\xa5\x60')  
            skippedBytes = self.ser.read(19) #Далее идут полезные пакеты с облаками

            self.ready.set()

            def AngCorrect(dist):
                cor = 0.0
                if (dist):
                    cor = 57.295779513 * math.atan(0.0218 / dist - 0.14037347)  #in degrees
                return cor

            angles = np.zeros(self.N // 2)
            dists = np.zeros(self.N // 2)
            
            self._dx_dy_alpha[:] = 0.0
            diffAngle = 0.0
            n = 0

            while (self.ready.is_set() and threading.main_thread().is_alive()):

                t0 = time.time()
                while(self.ser.read() != b'\xaa' or self.ser.read() != b'\x55'):
                    n = 0   #если случился проскок, данные должны записываться сначала - и то не факт!
                    # print('LIDAR DATA SLIP', log = True)
                    if (time.time() - t0 > self.ser.timeout):
                        print(f"Lidar {self.lidarID} does not response", log = True, exc = True)

                mode = self.ser.read()[0]    #можно использовать для определения круга (где-то на 346 градусах почему-то выставляется 1)
                length = self.ser.read()[0]
                angles[0] = (int.from_bytes(self.ser.read(2), 'little') >> 1) / 64.0
                angles[length - 1] = (int.from_bytes(self.ser.read(2), 'little') >> 1) / 64.0
                checkCode = self.ser.read(2)
                buff = self.ser.read(2 * length)
                for i in range(length):
                    dists[i] = int.from_bytes(buff[2*i : 2*(i+1)], 'little') / 4000.0

                if (n + length >= self.N):   #предупрежжение выхода за пределы массива, размеченного под облако
                    n = 0 
                    continue

                if (length > 1):
                    if (angles[length - 1] < angles[0]):
                        diffAngle = (angles[length - 1] + 360.0 - angles[0]) / (length - 1)
                    else:
                        diffAngle = (angles[length - 1] - angles[0]) / (length - 1)
                else:
                    diffAngle = 0.0

                isRound = False
                angles0 = angles[0]
                for i in range(length):
                    angles[i] = diffAngle * i + angles0 + AngCorrect(dists[i])

                    if (angles[i] > 360.0):
                        angles[i] -= 360.0
                        
                        #ROUND
                        if (not isRound and n): #возможен пропуск окончания круга, если проход оборота (360 град.) попадает аккурат между пакетами, но такого пока замечено не было
                            #!!!Нужно помнить, что начало круга из-за коррекции угла может весьма ощутимо плавать - до + нескольких градусов к желаемому PhiFrom

                            self._Nlines = getLines(self._lines, self._xy, n, 0.6, 2.0 * self.half_dphi, tolerance = 0.1)

                            xlim = self.ax.get_xlim()
                            ylim = self.ax.get_ylim()

                            self.ax.cla()

                            self.ax.set(xlim = xlim, ylim = ylim)
                            self.ax.set_aspect('equal')

                            self.ax.scatter(self._xy[0, :n], self._xy[1, :n], s = 10, marker = 'o', color = 'gray')

                            # self.ax.plot(self._lines[0, :self._Nlines], self._lines[1, :self._Nlines], linewidth = 4.0, color = 'red')

                            v = 0
                            while v < self._Nlines:
                                
                                u = v

                                while ((self._lines[0, v] > 0.01 or self._lines[1, v] > 0.01) and v < self._Nlines):
                                    v += 1

                                if (v != u):
                                    self.ax.plot(self._lines[0, u : v], self._lines[1, u : v], linewidth = 4.0)

                                if (v < (self._Nlines - 1) and self._lines[0, v] != 0.0 and self._lines[1, v] != 0.0):
                                    self.ax.plot([self._lines[0, v - 1], self._lines[0, v + 1]], [self._lines[1, v - 1], self._lines[1, v + 1]], color = 'black', linewidth = 4.0)

                                v += 1

                            self.fig.canvas.draw_idle()

                            # #DATA LOAD AND PREPARING
                            # with self._mutexXY:
                            #     if (not self._lockXY.is_set()):
                            #         self._NinRound = n
                            #         np.copyto(self.xy, self._xy)

                            # self._mutex_dx_dy_alpha.acquire()

                            # #WORK HERE

                            # self._mutex_dx_dy_alpha.release()

                            n = 0
                            isRound = True

                    if (angles[i] >= self.phiFrom - self.half_dphi and angles[i] < self.phiTo - self.half_dphi):
                        if (dists[i] > 0.1 and dists[i] < 5.0):    #it is the limit of lidar itself
                            self._xy[0, n] = dists[i] * -cos(0.0174532925199432957692369 * (angles[i] - self.mountPhi)) + self.lidarLC[0]
                            self._xy[1, n] = dists[i] * sin(0.0174532925199432957692369 * (angles[i] - self.mountPhi)) + self.lidarLC[1]
                            # self._rphi[0, n] = norm(self._xy[:2, n])
                            # self._rphi[1, n] = atan2(self._xy[1, n], self._xy[0, n])
                        else:
                            self._xy[:2, n] = 0.0
                            # self._rphi[:2, n] = 0.0
                        n += 1
        
        except:
            if (sys.exc_info()[0] is not RoboException):
                print('Lidar ' + str(self.lidarID) + ' error! ' + str(sys.exc_info()[1]) + '; line: ' + str(sys.exc_info()[2].tb_lineno), log = True)
        finally:
            if self._mutex_dx_dy_alpha.locked():
                self._mutex_dx_dy_alpha.release()
            if self._mutexXY.locked():
                self._mutexXY.release()
            self.ready.clear()
            if self.ser.is_open:
                # print('Lidar port closing')
                self.ser.close()
            self._isFree.set()
    
    def Stop(self):
        '''Stops the lidar and waits for the port to be stopped.'''
        # ret = 0
        # if self.GetXY() < 0:
        #     ret -= 1
        # if self.Get_dx_dy_alpha() < 0:
        #     ret -= 2
        self.ready.clear()
        #NO NEED TO WAIT TILL IT REALLY STOPS, THERE IS A SMALL TIMEOUT FOR THE CLOSE NEXT START  
        # while (self.ser.is_open):   
        #     time.sleep(0.0001)
        # return ret

    def Get_dx_dy_alpha(self, default_dx_dy_alpha = np.zeros([3])):
        '''Returns non-missing status or -1 if getting data failed.\n
        Default dx_dy_alpha is np.zeros([0.0, 0.0, 0.0]) (if status is 0).\n
        Do not forget about copying default if neccessary.\n
        l in meters, alpha in radians.'''

        with self._mutex_dx_dy_alpha:    #let there will be no exceptions next    
            ret = 0
            if not self.ready.is_set():
                ret = -1
                
            if ret > 0:
                np.copyto(self.dx_dy_alpha, self._dx_dy_alpha)
                self._dx_dy_alpha[:] = 0.0
            else:
                np.copyto(self.dx_dy_alpha, default_dx_dy_alpha)
                self._dx_dy_alpha[:] = 0.0
            return ret

    def GetLocalTransform(self, defaultLT = np.diagflat([1.0, 1.0, 1.0, 1.0])):
        '''Returns non-missing status or -1 if getting data failed.\n
        Default LT is diagonal ones.
        Do not forget about copying default if neccessary'''

        with self._mutex_dx_dy_alpha:    #let there will be no exceptions next
        
            ret = 0
            if not self.ready.is_set():
                ret = -1
                
            if ret > 0:
                self.LT = Transform_mat(self._dx_dy_alpha[0], self._dx_dy_alpha[1], 0.0, 0.0, 0.0, self._dx_dy_alpha[2], 1)
                self._dx_dy_alpha[:] = 0.0
            else:
                np.copyto(self.LT, defaultLT)
                self._dx_dy_alpha[:] = 0.0
            return ret

    # def acquireXY(self):
    #     '''Locks the XY sampling process. Call releaseXY() after processing!\n
    #     Returns the number of points in a round or -1 if getting data failed and does not lock.\n
    #     XY in meters'''
    #     self._mutexXY.acquire()
    #     if self.ready.is_set():
    #         self._lockXY.set()
    #         self._mutexXY.release()
    #         return self._NinRound
    #     else:
    #         self._mutexXY.release()
    #         return -1
    
    # def releaseXY(self):
    #     self._lockXY.clear()

    @classmethod
    def Create(cls, lidarSN, ax, fig):
        "Start lidar the first one of any other devices!!!"
        try:
            lidar = Lidar(cls._rndKey, lidarSN, ax, fig)
            lidar.Start = lidar._Start_wrp(lidar.Start)
            return lidar
        except:
            if (sys.exc_info()[0] is not RoboException):
                print('Lidar ' + str(lidar.lidarID) + ' error! ' + str(sys.exc_info()[1]) + '; line: ' + str(sys.exc_info()[2].tb_lineno))
            return None