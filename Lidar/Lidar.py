import sys
import serial
import time
import threading
import numpy as np
from numpy.linalg import norm as norm 
import math
from math import inf as inf
from math import pi as pi
from math import tan as tan
from math import sin as sin
from math import cos as cos
from math import atan as atan
from math import atan2 as atan2
from RoboMath import*   #always needed
from Config import*     #always needed

import StraightLineFilter as SLF

__all__ = ['Lidar']

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
        self.mountPhi = float(mainDevicesPars["lidarMountPhi"])
        self.phiFrom = 180.0 - (float(mainDevicesPars["lidarPhiTo"]) - self.mountPhi - self.half_dphi)
        """on this lidar counterclockwise is positive and 180.0 deg. shifted angle"""
        self.phiTo = 180.0 - (float(mainDevicesPars["lidarPhiFrom"]) - self.mountPhi - self.half_dphi)
        """on this lidar clockwise is positive and 180.0 deg. shifted angle"""
        self.deep = 5.0

        self.N = round(1.1 * (self.phiTo - self.phiFrom) / (self.half_dphi * 2.0) ) + 40

        self.phiFromPositive = True
        if (self.phiFrom < 0.0):
            self.phiFrom += 360.0
            self.phiFromPositive = False
        if (self.phiTo < 0.0):
            self.phiTo += 360
        
        #PRIVATE
        # self._rphi = np.zeros([2, self.N])
        self._xy = np.zeros([3, self.N])
        self._xy[2, :] = 1.0
        self._phi = np.zeros([self.N])
        self._NinRound = 0

        self._dx_dy_alpha = np.zeros([3])
        self._dx_dy_alphaE = np.zeros([3])

        self._linesXY = np.zeros([3, self.N])
        self._linesXY[2, :] = 1.0
        self._Nlines = 0

        #PUBLIC
        self.xy = np.zeros([3, self.N])
        self.xy[2, :] = 1.0
        #self.dxy = np.zeros([2, self.N])
        self.dx_dy_alpha = np.zeros([3])
        #self.dx_dy_alphaE = np.array([0.0, 0.0, 0.0])
        self.LT = np.zeros([4, 4])

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
                if (dist):
                    return 57.295779513 * math.atan(0.0218 / dist - 0.14037347)  #in degrees
                else:
                    return 0.0

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

                            if (self._xy[0, 0] == 0.0 and self._xy[1, 0] == 0.0):   #лидар отвратительно измеряет углы, если нет дистанции, а нач. и кон. угол очень важны
                                self._phi[0] = 0.0174532925199432957692369 * (180.0 - (self.phiFrom - self.mountPhi))
                            if (self._xy[0, n - 1] == 0.0 and self._xy[1, n - 1] == 0.0):
                                self._phi[self._Nlines - 1] = 0.0174532925199432957692369 * (180.0 - (self.phiTo - self.mountPhi))

                            self._Nlines = SLF.getLines(self._linesXY, self._xy, self._phi, n, deep = self.deep, continuity = 0.6, half_dphi = 2.0 * self.half_dphi, tolerance = 0.1)

                            xlim = self.ax.get_xlim()
                            ylim = self.ax.get_ylim()

                            self.ax.cla()

                            self.ax.set(xlim = xlim, ylim = ylim)
                            self.ax.set_aspect('equal')

                            self.ax.scatter(self._xy[0, :n], self._xy[1, :n], s = 30, marker = 'o', color = 'gray')

                            self.ax.plot([0.0, self._linesXY[0, 0]], [0.0, self._linesXY[1, 0]], color = 'black', linewidth = 4.0)

                            v = 0
                            while v < self._Nlines:
                                
                                u = v

                                while (v < self._Nlines and (abs(self._linesXY[0, v]) > 0.01 or abs(self._linesXY[1, v]) > 0.01)):
                                    v += 1

                                if (v > u + 1):
                                    self.ax.plot(self._linesXY[0, u : v], self._linesXY[1, u : v], linewidth = 4.0)

                                if (v < self._Nlines): 
                                    if (self._linesXY[0, v] != 0.0 and self._linesXY[1, v] != 0.0):
                                        self.ax.plot([self._linesXY[0, v - 1], self._linesXY[0, v + 1]], [self._linesXY[1, v - 1], self._linesXY[1, v + 1]], color = 'black', linewidth = 4.0)
                                else:
                                    self.ax.plot([self._linesXY[0, v - 1], 0.0], [self._linesXY[1, v - 1], 0.0], color = 'black', linewidth = 4.0)

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



                    if (self.phiFromPositive and (angles[i] >= self.phiFrom and angles[i] < self.phiTo)) or \
                        (not self.phiFromPositive and (angles[i] >= self.phiFrom or angles[i] < self.phiTo)):

                        self._phi[n] = 0.0174532925199432957692369 * (180.0 - (angles[i] - self.mountPhi))

                        if (dists[i] > 0.1 and dists[i] < self.deep):    #it is the limit of lidar itself
                            self._xy[0, n] = dists[i] * cos(self._phi[n]) + self.lidarLC[0]
                            self._xy[1, n] = dists[i] * sin(self._phi[n]) + self.lidarLC[1]
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