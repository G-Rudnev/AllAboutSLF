import numpy as np
from numpy.random import rand as rand 
from numpy.random import randn as randn 
import time
import threading
from RealSense import RealSense

import polylineCpp as pCpp

import math
from math import fabs as fabs
from math import pi as pi
_2pi = 2.0 * pi
from math import sqrt as sqrt
from math import sin as sin
from math import cos as cos
from math import tan as tan
from math import atan as atan
from math import atan2 as atan2

from Config import*
from RoboMath import*

import matplotlib.pyplot as plt

class e:

    L2G = None

    half_length = float(globalPars["half_length"])
    half_width = float(globalPars["half_width"])
    length = 2.0 * half_length
    width = 2.0 * half_width
    radius = sqrt(half_length ** 2 + half_width ** 2)
    safety = 1.0 + float(globalPars["safetyBox"]) / 100.0
    safety_half_length = half_length * safety
    safety_half_width = half_width * safety
    safety_length = length * safety
    safety_width = width * safety
    safety_radius = radius * safety

    @classmethod
    def plot(cls, ax, color_ = 'blue'):
        
        obb_pnts = np.array([
        [cls.L2G[0, 2] + cls.L2G[0, 0] * cls.half_length + cls.L2G[0, 1] * cls.half_width, \
            cls.L2G[0, 2] + cls.L2G[0, 0] * cls.half_length - cls.L2G[0, 1] * cls.half_width, \
                cls.L2G[0, 2] - cls.L2G[0, 0] * cls.half_length - cls.L2G[0, 1] * cls.half_width, \
                    cls.L2G[0, 2] - cls.L2G[0, 0] * cls.half_length + cls.L2G[0, 1] * cls.half_width, \
                        cls.L2G[0, 2] + cls.L2G[0, 0] * cls.half_length + cls.L2G[0, 1] * cls.half_width], \
            \
        [cls.L2G[1, 2] + cls.L2G[1, 0] * cls.half_length + cls.L2G[1, 1] * cls.half_width, \
            cls.L2G[1, 2] + cls.L2G[1, 0] * cls.half_length - cls.L2G[1, 1] * cls.half_width, \
                cls.L2G[1, 2] - cls.L2G[1, 0] * cls.half_length - cls.L2G[1, 1] * cls.half_width, \
                    cls.L2G[1, 2] - cls.L2G[1, 0] * cls.half_length + cls.L2G[1, 1] * cls.half_width, \
                        cls.L2G[1, 2] + cls.L2G[1, 0] * cls.half_length + cls.L2G[1, 1] * cls.half_width] \
        ])

        ax.plot(obb_pnts[0, :], obb_pnts[1, :], color = color_, linewidth = 1.0)

class m:

    ##Move modes
    NORMAL = 0
    STOP = 1
    """This is only about the end of the frame - will the robo stop in the end of it or not.\n
    An arc [1.0, 0.5, 0.0] will move the robo for 0.5 m and than stop him, no matter was the robo staying or moving before"""
    JUMP = 2
    """In loop mode the r.ARC_Set[1, it] is a next iterator"""
    HOLD = 3
    """No movements or stop as it is no movements expected, whatever, it's mean: mode = STOP, s = alpha = 0.0"""

    r0 = float(globalPars["r0"])
    alpha0 = pi / 8.0

    mode = STOP
    s, alpha = 0.0, 0.0
    """For backward moving: s - positive, alpha - global oriented.\n
    Actually both parameters consider globally as the length of an arc and the rotation angle, respectively.\n
    Direction defines by s sign."""
    l = 0.0
    """Chord for frame calculations."""
    type = 0
    """Move type by possible route activities"""

    LT = np.eye(4)

class Route:

    #Possible types of route activity
    AS_REQUIRED = 0
    """We have reached the goal, iterate next and go on as route requires"""
    TO_GOAL_CLOSE = 1
    """We are close to the goal"""
    TO_GOAL_FAR_ZERO = 2
    """We are far to the goal and we are going to reach it normally (straight)"""
    TO_GOAL_FAR_POSITIVE = 3
    """We are far to the goal and we are going to reach it from the right"""
    TO_GOAL_FAR_NEGATIVE = 4
    """We are far to the goal and we are going to reach it from the left"""

    def __init__(self, N):
        """
            Sets self.it and self.NSteps to 0, arg N is not a self.NSteps, it is max route length.
        """

        self.L2G0 = np.eye(4)
        """Initial route transformation"""
        self.L2G = np.eye(4)
        """Current route transformation"""

        self.ARC_Set = np.zeros([3, N])
        """[mode, distance (typically), angle (typically)]."""

        self._mode = m.STOP        
        """Current required operational mode"""
        self._s = 0.0          
        """Current required arc length"""
        self._alpha = 0.0     
        """Current required route angle"""

        self.mode = m.STOP        
        """Current operational mode to use"""
        self.s = 0.0          
        """Current arc length to use"""
        self.alpha = 0.0     
        """Current angle to use"""
        self.l = 0.0     
        """Current chord to use"""
        self.direction = True
        """Current direction to use, true - forward, false - backward"""
        self.type = Route.AS_REQUIRED
        """Considered move type"""

        self.NSteps = 0
        self.it = 0

        self.ready = False

    def Calculate(self, eL2G : np.ndarray, prev_type, accuracy, latencyDist, iterateBy = None, eG2L : np.ndarray = None, r00 = m.r0):
            """Calculates current frame to be executed, stores it in self.\n
            eL2G can be either 2D (3x3) or 3D (4x4) transformation.\n
            Returns false if and only if we iterate by iterateBy and reach the end or an error.\n
            accuracy in parts (0..1) of m.r0, the smaller, the accurate.\n
            latencyDist is the distance to goal of the medium smoothness and have to be > 0.0 (the farther the less smoothnes)"""

            if (eG2L is None):
                eG2L = inv(eL2G)

            if (len(eL2G) == 4):
                rLC = np.array([np.dot(eG2L[0, :], self.L2G[:, 3]), np.dot(eG2L[1, :], self.L2G[:, 3])])
            else:
                rLC = np.array([np.dot(eG2L[0, :2], self.L2G[:2, 3]) + eG2L[0, 2], np.dot(eG2L[1, :2], self.L2G[:2, 3]) + eG2L[1, 2]])
            dist = norm(rLC)
            alphaDist = atan2(eL2G[0, 0] * self.L2G[1, 0] - eL2G[1, 0] * self.L2G[0, 0], eL2G[0, 0] * self.L2G[0, 0] + eL2G[1, 0] * self.L2G[1, 0])

            #if the goal reached by position and by angle
            if (iterateBy is not None and dist < 1.05 * r00): # and fabs(alphaDist) <= m.alpha0):

                self.type = Route.AS_REQUIRED
                
                if (not iterateBy.Iterate(self)):     #the end of the route arc set or an error on route
                    return False

                self.mode = self._mode

                if (fabs(self._alpha) > 1e-4):
                    matmul(self.L2G, Transform_mat(self._s * sin(self._alpha) / self._alpha, self._s * (1.0 - cos(self._alpha)) / self._alpha, 0.0, 0.0, 0.0, self._alpha, 1), self.L2G)
                else:
                    matmul(self.L2G, Transform_mat(self._s, 0.0, 0.0, 0.0, 0.0, self._alpha, 1), self.L2G)

                attractionVal = atan(rLC[1] / r00) / 2.0
                
                if (len(eL2G) == 4):
                    rLC = np.array([np.dot(eG2L[0, :], self.L2G[:, 3]), np.dot(eG2L[1, :], self.L2G[:, 3])])
                else:
                    rLC = np.array([np.dot(eG2L[0, :2], self.L2G[:2, 3]) + eG2L[0, 2], np.dot(eG2L[1, :2], self.L2G[:2, 3]) + eG2L[1, 2]])
                dist = norm(rLC)
                alphaDist = atan2(eL2G[0, 0] * self.L2G[1, 0] - eL2G[1, 0] * self.L2G[0, 0], eL2G[0, 0] * self.L2G[0, 0] + eL2G[1, 0] * self.L2G[1, 0])

                if (fabs(alphaDist - _2pi - self._alpha) < fabs(alphaDist - self._alpha)):
                    alphaDist -= _2pi
                elif (fabs(alphaDist + _2pi - self._alpha) < fabs(alphaDist - self._alpha)):
                    alphaDist += _2pi

                if (self.direction):
                
                    if (fabs(alphaDist) > 1e-4):
                        self.alpha = alphaDist + attractionVal * sin(alphaDist) / alphaDist
                    else:
                        self.alpha = alphaDist + attractionVal

                    if (fabs(self.alpha) > 1e-4):
                        self.s = dist * fabs(self.alpha / 2.0 / sin(self.alpha / 2.0))
                    else:
                        self.s = dist

                    if (self.s - self._s > r00):
                        self.s = self._s + r00
                    
                else:
                    
                    if (fabs(alphaDist) > 1e-4):
                        self.alpha = alphaDist - attractionVal * sin(alphaDist) / alphaDist
                    else:
                        self.alpha = alphaDist - attractionVal

                    if (fabs(self.alpha) > 1e-4):
                        self.s = -dist * fabs(self.alpha / 2.0 / sin(self.alpha / 2.0))
                    else:
                        self.s = -dist

                    if (self._s - self.s > r00):
                        self.s = self._s - r00

            #if the goal is not reached we have to get to it
            else:

                if (self.direction):
                    r0 = (r00 + rLC[0]) * accuracy
                else:
                    r0 = (r00 - rLC[0]) * accuracy
                
                if (r0 > r00):
                    r0 = r00
                elif (r0 <= r00 * accuracy):
                    r0 = r00 * accuracy
                
                if (dist < r0):  #WE ARE CLOSE
                                    
                    self.type = Route.TO_GOAL_CLOSE

                    self.mode = self._mode
                    self.alpha = alphaDist / 2.0

                    if ((rLC[0] - dist * cos(self.alpha))**2 + (rLC[1] - dist * sin(self.alpha))**2 <= \
                        (rLC[0] + dist * cos(self.alpha))**2 + (rLC[1] + dist * sin(self.alpha))**2):
                        if (self.direction):    #move direction is only as the route defines
                            l = dist
                        else:
                            l = 0.0
                    else:
                        if (not self.direction):
                            l = -dist
                        else:
                            l = 0.0

                    if (r0 * r0 < ((rLC[0] - l * cos(self.alpha))**2 + (rLC[1] - l * sin(self.alpha))**2) ):
                        l /= 2.0

                    if (fabs(self.alpha) > 1e-4):
                        self.s = l * self.alpha / sin(self.alpha)
                    else:
                        self.s = l      

                    self.alpha *= 2.0
                
                else:   #WE ARE FAR

                    if (prev_type > Route.TO_GOAL_CLOSE):
                        self.type = prev_type   #to calculate related move
                    else:
                        self.type = Route.TO_GOAL_FAR_ZERO

                    self.mode = m.NORMAL

                    if (self.direction):
                        l = r0
                    else:
                        rLC[0] = -rLC[0]
                        rLC[1] = -rLC[1]
                        l = -r0

                    alphaFarEst = atan2(rLC[1], rLC[0])
                    alphaDistAfter = alphaDist - alphaFarEst

                    if (self.type == Route.TO_GOAL_FAR_ZERO):
                        if (alphaDistAfter > 0.9 * pi):
                            self.type = Route.TO_GOAL_FAR_POSITIVE
                        elif (alphaDistAfter < -0.9 * pi):
                            self.type = Route.TO_GOAL_FAR_NEGATIVE
                    elif (fabs(alphaDistAfter) < 0.75 * pi):
                        self.type = Route.TO_GOAL_FAR_ZERO
                    
                    if (self.type == Route.TO_GOAL_FAR_NEGATIVE and alphaDistAfter > 0.0):
                        alphaDistAfter -= _2pi
                    elif (self.type == Route.TO_GOAL_FAR_POSITIVE and alphaDistAfter < 0.0):
                        alphaDistAfter += _2pi

                    if (alphaDistAfter > pi):
                        alphaDistAfter -= _2pi
                    elif (alphaDistAfter < -pi):
                        alphaDistAfter += _2pi

                    latency = latencyDist / (dist + latencyDist)

                    if (dist <= 2.0 * r00):
                        self.alpha = atan2(rLC[1] - (2.0 * (fabs(alphaFarEst) < pi / 2.0) - 1.0) * r0 * alphaDistAfter / pi * latency * (1.0 + (dist / r00 / 2.0) ** 2), rLC[0])
                    else:
                        self.alpha = atan2(rLC[1] - (2.0 * (fabs(alphaFarEst) < pi / 2.0) - 1.0) * r0 * alphaDistAfter / pi * latency * 2.0, rLC[0]) * ((2.0 * r00 / dist) ** latency)

                    if (fabs(self.alpha) > 1e-4):
                        self.s = l * self.alpha / 2.0 / sin(self.alpha / 2.0)
                    else:
                        self.s = l

                # if (self.alpha > 0.0):
                #     if (self.alpha > pi / 3.0):
                #         self.s *= pi / 3.0 / self.alpha
                #         self.alpha = pi / 3.0
                # else:
                #     if (self.alpha < pi / -3.0):
                #         self.s *= pi / -3.0 / self.alpha
                #         self.alpha = pi / -3.0
                    
            return True

    def plot(self, ax):
        ax.scatter(self.L2G[0, 3], self.L2G[1, 3], s = 50.0, c = 'green', alpha = 0.25, linewidths = 1.0)
        ax.plot([self.L2G[0, 3], self.L2G[0, 3] + 0.4 * self.L2G[0, 0]], [self.L2G[1, 3], self.L2G[1, 3] + 0.4 * self.L2G[1, 0]], color = 'green', linewidth = 1.0)

e.L2G = np.eye(3)

r = Route(0)
r.L2G = Transform_mat(-1.0, -0.02, 0.0, 0.0, 0.0, pi, 1)
# r.L2G = Transform2D_mat(6.0 * rand(), 6.0 * rand(), randn() * pi / 2.0, 1)

rA = Route(0)

def foo(rs, ax, fig):

    time.sleep(0.2)

    polyline = Polyline(rs.N)

    id = pCpp.init(polyline, polyline.checkList, polyline.Nlines, (polyline.Nlines[0], ))

    ret_tup_0 = (0, )
    ret_tup_3 = (0, np.zeros([2]), 0.0)

    while plt.get_fignums():

        Nlines = 0

        while Nlines == 0:
######## ВОТ ЭТА ФУНКЦИЯ НУЖНА - грузит самую актуальную версию полилинии с камеры в polyline ############
            Nlines = rs.GetLinesXY(polyline)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        ax.cla()
        
        if (norm(r.L2G[:2, 3] - e.L2G[:2, 2]) < m.r0):
            e.L2G = np.eye(3)

        ax.set(xlim = xlim, ylim = ylim)
        ax.set_aspect('equal')

        polyline.plotLines(ax)
        r.plot(ax)

        pCpp.check_if_obb_intersection(id, e.L2G, e.half_length, e.half_width, 1, Nlines - 1, ret_tup_0)
        pCpp.synchronize(id)

        num = ret_tup_0[0]
        print(num)
        
        if (num > 0):
            color = 'red'
        else:
            color = 'blue'

        e.plot(ax, color)

        fig.canvas.draw_idle()
        
        r.Calculate(e.L2G, r.type, 0.5, 0.1)
        if (fabs(r.alpha) > 1e-4):
            np.matmul(e.L2G, Transform2D_mat(r.s * sin(r.alpha) / r.alpha, r.s * (1.0 - cos(r.alpha)) / r.alpha, r.alpha, 1), e.L2G)
        else:
            np.matmul(e.L2G, Transform2D_mat(r.s, 0.0, r.alpha, 1), e.L2G)

def main():

    fig = plt.figure()
    ax = plt.axes()
    ax.set(xlim = (-2, 2), ylim = (-2, 2))
    ax.set_aspect('equal')

    RS0 = RealSense.Create(0)

    while not RS0.Start():
        time.sleep(1.0)

    threading.Thread(target=foo, args = (RS0, ax, fig)).start()

    plt.show()

    exit()

if __name__ == "__main__":
    main()